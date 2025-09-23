import asyncio
import logging
import multiprocessing
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

import rpyc
import torch
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from fastmoe.utils.hf_transformers_utils import get_tokenizer
from fastmoe.serve.io_struct import (
    BatchTokenIDOut,
    BatchTokenizedGenerateReqInput,
)
from fastmoe.backend.execution_engine import ExecutionEngine
from fastmoe.backend.utils import HardwareConfig
from fastmoe.backend.task import Batch, Req
from fastmoe.backend.model_runner import ModelRunner, _set_default_torch_dtype
from fastmoe.backend.expert_reallocation import (
    ReallocationManager, ExpertMover, ExpertReallocationRequest as ReallocationReq,
    BatchReallocationRequest, ReallocationAction, ReallocationStatus
)
import uuid
from fastmoe.serve.server_args import PortArgs, ServerArgs
from fastmoe.utils.model_config import ModelConfig
from fastmoe.utils.utils import (
    get_exception_traceback,
    get_int_token_logit_bias,
    set_random_seed,
)

logger = logging.getLogger("model_rpc")


class ModelRpcServer(rpyc.Service):
    def exposed_init_model(
        self,
        tp_rank: int,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        server_args, port_args = [obtain(x) for x in [server_args, port_args]]

        # Copy arguments
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.server_args = server_args  # Store server_args for later use

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path, server_args.trust_remote_code
        )
        self.model_runner = ModelRunner(
            self.model_config,
            server_args.mem_fraction_static,
            tp_rank,
            server_args.tp_size,
            port_args.nccl_port,
            server_args.load_format,
            server_args.trust_remote_code,
        )
        
        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self.int_token_logit_bias = torch.tensor(
            get_int_token_logit_bias(self.tokenizer, self.model_config.vocab_size), dtype=torch.float32
        )
        set_random_seed(server_args.random_seed)
        logger.info(
            f"Rank {self.tp_rank}: "
            f"context_len={self.model_config.context_len}, "
        )

        self.hardware_config = HardwareConfig.init(torch.cuda.get_device_name(0), 
                                                   self.model_runner.total_cpu_memory,
                                                   server_args.cpu_mem_bdw,
                                                   server_args.tp_size
                                                   )

        # Init running status
        self.forward_queue: List[Req] = []
        self.running_batch: Batch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval

        self.exe_engine: ExecutionEngine = None
        self.reallocation_manager = None
        self.expert_mover = None

        with _set_default_torch_dtype(torch.float16):
            # Use default values if not provided
            avg_prompt_len = server_args.avg_prompt_len or 77
            gen_len = server_args.gen_len or 32
            self.build_tasks_and_exec_ctx(avg_prompt_len, gen_len)

    def flush_cache(self):
        if len(self.forward_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.exe_engine.reset()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
        else:
            warnings.warn(
                "Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.forward_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )

    def exposed_step(self, recv_reqs):
        if self.tp_size != 1:
            recv_reqs = obtain(recv_reqs)

        try:
            # Recv requests
            for recv_req in recv_reqs:
                if isinstance(recv_req, BatchTokenizedGenerateReqInput):
                    self.handle_batch_generate_request(recv_req)
                    # Execute Graph
                    self.batch_forward_step()
                    # Clean up
                    self.flush_cache()
                else:
                    raise ValueError(f"Invalid request: {recv_req}")

        except Exception:
            logger.error("Exception in ModelRpcClient:\n" + get_exception_traceback())

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret
    
    def build_tasks_and_exec_ctx(self, avg_prompt_len, gen_len):
        # Get KV cache configuration from server args
        use_paged_kv_cache = hasattr(self.server_args, 'use_paged_kv_cache') and self.server_args.use_paged_kv_cache
        kv_cache_config = {
            'kv_cache_size_gb': getattr(self.server_args, 'kv_cache_size_gb', 4.0),
            'kv_block_size': getattr(self.server_args, 'kv_block_size', 16),
            'gpu_fraction': getattr(self.server_args, 'kv_cache_gpu_fraction', 0.2),
            'override_capacity': getattr(self.server_args, 'kv_cache_override_capacity', None)
        }
        self.exe_engine = ExecutionEngine(self.model_runner, self.model_config, self.hardware_config, avg_prompt_len, gen_len, use_paged_kv_cache, kv_cache_config)
        self.exe_engine.init_gpu_experts()
        torch.cuda.synchronize()
        
        # Initialize reallocation components
        self.reallocation_manager = ReallocationManager(max_concurrent_moves=2)
        self.expert_mover = ExpertMover(self.exe_engine, self.exe_engine.expert_tracker)

    @torch.inference_mode()
    def batch_forward_step(self):
        num_mb, abort_requests = self.exe_engine.create_micro_batches(self.forward_queue)
        self.forward_queue = []
        if abort_requests:
            self.handle_aborted_requests(abort_requests)
        self.exe_engine.init_weights_prefetch_meta()

        # prefill
        self.exe_engine.prepare_for_prefill(self.int_token_logit_bias)
        self.exe_engine.prefetch_experts(0)
        for i in range(self.model_config.num_hidden_layers):
            for j in range(self.exe_engine.weights_prefetch_num_pages_cpu):
                for k in range(num_mb):
                    # preattn and attn are done in layer(i, 0, k)
                    self.exe_engine.layer(i, j, k)
                    if j == 0:
                        self.exe_engine.offload_kv_cache(i, k)
                    if num_mb > 1:
                        self.exe_engine.offload_hidden_prefill(k)
                        self.exe_engine.load_hidden_prefill((k + 1) % num_mb)
                if j == self.exe_engine.weights_prefetch_num_pages_cpu - 1:
                    self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, 0)
                else:
                    self.exe_engine.prefetch_experts(i, j + 1, 0)

        # update execution context for decode if necessary
        # decode
        decode_step = 0
        num_decode_steps = self.exe_engine.context.gen_len - 1 if num_mb > 1 else 1
        while decode_step < num_decode_steps:
            print("decode step: ", decode_step)
            self.exe_engine.prepare_for_decode()
            # Prologue
            for k, ub in enumerate(self.exe_engine.micro_batches[:2]):
                self.exe_engine.pre_attention(0, k)
                self.exe_engine.offload_qkv(0, k)
                self.exe_engine.cpu_attention(0, k)
                if self.exe_engine.weights_prefetch_num_pages_cpu > 1:
                    self.exe_engine.prefetch_experts_to_pin(0, k)
                else:
                    self.exe_engine.prefetch_experts_to_pin(1, k)
            
            for i in range(self.model_config.num_hidden_layers):
                j = 0
                for k in range(num_mb):
                    self.exe_engine.load_hidden(i, k)
                    if self.exe_engine.weights_prefetch_num_pages_cpu > 1:
                        # expert-granularity
                        self.exe_engine.prefetch_experts(i, 1, k, stage="decode")
                    else:
                        self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, k, stage="decode")
                    self.exe_engine.post_attention(i, 0, k)
                    if num_mb > 2:
                        if k + 2 < num_mb:
                            self.exe_engine.pre_attention(i, k + 2)
                            self.exe_engine.offload_qkv(i, k + 2)
                            self.exe_engine.cpu_attention(i, k + 2)
                        elif k + 2 >= num_mb and i < self.model_config.num_hidden_layers - 1:
                            self.exe_engine.pre_attention(i + 1, (k + 2) % num_mb)
                            self.exe_engine.offload_qkv(i + 1, (k + 2) % num_mb)
                            self.exe_engine.cpu_attention(i + 1, (k + 2) % num_mb)
                    
                    layer_id, slot_id = self.exe_engine.get_prefetch_e2p_idx(i, j, k, num_mb)
                    if layer_id is not None and slot_id is not None:
                        self.exe_engine.prefetch_experts_to_pin(layer_id, slot_id)
                # self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, 0)
                    
                    
                # for expert-granularity compute
                for j in range(1, self.exe_engine.weights_prefetch_num_pages_cpu):
                    for k in range(num_mb):
                        if j == self.exe_engine.weights_prefetch_num_pages_cpu - 1:
                            self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, k, stage="decode")
                        else:
                            self.exe_engine.prefetch_experts(i, j, k, stage="decode")
                        self.exe_engine.post_attention(i, j, k)
                        
                        layer_id, slot_id = self.exe_engine.get_prefetch_e2p_idx(i, j, k, num_mb)
                        if layer_id and slot_id:
                            self.exe_engine.prefetch_experts_to_pin(layer_id, slot_id)
                            
            decode_step += 1
        
        for ub in self.exe_engine.micro_batches:
            self.handle_finished_requests(ub)
    
    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        for i in range(len(recv_req.rid)):
            req = Req(recv_req.rid[i], recv_req.input_text[i], recv_req.input_ids[i])
            req.sampling_params = recv_req.sampling_params
            req.return_logprob = recv_req.return_logprob
            req.logprob_start_len = recv_req.logprob_start_len
            req.stream = recv_req.stream
            req.tokenizer = self.tokenizer

            # Truncate long prompts
            req.input_ids = req.input_ids[: self.model_config.context_len - 1]
            req.sampling_params.max_new_tokens = min(
                req.sampling_params.max_new_tokens,
                self.model_config.context_len - 1 - len(req.input_ids),
            )
            self.forward_queue.append(req)

    def handle_finished_requests(self, batch: Batch):
        output_rids = []
        output_tokens = []
        output_hit_stop_str = []
        output_skip_special_tokens = []
        output_meta_info = []
        output_finished = []
        finished_indices = []
        for i, req in enumerate(batch.reqs):
            if req.finished:
                finished_indices.append(i)

            if req.finished:
                output_rids.append(req.rid)
                output_tokens.append(req.output_ids)
                output_hit_stop_str.append(req.hit_stop_str)
                output_skip_special_tokens.append(
                    req.sampling_params.skip_special_tokens
                )
                meta_info = {
                    "prompt_tokens": len(req.input_ids),
                    "completion_tokens": len(req.output_ids),
                }
                if req.return_logprob:
                    meta_info["prompt_logprob"] = req.logprob
                    meta_info["normalized_prompt_logprob"] = req.normalized_logprob
                output_meta_info.append(meta_info)
                output_finished.append(req.finished)

        # Send to detokenizer
        if output_rids:
            self.out_pyobjs.append(
                BatchTokenIDOut(
                    output_rids,
                    output_tokens,
                    output_hit_stop_str,
                    output_skip_special_tokens,
                    output_meta_info,
                    output_finished,
                )
            )

        # Clear Batch
        if finished_indices:
            batch.reqs = []
    
    def handle_aborted_requests(self, requests: List[Req]):
        output_rids = []
        output_tokens = []
        output_hit_stop_str = []
        output_skip_special_tokens = []
        output_meta_info = []
        output_finished = []
        for req in requests:
            if req.finished:
                output_rids.append(req.rid)
                output_tokens.append([])
                output_hit_stop_str.append(req.hit_stop_str)
                output_skip_special_tokens.append(
                    req.sampling_params.skip_special_tokens
                )
                meta_info = {
                    "prompt_tokens": len(req.input_ids),
                    "completion_tokens": 0,
                }
                if req.return_logprob:
                    meta_info["prompt_logprob"] = req.logprob
                    meta_info["normalized_prompt_logprob"] = req.normalized_logprob
                output_meta_info.append(meta_info)
                output_finished.append(req.finished)

        # Send to detokenizer
        if output_rids:
            self.out_pyobjs.append(
                BatchTokenIDOut(
                    output_rids,
                    output_tokens,
                    output_hit_stop_str,
                    output_skip_special_tokens,
                    output_meta_info,
                    output_finished,
                )
            )
    
    def exposed_get_expert_locations(self):
        """Get current expert location tracking information."""
        if self.exe_engine is not None:
            return self.exe_engine.get_expert_locations()
        else:
            return {
                "error": "Execution engine not initialized",
                "summary": {},
                "all_experts": []
            }
    
    def exposed_get_layer_expert_summary(self, layer_id: int):
        """Get expert location summary for a specific layer."""
        if self.exe_engine is not None:
            return self.exe_engine.get_layer_expert_summary(layer_id)
        else:
            return {"error": "Execution engine not initialized"}
    
    def exposed_request_expert_reallocation(self, layer_id: int, expert_id: int, 
                                          action: str, priority: int = 0, 
                                          metadata: dict = None):
        """Request reallocation of a single expert."""
        if self.reallocation_manager is None:
            return {"error": "Reallocation manager not initialized"}
        
        try:
            request_id = str(uuid.uuid4())
            request = ReallocationReq(
                request_id=request_id,
                layer_id=layer_id,
                expert_id=expert_id,
                action=ReallocationAction(action),
                priority=priority,
                metadata=metadata or {}
            )
            
            self.reallocation_manager.submit_request(request)
            
            # Start processing in background
            # In a real implementation, this would be handled by a worker thread
            success, error = self._process_reallocation_request(request)
            
            # Update the request status
            request.status = ReallocationStatus.COMPLETED if success else ReallocationStatus.FAILED
            request.error_message = error
            request.completed_at = time.time()
            
            # Move from active to completed if needed
            with self.reallocation_manager._lock:
                if request.request_id in self.reallocation_manager.active_requests:
                    self.reallocation_manager.completed_requests[request.request_id] = request
                    del self.reallocation_manager.active_requests[request.request_id]
            
            return {
                "request_id": request_id,
                "status": "completed" if success else "failed",
                "message": error if error else "Expert reallocation completed",
                "success": success
            }
        except Exception as e:
            return {"error": str(e)}
    
    def exposed_request_batch_expert_reallocation(self, requests: list, atomic: bool = True):
        """Request reallocation of multiple experts."""
        if self.reallocation_manager is None:
            return {"error": "Reallocation manager not initialized"}
        
        try:
            batch_id = str(uuid.uuid4())
            realloc_requests = []
            
            for req in requests:
                realloc_req = ReallocationReq(
                    request_id=str(uuid.uuid4()),
                    layer_id=req["layer_id"],
                    expert_id=req["expert_id"],
                    action=ReallocationAction(req["action"]),
                    priority=req.get("priority", 0),
                    metadata=req.get("metadata", {})
                )
                realloc_requests.append(realloc_req)
            
            batch = BatchReallocationRequest(
                request_id=batch_id,
                requests=realloc_requests,
                atomic=atomic
            )
            
            self.reallocation_manager.submit_batch(batch)
            
            # Process batch
            results = []
            for req in realloc_requests:
                success, error = self._process_reallocation_request(req)
                
                # Update the request status
                req.status = ReallocationStatus.COMPLETED if success else ReallocationStatus.FAILED
                req.error_message = error
                req.completed_at = time.time()
                
                # Move from active to completed if needed
                with self.reallocation_manager._lock:
                    if req.request_id in self.reallocation_manager.active_requests:
                        self.reallocation_manager.completed_requests[req.request_id] = req
                        del self.reallocation_manager.active_requests[req.request_id]
                
                results.append({"expert": f"L{req.layer_id}E{req.expert_id}", 
                              "success": success, "error": error})
            
            return {
                "request_id": batch_id,
                "status": "completed",
                "results": results
            }
        except Exception as e:
            return {"error": str(e)}
    
    def exposed_get_reallocation_status(self, request_id: str):
        """Get the status of a reallocation request."""
        if self.reallocation_manager is None:
            return None
        
        request = self.reallocation_manager.get_request_status(request_id)
        if request:
            return {
                "request_id": request.request_id,
                "status": request.status.value,
                "message": request.error_message or "",
                "layer_id": request.layer_id,
                "expert_id": request.expert_id,
                "action": request.action.value
            }
        return None
    
    def exposed_get_reallocation_stats(self):
        """Get statistics about expert reallocation."""
        if self.reallocation_manager is None:
            return {"error": "Reallocation manager not initialized"}
        
        stats = self.reallocation_manager.get_reallocation_stats()
        
        # Add memory statistics if available
        if self.expert_mover:
            memory_stats = self.expert_mover.get_memory_stats()
            stats["memory"] = memory_stats
            
            # Clean up completed events periodically
            cleaned = self.expert_mover.cleanup_completed_events()
            if cleaned > 0:
                logger.debug(f"Cleaned up {cleaned} completed transfer events")
        
        return stats
    
    def exposed_get_kv_cache_capacity(self):
        """Get the computed KV cache capacity from the execution engine."""
        if self.exe_engine is None:
            return None
        return self.exe_engine._calculate_paged_cache_capacity()
    
    def _process_reallocation_request(self, request: ReallocationReq):
        """Process a single reallocation request."""
        try:
            if request.action == ReallocationAction.MOVE_TO_GPU:
                return self.expert_mover.move_expert_to_gpu(request.layer_id, request.expert_id)
            elif request.action == ReallocationAction.MOVE_TO_CPU:
                return self.expert_mover.move_expert_to_cpu(request.layer_id, request.expert_id)
            else:
                return False, f"Unsupported action: {request.action}"
        except Exception as e:
            return False, str(e)


class ModelRpcClient:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        tp_size = server_args.tp_size

        if tp_size == 1:
            # Init model
            self.model_server = ModelRpcServer()
            self.model_server.exposed_init_model(0, server_args, port_args)

            # Wrap functions
            def async_wrap(f):
                async def _func(*args, **kwargs):
                    return f(*args, **kwargs)

                return _func

            self.step = async_wrap(self.model_server.exposed_step)
            self.get_expert_locations = async_wrap(self.model_server.exposed_get_expert_locations)
            self.get_layer_expert_summary = async_wrap(self.model_server.exposed_get_layer_expert_summary)
            self.request_expert_reallocation = async_wrap(self.model_server.exposed_request_expert_reallocation)
            self.request_batch_expert_reallocation = async_wrap(self.model_server.exposed_request_batch_expert_reallocation)
            self.get_reallocation_status = async_wrap(self.model_server.exposed_get_reallocation_status)
            self.get_reallocation_stats = async_wrap(self.model_server.exposed_get_reallocation_stats)
            self.get_kv_cache_capacity = async_wrap(self.model_server.exposed_get_kv_cache_capacity)
        else:
            with ThreadPoolExecutor(tp_size) as executor:
                # Launch model processes
                rets = executor.map(start_model_process, port_args.model_rpc_ports)
                self.model_servers = [x[0] for x in rets]
                self.procs = [x[1] for x in rets]

                # Init model
                def init_model(i):
                    return self.model_servers[i].init_model(i, server_args, port_args)

                rets = [obtain(x) for x in executor.map(init_model, range(tp_size))]

            # Wrap functions
            def async_wrap(func_name):
                fs = [rpyc.async_(getattr(m, func_name)) for m in self.model_servers]

                async def _func(*args, **kwargs):
                    tasks = [f(*args, **kwargs) for f in fs]
                    await asyncio.gather(*[asyncio.to_thread(t.wait) for t in tasks])
                    return obtain(tasks[0].value)

                return _func

            self.step = async_wrap("step")
            self.get_expert_locations = async_wrap("get_expert_locations")
            self.get_layer_expert_summary = async_wrap("get_layer_expert_summary")
            self.request_expert_reallocation = async_wrap("request_expert_reallocation")
            self.request_batch_expert_reallocation = async_wrap("request_batch_expert_reallocation")
            self.get_reallocation_status = async_wrap("get_reallocation_status")
            self.get_reallocation_stats = async_wrap("get_reallocation_stats")
            self.get_kv_cache_capacity = async_wrap("get_kv_cache_capacity")


def start_model_process(port):
    def _init_service(port):
        t = ThreadedServer(
            ModelRpcServer(),
            port=port,
            protocol_config={"allow_pickle": True, "sync_request_timeout": 1800},
        )
        t.start()

    proc = multiprocessing.Process(target=_init_service, args=(port,))
    proc.start()
    time.sleep(1)

    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect(
                "localhost",
                port,
                config={"allow_pickle": True, "sync_request_timeout": 1800},
            )
            break
        except ConnectionRefusedError:
            time.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise RuntimeError("init rpc env error!")

    assert proc.is_alive()
    return con.root, proc
