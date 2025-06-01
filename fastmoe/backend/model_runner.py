import importlib
import logging
from functools import lru_cache
from pathlib import Path

import torch
from fastmoe.backend.task import Batch
from fastmoe.backend.task_meta import DecodePart, ForwardMode, InputMetadata
from fastmoe.backend.memory import TokenToKVPool
from fastmoe.utils.utils import get_available_gpu_memory, get_available_cpu_memory
from vllm.model_executor.model_loader import _set_default_torch_dtype
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel

import fastmoe

logger = logging.getLogger("model_runner")



@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    for module_path in (Path(fastmoe.__file__).parent / "models").glob("*.py"):
        try:
            module = importlib.import_module(f"fastmoe.models.{module_path.stem}")
            if hasattr(module, "EntryClass"):
                model_arch_name_to_cls[module.EntryClass.__name__] = module.EntryClass
        except ImportError as e:
            logger.warning(f"Failed to import {module_path.stem}: {e}")
            continue
    return model_arch_name_to_cls


def get_model_cls_by_arch_name(model_arch_names, offload):
    model_arch_name_to_cls = import_model_classes()

    model_class = None
    for arch in model_arch_names:
        if offload:
            arch += "Off"
        if arch in model_arch_name_to_cls:
            model_class = model_arch_name_to_cls[arch]
            break
    else:
        raise ValueError(
            f"Unsupported architectures: {arch}. "
            f"Supported list: {list(model_arch_name_to_cls.keys())}"
        )
    return model_class

class ModelRunner:
    def __init__(
        self,
        model_config,
        mem_fraction_static,
        tp_rank,
        tp_size,
        nccl_port,
        load_format="auto",
        trust_remote_code=True,
    ):
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.nccl_port = nccl_port
        self.load_format = load_format
        self.trust_remote_code = trust_remote_code
        self.offload = True

        # Init torch distributed
        torch.cuda.set_device(self.tp_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=self.tp_size,
            rank=self.tp_rank,
            init_method=f"tcp://127.0.0.1:{self.nccl_port}",
        )

        # A small all_reduce for warmup.
        if self.tp_size > 1:
            torch.distributed.all_reduce(torch.zeros(1).cuda())
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        self.total_gpu_memory = get_available_gpu_memory(
            self.tp_rank, distributed=self.tp_size > 1
        )
        print(f"Total GPU memory: {self.total_gpu_memory}")
        self.total_cpu_memory = get_available_cpu_memory()
        self.load_model()
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        print(f"Free GPU memory: {free_gpu_memory / (1 << 30)}")

    def load_model(self):
        """See also vllm/model_executor/model_loader.py::get_model"""
        # Select model class
        architectures = getattr(self.model_config.hf_config, "architectures", [])
        model_class = get_model_cls_by_arch_name(architectures, self.offload)
        logger.info(f"Rank {self.tp_rank}: load weight begin.")

        # Load weights
        linear_method = None
        with _set_default_torch_dtype(torch.float16):
            with torch.device("cuda"):
                model = model_class(
                    config=self.model_config.hf_config, linear_method=linear_method
                )
            model.load_weights(
                self.model_config.path,
                cache_dir=None,
                load_format=self.load_format,
                revision=None,
            )
        self.model = model.eval()

        logger.info(f"Rank {self.tp_rank}: load weight end.")
    
    @torch.inference_mode()
    def forward_prefill(
        self,
        input_ids,
        seq_lens,
        positions,
        start_loc,
        max_seq_len,
        out_cache_loc,
        token_to_kv_pool,
        return_logprob,
        hidden_states,
        residual,
        cur_layers,
        decode_part,
        attn_event,
        experts_mapping,
    ):
        input_metadata = InputMetadata.create(
            forward_mode=ForwardMode.PREFILL,
            decode_part=decode_part,
            seq_lens=seq_lens,
            positions=positions,
            start_loc=start_loc,
            max_seq_len=max_seq_len,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=token_to_kv_pool,
            return_logprob=return_logprob,
            attn_event=attn_event,
            experts_mapping=experts_mapping,
        )
        return self.model.forward(input_ids, input_metadata.positions, input_metadata, hidden_states, residual, cur_layers)
    
    @torch.inference_mode()
    def forward_pre_attn(
        self,
        input_ids,
        seq_lens,
        hidden_states,
        residual,
        cur_layers,
        decode_part,
    ) -> torch.Tensor:
        input_metadata = InputMetadata.create(
            forward_mode=ForwardMode.DECODE,
            decode_part=decode_part,
            seq_lens=seq_lens,
        )
        return self.model.forward(input_ids, input_metadata.positions, input_metadata, hidden_states, residual, cur_layers)
    
    @torch.inference_mode()
    def forward_post_attn(
        self,
        input_ids,
        hidden_states,
        residual,
        cur_layers,
        decode_part,
        experts_mapping,
    ) -> torch.Tensor:
        input_metadata = InputMetadata.create(
            forward_mode=ForwardMode.DECODE,
            decode_part=decode_part,
            experts_mapping=experts_mapping,
        )
        return self.model.forward(input_ids, input_metadata.positions, input_metadata, hidden_states, residual, cur_layers)

    @torch.inference_mode()
    def forward_cpu_attn(
        self,
        cpu_token_start_index,
        token_to_kv_pool,
        seq_lens,
        start_loc,
        out_cache_loc,
        cur_layers,
        decode_part,
        qkv_pin,
        hidden_pin,
    ) -> torch.Tensor:
        input_metadata = InputMetadata.create_cpu_attn(
            forward_mode=ForwardMode.DECODE,
            decode_part=decode_part,
            cpu_token_start_index=cpu_token_start_index,
            start_loc=start_loc,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            qkv_pin=qkv_pin,
            hidden_pin=hidden_pin,
            token_to_kv_pool=token_to_kv_pool
        )
        return self.model.forward(None, None, input_metadata, None, None, cur_layers)

    def prefill(self, batch: Batch, decode_part: DecodePart, layer_id: int, experts_mapping: torch.Tensor, return_logprob=False, attn_event=None) -> torch.Tensor:
        kwargs = {
            "input_ids": batch.input_ids,
            "seq_lens": batch.seq_lens,
            "positions": batch.positions,
            "start_loc": batch.start_loc_gpu,
            "max_seq_len": batch.max_seq_len,
            "out_cache_loc": batch.out_cache_loc,
            "token_to_kv_pool": batch.token_to_kv_pool,
            "return_logprob": return_logprob,
            "hidden_states": batch.hidden_states,
            "residual": batch.residual,
            "cur_layers": [layer_id],
            "decode_part": decode_part,
            "attn_event": attn_event,
            "experts_mapping": experts_mapping,
        }
        return self.forward_prefill(**kwargs)
    
    def pre_attn(self, batch: Batch, decode_part: DecodePart, layer_id: int) -> torch.Tensor:
        kwargs = {
            "input_ids": batch.input_ids,
            "seq_lens": batch.seq_lens,
        }
        kwargs["hidden_states"] = batch.hidden_states
        kwargs["residual"] = batch.residual
        kwargs["cur_layers"] = [layer_id]
        kwargs["decode_part"] = decode_part
        return self.forward_pre_attn(**kwargs)
    
    def cpu_attention(self, batch: Batch, decode_part: DecodePart, layer_id: int, input_pin: torch.Tensor, output_pin: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        kwargs["cpu_token_start_index"] = batch.cpu_kv_pool_start_loc
        kwargs["token_to_kv_pool"] = batch.token_to_kv_pool
        kwargs["seq_lens"] = batch.seq_lens_cpu
        kwargs["start_loc"] = batch.start_loc
        kwargs["qkv_pin"] = input_pin
        kwargs["hidden_pin"] = output_pin
        kwargs["cur_layers"] = [layer_id]
        kwargs["out_cache_loc"] = batch.decode_out_cache_loc
        kwargs["decode_part"] = decode_part
        return self.forward_cpu_attn(**kwargs)
    
    def post_attn(self, batch: Batch, decode_part: DecodePart, layer_id: int, experts_mapping: torch.Tensor) -> torch.Tensor:
        kwargs = {
            "input_ids": batch.input_ids,
        }
        kwargs["hidden_states"] = batch.hidden_states
        kwargs["residual"] = batch.residual
        kwargs["cur_layers"] = [layer_id]
        kwargs["decode_part"] = decode_part
        kwargs["experts_mapping"] = experts_mapping
        return self.forward_post_attn(**kwargs)
        
    @torch.inference_mode()
    def get_router_logits(self, batch: Batch, layer_id: int) -> torch.Tensor:
        """Get router logits for expert selection without running the full forward pass."""
        # Get the router for this layer
        # This assumes the model has a method to access routers by layer
        router = self.model.get_router(layer_id)
        
        # Run just the router forward pass
        hidden_states = batch.hidden_states
        if hidden_states is None:
            # If no hidden states yet, run embedding layer
            # Handle different model architectures
            if hasattr(self.model, 'embed_tokens'):
                hidden_states = self.model.embed_tokens(batch.input_ids)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # For models like PhiMoE where embed_tokens is nested
                hidden_states = self.model.model.embed_tokens(batch.input_ids)
            else:
                raise AttributeError("Cannot find embed_tokens in model structure")
            
        # For models with StackedLinear routers, we need to pass the layer index
        # The router should be able to handle this
        if hasattr(router, '__call__'):
            # If it's a StackedLinear, it expects (layer_index, input)
            router_logits, _ = router(layer_id, hidden_states)
        else:
            # Fallback for other router types
            router_logits = router(hidden_states)
            
        return router_logits
    
    @torch.inference_mode()
    def prefill_subset(self, sub_batch: Batch, layer_id: int, 
                      experts_mapping: torch.Tensor,
                      topk_weights: torch.Tensor, 
                      topk_ids: torch.Tensor) -> tuple:
        """Process a subset of tokens with their expert assignments."""
        # Create InputMetadata object
        input_metadata = InputMetadata.create(
            forward_mode=ForwardMode.PREFILL,
            decode_part=DecodePart.ALL,
            seq_lens=sub_batch.seq_lens if hasattr(sub_batch, 'seq_lens') else None,
            positions=sub_batch.positions if hasattr(sub_batch, 'positions') else None,
            start_loc=sub_batch.start_loc_gpu if hasattr(sub_batch, 'start_loc_gpu') else None,
            max_seq_len=sub_batch.max_seq_len if hasattr(sub_batch, 'max_seq_len') else None,
            out_cache_loc=sub_batch.out_cache_loc if hasattr(sub_batch, 'out_cache_loc') else None,
            token_to_kv_pool=sub_batch.token_to_kv_pool if hasattr(sub_batch, 'token_to_kv_pool') else None,
            return_logprob=False,
            attn_event=None,
            experts_mapping=experts_mapping,
        )
        
        # Store the pre-computed expert assignments in input_metadata
        input_metadata.topk_weights = topk_weights
        input_metadata.topk_ids = topk_ids
        
        # Call the model's forward_with_expert_assignments method
        return self.model.forward_with_expert_assignments(
            input_ids=sub_batch.input_ids,
            positions=input_metadata.positions,
            input_metadata=input_metadata,
            hidden_states=sub_batch.hidden_states,
            residual=sub_batch.residual,
            cur_layers=[layer_id],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        
