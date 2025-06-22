"""Inference-only Phi-3.5 MoE model."""
from typing import List, Optional, Tuple, Set, Dict
import torch
from torch import nn
from transformers import PhiConfig
from tqdm import tqdm
import time

from fastmoe.backend.task_meta import InputMetadata, DecodePart
from fastmoe.layers.attention import Attention
from fastmoe.layers.stacked_fused_moe import stack_fused_moe
from fastmoe.layers.linear import (LinearMethodBase,
                                               ColumnParallelLinear,
                                               StackedLinear,
                                               RowParallelLinear)
from fastmoe.layers.logits_processor import LogitsProcessor

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

class LayerNormWithResidual(nn.Module):
    """LayerNorm that handles residual connections and keeps parameter names compatible (weight/bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
        import torch.nn.functional as F
        if residual is None:
            residual = hidden_states
            out = F.layer_norm(hidden_states, hidden_states.shape[-1:], self.weight, self.bias, self.eps)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            out = F.layer_norm(hidden_states, hidden_states.shape[-1:], self.weight, self.bias, self.eps)
        return out, residual


class PhiMoE(nn.Module):
    """A tensor-parallel MoE implementation for Phi-3.5 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_layers = num_layers
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
            print(f"Using default dtype {params_dtype}")
        self.params_dtype = params_dtype

        self.gates = StackedLinear(self.num_layers,
                                    self.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     params_dtype=self.params_dtype,
                                     linear_method=None)
        self.ws = nn.Parameter(
            torch.empty(self.num_layers, self.num_total_experts,
                        3 * self.intermediate_size * self.hidden_size,
                        device="cpu",
                        dtype=self.params_dtype))

        set_weight_attrs(self.ws, {
            "weight_loader": self.weight_loader,
        })
        
        # Store the latest expert routing information for tracking
        self.last_topk_ids = None

        # === Per‑batch activation tracking ===
        # Dict[layer_id -> set(expert_ids)] for the current forward pass
        self.batch_activation_sets: Dict[int, Set[int]] = {
            i: set() for i in range(self.num_layers)
        }

        self.expert_activation_counts: Dict[int, torch.Tensor] = {
            i: torch.zeros(self.num_total_experts, dtype = torch.long)
            for i in range(self.num_layers)
        }

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int, layer_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        w3_offset = shard_size * self.hidden_size
        w2_offset = 2 * shard_size * self.hidden_size
        if weight_name.endswith("w1.weight"):
            param_data[layer_id, expert_id, 0 : w3_offset] = loaded_weight[shard, :].view(-1)
        if weight_name.endswith("w3.weight"):
            param_data[layer_id, expert_id,
                       w3_offset : w2_offset] = loaded_weight[shard, :].view(-1)
        if weight_name.endswith("w2.weight"):
            param_data[layer_id, expert_id, w2_offset:] = loaded_weight[:, shard].reshape(-1)

    # ------------------------------------------------------------------
    # Activation‑tracking helpers
    # ------------------------------------------------------------------
    def _update_activation_sets(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Merge newly‑active experts into the per‑layer set for this batch."""
        if topk_ids is None:
            return
        # Flatten, move to CPU (cheap metadata op), and convert to Python ints
        active = set(topk_ids.view(-1).tolist())
        self.batch_activation_sets[layer_id].update(active)
    
    def _increment_activation_counts(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Increment the activation counts for the given layer.
        """
        if topk_ids is None:
            return
        counts = torch.bincount(topk_ids.view(-1), 
                                minlength = self.num_total_experts)
        self.expert_activation_counts[layer_id] += counts

    def reset_activation_sets(self) -> None:
        """Clear all recorded activations (call once per new batch)."""
        for k in self.batch_activation_sets:
            self.batch_activation_sets[k].clear()

    def reset_activation_counts(self) -> None:
        """Clear all recorded activation counts (call once per new batch)."""
        for k in self.expert_activation_counts:
            self.expert_activation_counts[k].zero_()

    def get_activation_sets(self) -> Dict[int, Set[int]]:
        """Return the dict {layer_id: set(active_expert_ids)}."""
        return self.batch_activation_sets

    def forward(self, index:int, hidden_states: torch.Tensor, experts_cache: torch.Tensor, 
                input_metadata: Optional[InputMetadata] = None) -> torch.Tensor:
        # Check if we have pre-computed expert assignments
        if (input_metadata is not None and 
            hasattr(input_metadata, 'topk_weights') and 
            hasattr(input_metadata, 'topk_ids') and
            input_metadata.topk_weights is not None and
            input_metadata.topk_ids is not None):
            # Use pre-computed assignments
            topk_weights = input_metadata.topk_weights
            topk_ids = input_metadata.topk_ids
            
            # Store the expert indices for tracking
            self.last_topk_ids = topk_ids

            # Record activations for this layer
            self._update_activation_sets(index, self.last_topk_ids)
            self._increment_activation_counts(index, self.last_topk_ids)

            # Get dimensions
            M, H = hidden_states.shape
            _, page_size = self.ws.gpu_cache.shape
            N = 2 * page_size // 3 // H
            E = self.num_total_experts
            
            # Configuration for block sizes
            config = {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }

            if topk_ids.numel() <= E:
                config = {
                    'BLOCK_SIZE_M': 16,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 64,
                    'GROUP_SIZE_M': 1
                }
            
            # Create intermediate buffers
            intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                            device=hidden_states.device,
                                            dtype=hidden_states.dtype)
            intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                            device=hidden_states.device,
                                            dtype=hidden_states.dtype)
            intermediate_cache3 = torch.empty((M, topk_ids.shape[1], H),
                                            device=hidden_states.device,
                                            dtype=hidden_states.dtype)

            # Prepare for fused MoE kernel
            from fastmoe.layers.stacked_fused_moe import (
                moe_align_block_size, invoke_fused_moe_kernel, ops
            )
            
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, config['BLOCK_SIZE_M'], E)

            # First MoE multiplication (w1 and w3)
            invoke_fused_moe_kernel(hidden_states, self.ws.gpu_cache.view(self.ws.gpu_cache.shape[0], -1, H), 
                                  experts_cache, intermediate_cache1,
                                  topk_weights, topk_ids, sorted_token_ids,
                                  expert_ids, num_tokens_post_padded, False,
                                  topk_ids.shape[1], config, N, H, 0)

            # Activation function
            ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

            # Second MoE multiplication (w2)
            invoke_fused_moe_kernel(intermediate_cache2, self.ws.gpu_cache.view(self.ws.gpu_cache.shape[0], -1, N//2), 
                                  experts_cache, intermediate_cache3,
                                  topk_weights, topk_ids, sorted_token_ids,
                                  expert_ids, num_tokens_post_padded, True, 1,
                                  config, H, N // 2, 2*H)

            # Sum and return
            final_hidden_states = torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                                          dim=1, out=hidden_states)
        else:
            # Normal path: compute router logits
            router_logits, _ = self.gates(index, hidden_states)
            
            # Modified stack_fused_moe to return expert routing information
            final_hidden_states, topk_ids = self._stack_fused_moe_with_routing_info(
                hidden_states,
                self.ws.gpu_cache,
                experts_cache,
                router_logits,
                self.top_k,
                renormalize=True,
                inplace=True)
            
            # Store the expert indices for tracking
            self.last_topk_ids = topk_ids

            # Record activations for this layer
            self._update_activation_sets(index, self.last_topk_ids)
            # Update cumulative activation histogram
            self._increment_activation_counts(index, self.last_topk_ids)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    def _stack_fused_moe_with_routing_info(self, hidden_states, w1, experts_cache, gating_output, topk, renormalize, inplace):
        """
        Modified version of stack_fused_moe that returns both the output and the expert routing information
        """
        # Import here to avoid circular imports
        import vllm._moe_C as moe_kernels
        from fastmoe.layers.stacked_fused_moe import moe_align_block_size, invoke_fused_moe_kernel
        from vllm._C import ops
        
        # Check constraints.
        assert hidden_states.shape[0] == gating_output.shape[0], (
            "Number of tokens mismatch")
        E = gating_output.shape[1]
        H = hidden_states.shape[1]
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert w1.is_contiguous(), "Expert weights1 must be contiguous"
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16
        ]
        M, _ = hidden_states.shape
        _, page_size = w1.shape
        N = 2 * page_size // 3 // H

        topk_weights = torch.empty(M,
                                  topk,
                                  dtype=torch.float32,
                                  device=hidden_states.device)
        topk_ids = torch.empty(M,
                              topk,
                              dtype=torch.int32,
                              device=hidden_states.device)
        token_expert_indicies = torch.empty(M,
                                          topk,
                                          dtype=torch.int32,
                                          device=hidden_states.device)
        moe_kernels.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output.float(),  # TODO(woosuk): Optimize this.
        )
        del token_expert_indicies  # Not used. Will be used in the future.
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        config = {
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }

        if topk_ids.numel() <= E:
            config = {
                'BLOCK_SIZE_M': 16,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 1
            }

        intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
        intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
        intermediate_cache3 = torch.empty((M, topk_ids.shape[1], H),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config['BLOCK_SIZE_M'], E)

        invoke_fused_moe_kernel(hidden_states, w1.view(w1.shape[0], -1, H), experts_cache, intermediate_cache1,
                                topk_weights, topk_ids, sorted_token_ids,
                                expert_ids, num_tokens_post_padded, False,
                                topk_ids.shape[1], config, N, H, 0)

        ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        invoke_fused_moe_kernel(intermediate_cache2, w1.view(w1.shape[0], -1, N//2), experts_cache, intermediate_cache3,
                                topk_weights, topk_ids, sorted_token_ids,
                                expert_ids, num_tokens_post_padded, True, 1,
                                config, H, N // 2, 2*H)

        if inplace:
            output = torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                             dim=1,
                             out=hidden_states)
        else:
            output = torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                             dim=1)
        
        return output, topk_ids


class PhiAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 layer_id: int = 0,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None,
                 rope_scaling: Optional[dict] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=True,
            linear_method=linear_method,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=True,
            linear_method=linear_method,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=True,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=True,
            linear_method=linear_method,
        )

        # Handle RoPE scaling configuration for Phi-3.5 MoE
        # For compatibility, just pass rope_scaling as-is
        # Note: Due to vllm compatibility issues with Phi-3.5's rope_scaling format,
        # we'll use None for now
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
            rope_scaling=None  # Using None due to compatibility issues
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.PREATTN:
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
            q, k = self.rotary_emb(positions, q, k)
            if input_metadata.decode_part == DecodePart.PREATTN:
                return torch.cat([q, k, v], dim=-1).contiguous()
            
        if input_metadata.decode_part == DecodePart.ALL:
            attn_output = self.attn(q, k, v, input_metadata)
            input_metadata.attn_event.record(torch.cuda.current_stream())
        if input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_output = self.attn(None, None, None, input_metadata)
            return attn_output
            
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            if input_metadata.decode_part == DecodePart.POSTATTN:
                attn_output = hidden_states
            output, _ = self.o_proj(attn_output)
        return output


class PhiDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PhiConfig,
        layer_id: int = 0,
        block_sparse_moe: PhiMoE = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = PhiAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            layer_id=layer_id,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window,
            linear_method=linear_method,
            rope_scaling=getattr(config, "rope_scaling", None))
        self.block_sparse_moe = block_sparse_moe
        self.input_layernorm = LayerNormWithResidual(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNormWithResidual(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Apply layernorm only when hidden_states is available (not None)
        if hidden_states is not None and (
            input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.PREATTN
        ):
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        if input_metadata.decode_part == DecodePart.PREATTN:
            qkv = self.self_attn(positions, hidden_states, input_metadata)
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.self_attn(positions, hidden_states, input_metadata)
            return attn_out
        elif input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            # Self Attention
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                input_metadata=input_metadata,
            )
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.block_sparse_moe(self.layer_id, hidden_states, input_metadata.experts_mapping, input_metadata)

        return hidden_states, residual


class PhiModel(nn.Module):

    def __init__(
        self,
        config: PhiConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        # init only one PhiMoE
        self.block_sparse_moe = PhiMoE(
            num_layers = config.num_hidden_layers,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size)
        self.layers = nn.ModuleList([
            PhiDecoderLayer(config, i, block_sparse_moe=self.block_sparse_moe, linear_method=linear_method)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = LayerNormWithResidual(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: List[int] = None,
    ) -> torch.Tensor:
        if hidden_states is None and input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        if cur_layers is None:
            cur_layers = range(len(self.layers))
        for i in cur_layers:
            layer = self.layers[i]
            if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
                hidden_states, residual = layer(positions, hidden_states,
                                                input_metadata,
                                                residual)
            elif input_metadata.decode_part == DecodePart.PREATTN:
                qkv, residual = layer(positions, hidden_states,
                                                input_metadata,
                                                residual)
                return qkv, residual
            elif input_metadata.decode_part == DecodePart.CPU_ATTN:
                attn_out = layer(None, None, input_metadata, None)
                return attn_out
        if cur_layers[-1] != len(self.layers) - 1:
            return hidden_states, residual
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states, _


class PhiMoEForCausalLMOff(nn.Module):

    def __init__(
        self,
        config: PhiConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = PhiModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: List[int] = None,
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            if hidden_states is not None:
                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
            hidden_states, residual = self.model(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
        elif input_metadata.decode_part == DecodePart.PREATTN:
            qkv, residual = self.model(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.model(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
            return attn_out
        if self.config.num_hidden_layers - 1 not in cur_layers:
            return hidden_states, residual
        else:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head.weight, input_metadata
            )

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("q_proj", "q_proj", "q"),
            ("k_proj", "k_proj", "k"),
            ("v_proj", "v_proj", "v"),
        ]

        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            ("block_sparse_moe.ws",
             f"layers.{layer_id}.block_sparse_moe.experts.{expert_id}.{weight_name}.weight", expert_id, layer_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
            for layer_id in range(self.config.num_hidden_layers)
        ]

        gate_params_mapping = [
            # (param_name, weight_name, layer_id)
            ("block_sparse_moe.gates.weight",
             f"layers.{layer_id}.block_sparse_moe.gate.weight", layer_id)
            for layer_id in range(self.config.num_hidden_layers)
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in tqdm(hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False)):
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, layer_id) in gate_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, layer_id)
                break
            else:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight)
                    break
                else:
                    for param_name, weight_name, expert_id, layer_id in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param,
                                    loaded_weight,
                                    weight_name,
                                    expert_id=expert_id,
                                    layer_id=layer_id)
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
    
    def get_experts_mem(self):
        return self.model.block_sparse_moe.ws.data
    
    def link_gpu_experts_cache(self, expert_pool):
        set_weight_attrs(self.model.block_sparse_moe.ws, {"gpu_cache": expert_pool})
    
    def get_expert_indices(self):
        """Get the expert indices for the current batch.
        
        Returns:
            torch.Tensor or None: The expert indices from the most recent forward pass.
                                 Shape: [num_tokens, top_k] where each value is an expert ID.
                                 Returns None if no forward pass has been executed yet.
        """
        if hasattr(self.model.block_sparse_moe, 'last_topk_ids') and self.model.block_sparse_moe.last_topk_ids is not None:
            # Return a flattened view of the expert indices for easier processing
            return self.model.block_sparse_moe.last_topk_ids.flatten()
        return None
    
    def get_router(self, layer_id: int):
        """Get the router (gate) for a specific layer.
        
        Args:
            layer_id: The layer index (0 to num_hidden_layers-1)
            
        Returns:
            The router module that can compute expert scores for that layer (also known as the gate)
        """
        return self.model.block_sparse_moe.gates
    
    def forward_with_expert_assignments(self,
                                      input_ids: torch.Tensor,
                                      positions: torch.Tensor,
                                      input_metadata: InputMetadata,
                                      hidden_states: torch.Tensor,
                                      residual: torch.Tensor,
                                      cur_layers: List[int],
                                      topk_weights: torch.Tensor,
                                      topk_ids: torch.Tensor,
                                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with pre-computed expert routing.
        
        This method bypasses the router computation and uses provided expert assignments.
        
        Args:
            input_ids: Input token IDs
            positions: Position IDs
            input_metadata: Metadata for the forward pass
            hidden_states: Current hidden states
            residual: Residual connection
            cur_layers: List containing the current layer index
            topk_weights: Pre-computed weights for top-k experts [num_tokens, top_k]
            topk_ids: Pre-computed expert IDs for each token [num_tokens, top_k]
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (hidden_states, residual) after processing
        """
        # Store the pre-computed expert assignments in input_metadata
        input_metadata.topk_weights = topk_weights
        input_metadata.topk_ids = topk_ids
        
        # Run normal forward pass - the PhiMoE layer will need to be modified
        # to use these pre-computed values instead of computing them
        return self.forward(input_ids, positions, input_metadata, 
                          hidden_states, residual, cur_layers)

# ------------------------------------------------------------------
# MoE activation utilities
# ------------------------------------------------------------------
    def get_and_print_batch_activation_sets(self):
        """Return the dict {layer_id: set(active_expert_ids)} collected during the latest forward pass."""
        set_all_possible_experts = set(range(self.config.num_local_experts))
        for layer in self.model.block_sparse_moe.batch_activation_sets:
            print(f"Layer {layer}: activated experts {sorted(list(self.model.block_sparse_moe.batch_activation_sets[layer]))}")
            print(f"Layer {layer}: unactivated experts {sorted(list(set_all_possible_experts - self.model.block_sparse_moe.batch_activation_sets[layer]))}")

        return self.model.block_sparse_moe.get_activation_sets()


    # ------------------------------------------------------------------
    #  MoE activation count passthrough helpers
    # ------------------------------------------------------------------
    @property
    def expert_activation_counts(self):
        """Expose the cumulative per-expert activation counters."""
        return self.model.block_sparse_moe.expert_activation_counts

    def reset_activation_counts(self):
        """Reset cumulative activation histogram inside the MoE block."""
        self.model.block_sparse_moe.reset_activation_counts()


EntryClass = PhiMoEForCausalLMOff