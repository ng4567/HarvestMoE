# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only PhiMoE model."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from fastmoe.backend.task_meta import InputMetadata, DecodePart
from fastmoe.layers.attention import Attention as FastMoEAttention
from fastmoe.layers.linear import (
    LinearMethodBase,
    QKVParallelLinear as FastMoEQKVParallelLinear,
    StackedLinear,
    RowParallelLinear as FastMoERowParallelLinear,
)
from fastmoe.layers.logits_processor import LogitsProcessor as FastLogitsProcessor
from fastmoe.layers.stacked_fused_moe import stack_fused_moe
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.attention.layer import Attention as VllmAttention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as VllmLogitsProcessor,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from fastmoe.layers.compat import (
    default_weight_loader as fm_default_weight_loader,
    hf_model_weights_iterator,
)


class PhiMoEConfig(PretrainedConfig):
    model_type = "phimoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        sliding_window=None,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=16,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        attention_bias=False,
        lm_head_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        if rope_parameters is None:
            rope_theta = kwargs.pop("rope_theta", 1e6)
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class mp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        multiplier: torch.Tensor,
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
        ctx,
        grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expanded = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expanded.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expanded,
            None,
            None,
            None,
            None,
        )


def sparsemixer(scores, jitter_eps=0.01):
    ################ first expert ################

    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (
            2 * jitter_eps
        )

    # apply mask
    masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))
    selected_experts = max_ind

    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    multiplier = multiplier_o

    # masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float("-inf"),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (
            2 * jitter_eps
        )

    # apply mask
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))
    selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2 = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )


def phimoe_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert topk == 2, "Only top-2 routing is supported"
    assert renormalize is False, "Renormalization is not supported"

    topk_weights, topk_ids = sparsemixer(gating_output)
    return topk_weights, topk_ids


class PhiMoE(nn.Module):
    """A tensor-parallel MoE implementation for PhiMoE that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
        )

        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=True,
            renormalize=False,
            quant_config=quant_config,
            tp_size=tp_size,
            custom_routing_function=phimoe_routing_function,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class PhiMoEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        head_dim: int | None = None,
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
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
        if head_dim is None:
            head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )
        self.attn = VllmAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class PhiMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PhiMoEConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        self.self_attn = PhiMoEAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(
                config, "head_dim", self.hidden_size // config.num_attention_heads
            ),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )
        self.block_sparse_moe = PhiMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe",
        )
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states, residual


@support_torch_compile
class PhiMoEModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PhiMoEDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class PhiMoEForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config

        self.quant_config = vllm_config.quant_config

        self.model = PhiMoEModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            bias=True,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = VllmLogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class PhiMoEOffload(nn.Module):
    """PhiMoE expert layer using fastmoe offload cache."""

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        tp_size: int | None = None,
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

        self.gates = StackedLinear(
            self.num_layers,
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=self.params_dtype,
            linear_method=None,
        )
        self.ws = nn.Parameter(
            torch.empty(
                self.num_layers,
                self.num_total_experts,
                3 * self.intermediate_size * self.hidden_size,
                device="cpu",
                dtype=self.params_dtype,
            )
        )
        set_weight_attrs(self.ws, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
        layer_id: int,
    ):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        w3_offset = shard_size * self.hidden_size
        w2_offset = 2 * shard_size * self.hidden_size
        if weight_name.endswith("w1.weight"):
            param_data[layer_id, expert_id, 0:w3_offset] = loaded_weight[
                shard, :
            ].view(-1)
        if weight_name.endswith("w3.weight"):
            param_data[layer_id, expert_id, w3_offset:w2_offset] = loaded_weight[
                shard, :
            ].view(-1)
        if weight_name.endswith("w2.weight"):
            param_data[layer_id, expert_id, w2_offset:] = loaded_weight[
                :, shard
            ].reshape(-1)

    def forward(
        self, index: int, hidden_states: torch.Tensor, experts_cache: torch.Tensor
    ) -> torch.Tensor:
        router_logits, _ = self.gates(index, hidden_states)
        final_hidden_states = stack_fused_moe(
            hidden_states,
            self.ws.gpu_cache,
            experts_cache,
            router_logits,
            self.top_k,
            renormalize=False,
            inplace=True,
        )
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states


class PhiMoEAttentionOff(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_parameters: dict | None = None,
        head_dim: int | None = None,
        linear_method: LinearMethodBase | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is None:
            head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = FastMoEQKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.o_proj = FastMoERowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=True,
            linear_method=linear_method,
        )
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": 10000}
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position,
            is_neox_style=True,
            rope_parameters=rope_parameters,
        )
        self.attn = FastMoEAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        if input_metadata.decode_part in (DecodePart.ALL, DecodePart.PREATTN):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            if input_metadata.decode_part == DecodePart.PREATTN:
                return torch.cat([q, k, v], dim=-1).contiguous()

        if input_metadata.decode_part == DecodePart.ALL:
            attn_output = self.attn(q, k, v, input_metadata)
            input_metadata.attn_event.record(torch.cuda.current_stream())
        if input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_output = self.attn(None, None, None, input_metadata)
            return attn_output

        if input_metadata.decode_part in (DecodePart.ALL, DecodePart.POSTATTN):
            if input_metadata.decode_part == DecodePart.POSTATTN:
                attn_output = hidden_states
            output, _ = self.o_proj(attn_output)
        return output


class PhiMoEDecoderLayerOff(nn.Module):
    def __init__(
        self,
        config: PhiMoEConfig,
        layer_id: int = 0,
        block_sparse_moe: PhiMoEOffload | None = None,
        linear_method: LinearMethodBase | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        # Build rope_parameters from config attributes
        # Note: "longrope" requires vLLM's global config which fastmoe doesn't set,
        # so we fall back to "default" rope for unsupported scaling types.
        rope_theta = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type", "default"))
            # longrope/phi3longrope require vLLM config - fall back to default
            if rope_type in ("longrope", "phi3longrope"):
                rope_parameters = {"rope_type": "default", "rope_theta": float(rope_theta)}
            else:
                rope_parameters = {
                    "rope_type": rope_type,
                    "rope_theta": float(rope_theta),
                    **{k: v for k, v in rope_scaling.items() if k not in ("type", "rope_type")},
                }
        else:
            rope_parameters = {"rope_type": "default", "rope_theta": float(rope_theta)}
        
        self.self_attn = PhiMoEAttentionOff(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            layer_id=layer_id,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(
                config, "head_dim", self.hidden_size // config.num_attention_heads
            ),
            rope_parameters=rope_parameters,
            linear_method=linear_method,
        )
        self.block_sparse_moe = block_sparse_moe
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_metadata.decode_part in (DecodePart.ALL, DecodePart.PREATTN):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        if input_metadata.decode_part == DecodePart.PREATTN:
            qkv = self.self_attn(positions, hidden_states, input_metadata)
            return qkv, residual
        if input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.self_attn(positions, hidden_states, input_metadata)
            return attn_out

        if input_metadata.decode_part in (DecodePart.ALL, DecodePart.POSTATTN):
            if input_metadata.decode_part == DecodePart.POSTATTN:
                attn_out = hidden_states
            else:
                attn_out = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    input_metadata=input_metadata,
                )
            hidden_states = attn_out + residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.block_sparse_moe(
                self.layer_id, hidden_states, input_metadata.experts_mapping
            )
            hidden_states = hidden_states + residual

        return hidden_states, residual


class PhiMoEModelOff(nn.Module):
    def __init__(self, config: PhiMoEConfig, linear_method: LinearMethodBase | None = None):
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
        self.block_sparse_moe = PhiMoEOffload(
            num_layers=config.num_hidden_layers,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.layers = nn.ModuleList(
            [
                PhiMoEDecoderLayerOff(
                    config, i, block_sparse_moe=self.block_sparse_moe, linear_method=linear_method
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: list[int] | None = None,
    ) -> torch.Tensor:
        if hidden_states is None and input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        if cur_layers is None:
            cur_layers = range(len(self.layers))
        for i in cur_layers:
            layer = self.layers[i]
            if input_metadata.decode_part in (DecodePart.ALL, DecodePart.POSTATTN):
                hidden_states, residual = layer(
                    positions, hidden_states, input_metadata, residual
                )
            elif input_metadata.decode_part == DecodePart.PREATTN:
                qkv, residual = layer(
                    positions, hidden_states, input_metadata, residual
                )
                return qkv, residual
            elif input_metadata.decode_part == DecodePart.CPU_ATTN:
                attn_out = layer(None, None, input_metadata, None)
                return attn_out
        if cur_layers[-1] != len(self.layers) - 1:
            return hidden_states, residual
        else:
            hidden_states = self.norm(hidden_states)
            return hidden_states, residual


class PhiMoEForCausalLMOff(nn.Module):
    def __init__(
        self,
        config: PhiMoEConfig,
        linear_method: LinearMethodBase | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = PhiMoEModelOff(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = FastLogitsProcessor(config)
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: list[int] | None = None,
    ) -> torch.Tensor:
        if cur_layers is None:
            cur_layers = list(range(self.config.num_hidden_layers))
        if input_metadata.decode_part in (DecodePart.ALL, DecodePart.POSTATTN):
            if hidden_states is not None:
                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
            hidden_states, residual = self.model(
                input_ids,
                positions,
                input_metadata,
                hidden_states=hidden_states,
                residual=residual,
                cur_layers=cur_layers,
            )
        elif input_metadata.decode_part == DecodePart.PREATTN:
            qkv, residual = self.model(
                input_ids,
                positions,
                input_metadata,
                hidden_states=hidden_states,
                residual=residual,
                cur_layers=cur_layers,
            )
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.model(
                input_ids,
                positions,
                input_metadata,
                hidden_states=hidden_states,
                residual=residual,
                cur_layers=cur_layers,
            )
            return attn_out

        if self.config.num_hidden_layers - 1 not in cur_layers:
            return hidden_states, residual
        else:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head.weight, input_metadata
            )

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: str | None = None,
        load_format: str = "auto",
        revision: str | None = None,
    ):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = [
            (
                "block_sparse_moe.ws",
                f"layers.{layer_id}.block_sparse_moe.experts.{expert_id}.{weight_name}.weight",
                expert_id,
                layer_id,
            )
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
            for layer_id in range(self.config.num_hidden_layers)
        ]

        gate_params_mapping = [
            (
                "block_sparse_moe.gates.weight",
                f"layers.{layer_id}.block_sparse_moe.gate.weight",
                layer_id,
            )
            for layer_id in range(self.config.num_hidden_layers)
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path,
            cache_dir,
            load_format,
            revision,
            fall_back_to_pt=False,
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, layer_id in gate_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, layer_id)
                break
            else:
                for param_name, weight_name, expert_id, layer_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        weight_name,
                        expert_id=expert_id,
                        layer_id=layer_id,
                    )
                    break
                else:
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        break
                    else:
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", fm_default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

    def get_experts_mem(self):
        return self.model.block_sparse_moe.ws.data

    def link_gpu_experts_cache(self, expert_pool):
        set_weight_attrs(self.model.block_sparse_moe.ws, {"gpu_cache": expert_pool})


EntryClass = PhiMoEForCausalLMOff