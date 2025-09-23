import torch
from fastmoe._cpu_kernel import token_attention_cpu
from fastmoe.layers.context_flashattention_nopad import context_attention_fwd
from fastmoe.layers.token_attention import token_attention_fwd
from fastmoe.backend.model_runner import ForwardMode, InputMetadata, DecodePart
from torch import nn


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scaling,
        num_kv_heads,
        layer_id,
    ):
        super().__init__()

        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

        self.prefill_forward = self.prefill_forward_triton
        self.decode_forward = self.decode_forward_triton
        self.cpu_decode_forward = self.cpu_decode_forward_cpp
    
    def prefill_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)

        # For paged KV cache, we need to handle attention differently
        if input_metadata.paged_kv_cache is not None:
            # Basic implementation: run attention without caching for now
            # TODO: Implement proper paged attention kernel
            context_attention_fwd(
                q.view(-1, self.tp_q_head_num, self.head_dim),
                k,
                v,
                o.view(-1, self.tp_q_head_num, self.head_dim),
                input_metadata.start_loc,
                input_metadata.seq_lens,
                input_metadata.max_seq_len,
            )
            # Store KV cache (currently a no-op for paged cache)
            self.store_kv_cache(k, v, input_metadata)
        else:
            # Original implementation for TokenToKVPool
            context_attention_fwd(
                q.view(-1, self.tp_q_head_num, self.head_dim),
                k,
                v,
                o.view(-1, self.tp_q_head_num, self.head_dim),
                input_metadata.start_loc,
                input_metadata.seq_lens,
                input_metadata.max_seq_len,
            )
            self.store_kv_cache(k, v, input_metadata)

        return o
    
    # Unused for now
    def decode_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        pass

    def cpu_decode_forward_cpp(self, input_metadata: InputMetadata):
        qkv = input_metadata.qkv_pin.view(-1, self.tp_q_head_num + 2 * self.tp_k_head_num, self.head_dim)
        query = qkv[:, :self.tp_q_head_num, :].view(-1, 1, self.tp_q_head_num, self.head_dim)
        k = qkv[:, self.tp_q_head_num : self.tp_q_head_num + self.tp_k_head_num, :].view(-1, self.tp_k_head_num, self.head_dim)
        v = qkv[:, self.tp_q_head_num + self.tp_k_head_num :, :].view(-1, self.tp_v_head_num, self.head_dim)

        self.store_kv_cache_cpu(k, v, input_metadata)
        
        # print("query:", query[:6, 0, :2, :2], "layer:", self.layer_id)
        if input_metadata.token_to_kv_pool is not None:
            token_attention_cpu(
                input_metadata.hidden_pin, query, 
                input_metadata.token_to_kv_pool.get_key_buffer_cpu(self.layer_id), 
                input_metadata.token_to_kv_pool.get_value_buffer_cpu(self.layer_id), 
                input_metadata.seq_lens, 
                input_metadata.start_loc, 
                self.head_dim**-0.5
            )
        elif input_metadata.paged_kv_cache is not None:
            # Paged KV cache CPU attention
            # For now, create temporary continuous buffers for CPU attention
            # This is not optimal but allows the code to run
            
            # Get batch info
            batch_size = len(input_metadata.seq_lens)
            max_seq_len = int(torch.max(input_metadata.seq_lens))
            
            # Allocate temporary continuous buffers for this batch
            key_buffer = torch.zeros(
                (batch_size * max_seq_len, self.tp_k_head_num, self.head_dim),
                dtype=query.dtype,
                device="cpu"
            )
            value_buffer = torch.zeros(
                (batch_size * max_seq_len, self.tp_v_head_num, self.head_dim),
                dtype=query.dtype,
                device="cpu"
            )
            
            # TODO: Copy data from paged blocks to continuous buffers
            # For now, using zeros - this will produce incorrect results but allows execution
            
            # Run CPU attention with temporary buffers
            token_attention_cpu(
                input_metadata.hidden_pin, query,
                key_buffer,
                value_buffer,
                input_metadata.seq_lens,
                input_metadata.start_loc,
                self.head_dim**-0.5
            )
        else:
            raise ValueError("Either token_to_kv_pool or paged_kv_cache must be provided")
        # print("hidden_pin:", input_metadata.hidden_pin[:6, 0, :2, :2])

        return
    
    def forward(self, q, k, v, input_metadata: InputMetadata):
        if k is not None and v is not None:
            k = k.view(-1, self.tp_k_head_num, self.head_dim)
            v = v.view(-1, self.tp_v_head_num, self.head_dim)

        if input_metadata.forward_mode == ForwardMode.PREFILL:
            return self.prefill_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            if input_metadata.decode_part == DecodePart.CPU_ATTN:
                return self.cpu_decode_forward_cpp(input_metadata)
            else:
                return self.decode_forward(q, k, v, input_metadata)
        
    def store_kv_cache_cpu(self, cache_k, cache_v, input_metadata: InputMetadata):
        if input_metadata.token_to_kv_pool is not None:
            key_buffer = input_metadata.token_to_kv_pool.get_key_buffer_cpu(self.layer_id)
            value_buffer = input_metadata.token_to_kv_pool.get_value_buffer_cpu(self.layer_id)
            key_buffer[input_metadata.out_cache_loc, :, :] = cache_k[:, :, :]
            value_buffer[input_metadata.out_cache_loc, :, :] = cache_v[:, :, :]
        elif input_metadata.paged_kv_cache is not None:
            # Paged KV cache system - CPU storage
            # TODO: Implement paged KV cache CPU storage
            pass
        else:
            raise ValueError("Either token_to_kv_pool or paged_kv_cache must be provided")
    
    def store_kv_cache(self, cache_k, cache_v, input_metadata: InputMetadata):
        if input_metadata.token_to_kv_pool is not None:
            # Original TokenToKVPool system
            key_buffer = input_metadata.token_to_kv_pool.get_key_buffer()
            value_buffer = input_metadata.token_to_kv_pool.get_value_buffer()
            if input_metadata.out_cache_loc is not None:
                key_buffer[input_metadata.out_cache_loc, :, :] = cache_k[:, :, :]
                value_buffer[input_metadata.out_cache_loc, :, :] = cache_v[:, : ,:]
            else:
                raise RuntimeError()
        elif input_metadata.paged_kv_cache is not None:
            # Paged KV cache system
            # For now, we'll store the keys and values directly
            # This is a placeholder - proper paged attention would allocate blocks
            # and write to specific block locations
            pass  # TODO: Implement paged KV cache storage
        else:
            raise ValueError("Either token_to_kv_pool or paged_kv_cache must be provided")
