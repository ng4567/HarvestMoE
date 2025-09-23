"""Paged KV Cache implementation for MoE Lightning.

This implements a paged attention-style KV cache with configurable block size,
similar to vLLM's PagedAttention but adapted for the MoE Lightning architecture.
"""
import math
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlockTable:
    """Manages block allocation for sequences."""
    num_blocks: int
    block_size: int
    device: str = "cuda"
    
    def __init__(self, num_blocks: int, block_size: int, device: str = "cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        # Free block indices
        self.free_blocks = list(range(num_blocks))
        # Mapping from sequence ID to list of block indices
        self.seq_to_blocks: Dict[int, List[int]] = {}
    
    def allocate_blocks(self, seq_id: int, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence."""
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(f"Not enough free blocks. Need {num_blocks_needed}, have {len(self.free_blocks)}")
        
        # Allocate blocks
        allocated_blocks = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop(0)
            allocated_blocks.append(block_idx)
        
        self.seq_to_blocks[seq_id] = allocated_blocks
        return allocated_blocks
    
    def free_sequence_blocks(self, seq_id: int):
        """Free blocks allocated to a sequence."""
        if seq_id in self.seq_to_blocks:
            blocks = self.seq_to_blocks[seq_id]
            self.free_blocks.extend(blocks)
            del self.seq_to_blocks[seq_id]
    
    def get_blocks(self, seq_id: int) -> List[int]:
        """Get block indices for a sequence."""
        return self.seq_to_blocks.get(seq_id, [])


class PagedKVCache:
    """Paged KV Cache implementation with configurable block size.
    
    This replaces TokenToKVPool with a paged attention approach where:
    - KV cache is divided into fixed-size blocks
    - Each block stores keys/values for `block_size` tokens
    - Blocks can be allocated/freed dynamically
    - Supports both GPU and CPU storage with block-based transfers
    """
    
    def __init__(self, 
                 num_gpu_blocks: int,
                 num_cpu_blocks: int,
                 block_size: int,
                 num_heads: int,
                 head_dim: int,
                 num_layers: int,
                 dtype: torch.dtype = torch.float16):
        """Initialize paged KV cache.
        
        Args:
            num_gpu_blocks: Number of blocks to allocate on GPU
            num_cpu_blocks: Number of blocks to allocate on CPU
            block_size: Number of tokens per block (must be power of 2)
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            num_layers: Number of model layers
            dtype: Data type for cache storage
        """
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        
        # Validate block size is power of 2
        if block_size & (block_size - 1) != 0:
            raise ValueError(f"Block size must be power of 2, got {block_size}")
        
        # GPU cache: [num_gpu_blocks, block_size, 2, num_heads, head_dim]
        # The '2' dimension is for keys and values
        self.gpu_cache = torch.zeros(
            (num_gpu_blocks, block_size, 2, num_heads, head_dim),
            dtype=dtype,
            device="cuda"
        )
        
        # CPU cache for offloading
        self.cpu_cache = []
        for layer_idx in range(num_layers):
            try:
                # Try to allocate pinned memory first
                cache = torch.zeros(
                    (num_cpu_blocks, block_size, 2, num_heads, head_dim),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=True
                )
            except (RuntimeError, torch.cuda.CudaError) as e:
                # Fall back to regular CPU memory if pinned allocation fails
                logger.warning(f"Failed to allocate pinned memory for layer {layer_idx}, using regular CPU memory: {e}")
                cache = torch.zeros(
                    (num_cpu_blocks, block_size, 2, num_heads, head_dim),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=False
                )
            self.cpu_cache.append(cache)
        
        # Block tables for each layer
        self.gpu_block_tables = [
            BlockTable(num_gpu_blocks, block_size, "cuda")
            for _ in range(num_layers)
        ]
        
        self.cpu_block_tables = [
            BlockTable(num_cpu_blocks, block_size, "cpu")
            for _ in range(num_layers)
        ]
        
        logger.info(f"Initialized PagedKVCache with {num_gpu_blocks} GPU blocks and "
                   f"{num_cpu_blocks} CPU blocks, block_size={block_size}")
    
    def allocate_gpu_blocks(self, seq_id: int, num_tokens: int, layer_id: int) -> List[int]:
        """Allocate GPU blocks for a sequence at a specific layer."""
        return self.gpu_block_tables[layer_id].allocate_blocks(seq_id, num_tokens)
    
    def allocate_cpu_blocks(self, seq_id: int, num_tokens: int, layer_id: int) -> List[int]:
        """Allocate CPU blocks for a sequence at a specific layer."""
        return self.cpu_block_tables[layer_id].allocate_blocks(seq_id, num_tokens)
    
    def write_kv(self, keys: torch.Tensor, values: torch.Tensor, 
                 block_indices: List[int], layer_id: int, 
                 start_token_idx: int = 0):
        """Write keys and values to specified blocks.
        
        Args:
            keys: [seq_len, num_heads, head_dim]
            values: [seq_len, num_heads, head_dim]
            block_indices: List of block indices to write to
            layer_id: Layer index
            start_token_idx: Starting token index within the sequence
        """
        seq_len = keys.shape[0]
        
        for i in range(seq_len):
            token_idx = start_token_idx + i
            block_idx = token_idx // self.block_size
            block_offset = token_idx % self.block_size
            
            if block_idx < len(block_indices):
                physical_block_idx = block_indices[block_idx]
                self.gpu_cache[physical_block_idx, block_offset, 0] = keys[i]
                self.gpu_cache[physical_block_idx, block_offset, 1] = values[i]
    
    def read_kv(self, block_indices: List[int], num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read keys and values from specified blocks.
        
        Returns:
            keys: [num_tokens, num_heads, head_dim]
            values: [num_tokens, num_heads, head_dim]
        """
        keys = torch.zeros((num_tokens, self.num_heads, self.head_dim), 
                          dtype=self.dtype, device="cuda")
        values = torch.zeros((num_tokens, self.num_heads, self.head_dim), 
                            dtype=self.dtype, device="cuda")
        
        for token_idx in range(num_tokens):
            block_idx = token_idx // self.block_size
            block_offset = token_idx % self.block_size
            
            if block_idx < len(block_indices):
                physical_block_idx = block_indices[block_idx]
                keys[token_idx] = self.gpu_cache[physical_block_idx, block_offset, 0]
                values[token_idx] = self.gpu_cache[physical_block_idx, block_offset, 1]
        
        return keys, values
    
    def swap_blocks(self, gpu_block_idx: int, cpu_block_idx: int, layer_id: int):
        """Swap a block between GPU and CPU memory."""
        # Copy GPU block to CPU
        self.cpu_cache[layer_id][cpu_block_idx].copy_(
            self.gpu_cache[gpu_block_idx], 
            non_blocking=True
        )
    
    def load_blocks(self, cpu_block_idx: int, gpu_block_idx: int, layer_id: int):
        """Load a block from CPU to GPU memory."""
        self.gpu_cache[gpu_block_idx].copy_(
            self.cpu_cache[layer_id][cpu_block_idx],
            non_blocking=True
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in GB."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        block_size_bytes = (self.block_size * 2 * self.num_heads * 
                           self.head_dim * element_size)
        
        gpu_blocks_total = self.gpu_cache.shape[0]
        gpu_blocks_used = sum(
            len(table.seq_to_blocks) 
            for table in self.gpu_block_tables
        ) / self.num_layers  # Average across layers
        
        cpu_blocks_total = self.cpu_cache[0].shape[0]
        cpu_blocks_used = sum(
            len(table.seq_to_blocks) 
            for table in self.cpu_block_tables
        ) / self.num_layers
        
        return {
            "gpu_cache_gb": gpu_blocks_total * block_size_bytes / (1024**3),
            "gpu_cache_used_gb": gpu_blocks_used * block_size_bytes / (1024**3),
            "cpu_cache_gb": cpu_blocks_total * block_size_bytes / (1024**3),
            "cpu_cache_used_gb": cpu_blocks_used * block_size_bytes / (1024**3),
            "block_size": self.block_size,
            "gpu_blocks_free": sum(len(t.free_blocks) for t in self.gpu_block_tables) / self.num_layers,
            "cpu_blocks_free": sum(len(t.free_blocks) for t in self.cpu_block_tables) / self.num_layers,
        }


def calculate_cache_blocks(cache_size_gb: float, 
                          block_size: int,
                          num_heads: int,
                          head_dim: int,
                          dtype: torch.dtype = torch.float16) -> int:
    """Calculate number of blocks that fit in given cache size.
    
    Args:
        cache_size_gb: Cache size in GB
        block_size: Tokens per block
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type
        
    Returns:
        Number of blocks that fit
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    # Each block stores keys and values (factor of 2)
    block_size_bytes = block_size * 2 * num_heads * head_dim * element_size
    cache_size_bytes = cache_size_gb * (1024**3)
    
    return int(cache_size_bytes / block_size_bytes)
