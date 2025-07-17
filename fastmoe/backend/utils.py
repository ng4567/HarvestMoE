import dataclasses
from fastmoe.utils.model_config import ModelConfig
import numpy as np
from collections import defaultdict
import torch
import os
import csv
import datetime
from typing import List 

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

@dataclasses.dataclass
class Policy:
    ubs: int
    n_ub: int

    wg: float
    wc: float
    cg: float
    cc: float

    eg: int

@dataclasses.dataclass
class HardwareConfig:
    gmem: int = 24 * GB
    cmem: int = 192 * GB

    ctog_bdw: float = 16 * GB
    g_bdw: float = 300 * GB
    c_bdw: float = 100 * GB

    gpu_flops: float = 104 * T
    cpu_flops: float = 13 * T

    tp_size: int = 1

    @classmethod
    def init(cls, gpu_device_name, cpu_mem, c_bdw, tp_size):
        if "H100" in gpu_device_name:
            return cls(
                gmem=93 * GB,  # H100 NVL has ~93GB available memory (measured)
                cmem=cpu_mem * GB,
                ctog_bdw=24 * GB,  # H100 has high PCIe 5.0/NVLink bandwidth
                g_bdw=3350 * GB,  # Measured ~3320-3397 GB/s, using 3350 GB/s
                c_bdw=c_bdw * GB,
                gpu_flops=450 * T,  # Measured ~400-472 TFLOPS sustained performance
                cpu_flops=1.6 * T,  # Conservative CPU estimate
                tp_size=tp_size
            )
        elif "L4" in gpu_device_name:
            return cls(
                gmem=24 * GB,
                cmem=cpu_mem * GB,
                ctog_bdw=16 * GB,
                g_bdw=285 * GB,
                c_bdw=c_bdw * GB,
                gpu_flops=104 * T,
                cpu_flops=1.6 * T,
                tp_size=tp_size
            )
        elif "T4" in gpu_device_name:
            return cls(
                gmem=15 * GB,
                cmem=cpu_mem * GB,
                ctog_bdw=16 * GB,
                g_bdw=300 * GB,
                c_bdw=c_bdw * GB,
                gpu_flops=65 * T,
                cpu_flops=0.8 * T,
                tp_size=tp_size
            )
        else:
            return cls

@dataclasses.dataclass
class CostModelConfig:
    s: int = 512
    n: int = 32

    l: int = 32
    h1: int = 4096
    h2: int = 4096 * 4
    nh: int = 32
    nkvh: int = 8

    n_experts: int = 8
    topk: int = 2

    gmem: int = 24 * GB
    cmem: int = 192 * GB

    ctog_bdw: float = 16 * GB
    g_bdw: float = 300 * GB
    c_bdw: float = 100 * GB

    gpu_flops: float = 104 * T
    cpu_flops: float = 13 * T

    tp_size: int = 1

    @classmethod
    def init(cls, model_config: ModelConfig, hardware_config: HardwareConfig, prompt_len, gen_len):
        tp_size = hardware_config.tp_size
        return cls(
            s=prompt_len,
            n=gen_len,
            l=model_config.num_hidden_layers,
            h1=model_config.hidden_size,
            h2=model_config.intermediate_size,
            nh=model_config.num_attention_heads,
            nkvh=model_config.num_key_value_heads,
            n_experts=model_config.num_local_experts,
            topk=model_config.topk,
            gmem=hardware_config.gmem * tp_size,
            cmem=hardware_config.cmem,
            ctog_bdw=hardware_config.ctog_bdw,
            g_bdw=hardware_config.g_bdw * tp_size,
            c_bdw=hardware_config.c_bdw,
            gpu_flops=hardware_config.gpu_flops * tp_size,
            cpu_flops=hardware_config.cpu_flops,
            tp_size=tp_size
        )
    
def dtype_size(dtype):
    if dtype == 'f32':
        return 4
    elif dtype == 'f16':
        return 2
    elif dtype == 'int8':
        return 1
    elif dtype == 'int4':
        return 0.5
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def attention_flops(batch_size, query_length, kv_sequence_length, num_q_head, head_dim):
    # FLOPs for dot products between 1 query and all keys (each key has 'head_dim' elements)
    dot_product_flops = batch_size * query_length * kv_sequence_length * num_q_head * head_dim * 2
    # Softmax and summing over all kv pairs for each query head
    softmax_sum_flops = batch_size * query_length * kv_sequence_length * num_q_head * 2
    return dot_product_flops + softmax_sum_flops

def attention_bytes(batch_size, query_length, kv_sequence_length, num_q_head, num_kv_head, head_dim, dtype="f16"):
    # Memory for 1 query, all keys, and all values
    type_size = dtype_size(dtype)
    query_memory = batch_size * query_length * num_q_head * head_dim * type_size
    key_memory = batch_size * kv_sequence_length * num_kv_head * head_dim * type_size
    value_memory = batch_size * kv_sequence_length * num_kv_head * head_dim * type_size
    # Output memory for the results of the attention mechanism
    output_memory = batch_size * query_length * num_q_head * head_dim * type_size
    return query_memory + key_memory + value_memory + output_memory


def MLP_flops(hidden_size1, hidden_size2, batch_size, topk):
    return 2 * batch_size * hidden_size1 * hidden_size2 * 3 * topk

def MLP_bytes(hidden_size1, hidden_size2, batch_size, n_experts):
    # Input, output, and weights; biases are typically small and can be neglected for this estimation
    input_bytes = 2 * batch_size * hidden_size1  # Input data bytes
    output_bytes = 2 * batch_size * hidden_size2  # Output data bytes
    weight_bytes = 2 * hidden_size1 * hidden_size2 * 3 * n_experts  # Weight data bytes
    return input_bytes + output_bytes + weight_bytes

def log_gpu_memory_usage(expert_size_bytes: int):
    num_gpus = torch.cuda.device_count()
    output = defaultdict(str)

    for i in range(num_gpus):
        # Get total memory in bytes and convert to GB
        total_memory_bytes = torch.cuda.get_device_properties(i).total_memory
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        
        # Get free memory in bytes and convert to GB
        free_memory_bytes = torch.cuda.mem_get_info(i)[0]
        free_memory_gb = free_memory_bytes / (1024 ** 3)
        
        # Get allocated memory in bytes and convert to GB
        allocated_memory_bytes = torch.cuda.memory_allocated(i)
        allocated_memory_gb = allocated_memory_bytes / (1024 ** 3)
        
        output[f"GPU_{i}_total_mem_capacity"] = float(total_memory_gb)
        output[f"GPU_{i}_mem_usage"] = float(allocated_memory_gb)
        output[f"GPU_{i}_mem_free"] = float(free_memory_gb)
        output[f"GPU_{i}_experts_can_fit"] = int(free_memory_bytes // expert_size_bytes)
        
    return output

def _log_moe_layer_data(self, batch_id: int, layer_id: int, experts_activated: List[int]):
        """Log GPU memory usage and expert activations for a MoE layer to CSV."""
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Get GPU memory usage
        memory_data = log_gpu_memory_usage(self.expert_size_bytes)
        
        # Prepare CSV row
        row = [timestamp, batch_id, layer_id, str(experts_activated)]
        
        # Add GPU memory data
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            row.extend([
                memory_data[f"GPU_{i}_total_mem_capacity"],
                memory_data[f"GPU_{i}_mem_usage"],
                memory_data[f"GPU_{i}_mem_free"],
                memory_data[f"GPU_{i}_experts_can_fit"]
            ])
        
        # Write to CSV file
        with open(self.csv_logger, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

def _init_csv_logger(self):
        """Initialize CSV logger for tracking GPU memory usage and expert activations."""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/moe_layer_memory_log_{timestamp}.csv"
        
        # Get number of GPUs for column headers
        num_gpus = torch.cuda.device_count()
        
        # Define CSV headers
        headers = ["timestamp", "batch_id", "moe_layer_id", "experts_activated"]
        
        # Add GPU memory columns for each GPU
        for i in range(num_gpus):
            headers.extend([
                f"GPU_{i}_total_mem_capacity",
                f"GPU_{i}_mem_usage", 
                f"GPU_{i}_mem_free",
                f"GPU_{i}_experts_can_fit"
            ])
        
        # Create CSV file and write headers
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        
        return filename