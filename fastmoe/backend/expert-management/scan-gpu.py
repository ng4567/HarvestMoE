import torch
from fastmoe.utils.model_config import ModelConfig
import os

num_gpus = torch.cuda.device_count()

mc = ModelConfig("mistralai/Mixtral-8x7B-Instruct-v0.1")

tp_size = 1
dtype = torch.get_default_dtype()
bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

elem_dim_per_layer = 3 * (mc.intermediate_size // tp_size) * mc.hidden_size
bytes_per_expert_per_layer = elem_dim_per_layer * bytes_per_elem
bytes_per_expert_all_layers = mc.num_hidden_layers * bytes_per_expert_per_layer

def get_num_experts_that_fit(gpu_id: int, full_expert=True, safety=0.8):
    free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
    budget = int(free_bytes * safety)
    unit = bytes_per_expert_all_layers if full_expert else bytes_per_expert_per_layer
    return budget // unit

if __name__ == "__main__":
    for gpu in range(num_gpus):
        print(f"GPU {gpu} can fit {get_num_experts_that_fit(bytes_per_expert_per_layer, gpu)} experts")
