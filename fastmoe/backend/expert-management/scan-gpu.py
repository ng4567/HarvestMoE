import torch
from fastmoe.utils.model_config import ModelConfig
import os

num_gpus = torch.cuda.device_count()

mc = ModelConfig("mistralai/Mixtral-8x7B-Instruct-v0.1")

# If you add TP later, set this accordingly
tp_size = 1
# Match model param dtype (fp16)
dtype = torch.get_default_dtype()
bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

# Per expert per layer (per TP rank)
elem_dim_per_layer = 3 * (mc.intermediate_size // tp_size) * mc.hidden_size
bytes_per_expert_per_layer = elem_dim_per_layer * bytes_per_elem
# Full expert across all layers on this rank
bytes_per_expert_all_layers = mc.num_hidden_layers * bytes_per_expert_per_layer


def get_num_experts_that_fit(gpu_id: int, full_expert: bool = True, safety: float = 0.8) -> int:
    # Query free memory on the target device
    with torch.cuda.device(gpu_id):
        free_bytes, _ = torch.cuda.mem_get_info()
    budget = int(free_bytes * safety)
    unit = bytes_per_expert_all_layers if full_expert else bytes_per_expert_per_layer
    return budget // unit


if __name__ == "__main__":
    for gpu in range(num_gpus):
        full = get_num_experts_that_fit(gpu, full_expert=True)
        per_layer = get_num_experts_that_fit(gpu, full_expert=False)
        print(f"GPU {gpu}: full experts={full}, per-layer experts={per_layer}")
