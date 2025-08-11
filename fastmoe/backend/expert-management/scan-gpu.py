import torch
from fastmoe.utils.model_config import ModelConfig
import os

num_gpus = torch.cuda.device_count()

model_config_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_config = ModelConfig(model_config_path)

dtype = torch.get_default_dtype()

elem_dim = 3 * (model_config.intermediate_size) * model_config.hidden_size
bytes_per_element = torch.tensor([], dtype=dtype).element_size()
expert_size = elem_dim * bytes_per_element

def get_num_experts_that_fit(expert_size: int, gpu_id: int):
    return torch.cuda.get_device_properties(gpu_id).total_memory // expert_size

if __name__ == "__main__":
    for gpu in range(num_gpus):
        print(f"GPU {gpu} can fit {get_num_experts_that_fit(expert_size, gpu)} experts")