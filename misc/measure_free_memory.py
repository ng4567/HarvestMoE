from pynvml import *
from transformers import AutoConfig
import torch

def count_available_expert_memory(expert_size_bytes: int, gpuid: int) -> int:
    """"
    Counts the number of experts that can fit in the free memory of the given GPU.
    """
    
    handle = nvmlDeviceGetHandleByIndex(gpuid)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // expert_size_bytes

print("Starting free memory measurement script")

nvmlInit()
print ("Driver Version:", nvmlSystemGetDriverVersion())

model_id = "microsoft/Phi-3.5-MoE-instruct"
dtype_bytes = 2
num_gpus = torch.cuda.device_count()

print(f"Getting model config for model {model_id}")
config = AutoConfig.from_pretrained(model_id)

num_experts = config.num_local_experts

if hasattr(config, "num_local_experts") and hasattr(config, "hidden_size") and hasattr(config, "intermediate_size"):
    # Phi-3.5-MoE style: w1, w2, w3
    shapes = {
        "w1": (config.intermediate_size, config.hidden_size),
        "w2": (config.hidden_size, config.intermediate_size),
        "w3": (config.intermediate_size, config.hidden_size),
    }
else:
    raise ValueError(f"Cannot infer expert shapes for model '{model_id}' — missing expected config attributes.")

per_expert_params = sum(h * w for (h, w) in shapes.values())
total_params = per_expert_params * num_experts
per_expert_bytes = per_expert_params * dtype_bytes

print(f"Each expert has {per_expert_params} parameters and uses {per_expert_bytes} bytes")


for i in range(num_gpus):
    print(f"GPU {i}: can fit {count_available_expert_memory(per_expert_bytes, i)} experts")



