from transformers import AutoConfig
import torch
import time

model_id = "microsoft/Phi-3.5-MoE-instruct"
dtype_bytes = 2
num_trials = 1000


print("-----Testing if GPU count >=2 ------\n")
assert torch.cuda.device_count() >= 2, "GPU count must be >=2"
print("GPU count is >=2, continuing...")

print("-----Testing if NVLink is available------\n")
try:
    # Check if we have multiple GPUs
    if torch.cuda.device_count() < 2:
        raise Exception("Less than 2 GPUs available")
    
    # Check P2P access
    can_access_peer = torch.cuda.can_device_access_peer(0, 1)
    
    # Get device properties
    props_0 = torch.cuda.get_device_properties(0)
    props_1 = torch.cuda.get_device_properties(1)
    
    print(f"GPU 0: {props_0.name}, GPU 1: {props_1.name}")
except Exception as e:
    raise Exception(f"Error checking NVLink: {e}")

print("NVLink is available, warming up CUDA...")
# Warm up CUDA on both GPUs
_ = torch.ones((1,), device="cuda:0")
_ = torch.ones((1,), device="cuda:1")
print("CUDA warm-up completed for both GPUs")
print("Getting model config...")

config = AutoConfig.from_pretrained(model_id)
#print(f"Model {model_id} config: {config}\n")
num_experts = config.num_local_experts

# Infer expert weight shapes programmatically
# Common structure: 2 or 3 Linear layers with shapes involving hidden and intermediate sizes
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
total_bytes = total_params * dtype_bytes

details = {
    "expert_shapes": shapes,
    "params_per_expert": per_expert_params,
    "total_params_all_experts": total_params,
    "bytes_per_expert": per_expert_bytes,
    "total_bytes_all_experts": total_bytes,
    "num_experts": num_experts,
    "dtype": f"fp{dtype_bytes * 8}"
}

print(f"\nModel {model_id} details: {details}\n")

print("------starting CPU ↔ GPU back-and-forth transfer test ------\n")

# Create a tensor on CPU (this will be our "persistent" tensor)
expert_tensor = torch.randn(
    per_expert_params, 
    device="cpu", 
    dtype=torch.float16
).pin_memory()

expert_size_mb = expert_tensor.nelement() * expert_tensor.element_size() / (1024 * 1024)
print(f"Created tensor of size {expert_size_mb:.2f} MB on CPU")

# Current tensor location: starts on CPU
current_tensor = expert_tensor

print(f"Testing {num_trials} CPU>GPU transfer speed...")

torch.cuda.synchronize()
start = time.perf_counter()
for i in range(num_trials):
    print(f"Round-trip {i+1}/{num_trials}")
    
    expert_tensor = expert_tensor.to("cuda:0", non_blocking=True)
    torch.cuda.synchronize()
    
    expert_tensor = expert_tensor.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
cpu_gpu_transfer_time = time.perf_counter() - start

avg_time = cpu_gpu_transfer_time / num_trials
cpu_gpu_transfer_speed_gbps = (expert_size_mb / 1024) / avg_time * 8
one_way_speed_gbps = cpu_gpu_transfer_speed_gbps / 2

print(f"\nAverage transfer time: {avg_time * 1000:.2f} ms")
print(f"Average transfer speed: {cpu_gpu_transfer_speed_gbps:.2f} Gb/s")
print(f"Estimated one-way speed: {one_way_speed_gbps:.2f} Gb/s\n")

print("------starting GPU > GPU transfer test ------\n")

# Warm-up
expert_tensor = torch.randn(per_expert_params, device="cuda:0", dtype=torch.float16)
_ = expert_tensor.to("cuda:1")
_ = expert_tensor.to("cuda:0")
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for i in range(num_trials):
    print(f"Round-trip {i+1}/{num_trials}")
    expert_tensor = expert_tensor.to("cuda:1", non_blocking=True)
    torch.cuda.synchronize()
    expert_tensor = expert_tensor.to("cuda:0", non_blocking=True)
    torch.cuda.synchronize()
gpu_gpu_transfer_time = time.perf_counter() - start

avg_time = gpu_gpu_transfer_time / num_trials
gpu_gpu_transfer_speed_gbps = (expert_size_mb / 1024) / avg_time * 8
one_way_speed_gbps = gpu_gpu_transfer_speed_gbps / 2

print(f"\nAverage transfer time: {avg_time * 1000:.2f} ms")
print(f"Average transfer speed: {gpu_gpu_transfer_speed_gbps:.2f} Gb/s (round-trip)")
print(f"Estimated one-way speed: {one_way_speed_gbps:.2f} Gb/s\n")

print("------Final results------\n")

print(f"Num trials: {num_trials}")
if cpu_gpu_transfer_speed_gbps > gpu_gpu_transfer_speed_gbps:
    print(f"CPU > GPU speed: {cpu_gpu_transfer_speed_gbps:.2f} Gb/s transfer is faster than GPU {gpu_gpu_transfer_speed_gbps:.2f} GB/s")
else:
    print(f"GPU > GPU transfer speed {gpu_gpu_transfer_speed_gbps:.2f} GB/s is faster than CPU > GPU transfer {cpu_gpu_transfer_speed_gbps:.2f} Gb/s")