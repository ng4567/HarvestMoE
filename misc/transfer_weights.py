import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors import safe_open
import os
import time
import threading
import concurrent.futures
from huggingface_hub import snapshot_download

print("Starting script...")

model_name = "microsoft/Phi-3.5-MoE-instruct"
# Skip model and tokenizer loading for transfer speed benchmark
print("⚡ Skipping model/tokenizer loading (transfer benchmark only)")
model = None
tokenizer = None
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

loaded_experts = set()
num_experts = config.num_local_experts
all_expert_indices = set(range(num_experts))

# Download model files locally (with caching)
print(f"Checking for cached model: {model_name}")
try:
    # First try to get from cache without downloading
    model_path = snapshot_download(
        model_name,
        local_files_only=True,  # Only use local cache
        cache_dir=None  # Use default cache directory
    )
    print(f"✅ Found cached model at: {model_path}")
except Exception:
    # If not in cache, download it
    print(f"📥 Model not cached, downloading: {model_name}")
    model_path = snapshot_download(
        model_name,
        local_files_only=False,  # Allow downloading
        cache_dir=None,  # Use default cache directory
        resume_download=True  # Resume partial downloads
    )
    print(f"✅ Model downloaded to: {model_path}")

# Find all safetensors files
safetensor_files = [
    os.path.join(model_path, f) for f in os.listdir(model_path)
    if f.endswith('.safetensors')
]

# Cache keys for expert-related files only (skip non-expert files)
print("Caching keys for expert-related safetensor files...")
safetensor_keys_cache = {}
expert_file_count = 0

for safetensor_file in safetensor_files:
    with safe_open(safetensor_file, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        # Only cache files that contain expert weights
        expert_keys = {key for key in all_keys if "experts" in key}
        if expert_keys:
            safetensor_keys_cache[safetensor_file] = expert_keys
            expert_file_count += 1
#            print(f"Cached {len(expert_keys)} expert keys from {os.path.basename(safetensor_file)}")

print(f"Expert key caching complete for {expert_file_count}/{len(safetensor_files)} files")

# Global expert tensor cache - stores standalone expert weights
expert_tensors_cache = {}  # {expert_idx: {param_name: tensor}}

def check_nvlink_status():
    """Check if NVLink is available between GPUs"""
    try:
        # Check if we have multiple GPUs
        if torch.cuda.device_count() < 2:
            return False, "Only 1 GPU available"
        
        # Check P2P access
        can_access_peer = torch.cuda.can_device_access_peer(0, 1)
        
        # Get device properties
        props_0 = torch.cuda.get_device_properties(0)
        props_1 = torch.cuda.get_device_properties(1)
        
        return can_access_peer, f"GPU 0: {props_0.name}, GPU 1: {props_1.name}"
    except Exception as e:
        return False, f"Error checking NVLink: {e}"

def main():
    print("=== MoE Expert Transfer Speed Benchmark ===")
    print("Testing: CPU→GPU vs GPU→GPU transfer speeds")
    
    # Check NVLink status
    nvlink_available, nvlink_info = check_nvlink_status()
    print(f"🔗 NVLink Status: {'✅ Available' if nvlink_available else '❌ Not Available'}")
    print(f"   {nvlink_info}")
    print()
    
    # Test parameters
    experts_to_test = [8, 9, 10]
    
    print(f"📊 Benchmark Parameters:")
    print(f"Testing Streaming of experts: {experts_to_test}")
    
    # =========================
    # BENCHMARK 1: CPU → GPU 1
    # =========================
    print("🔄 BENCHMARK 1: CPU → GPU 1 Transfer")
    print("Loading expert weights from disk to GPU 1...")
    
    torch.cuda.synchronize(1)          # ensure previous GPU‑1 work is done
    cpu_to_gpu_start = time.time()
    cpu_to_gpu_result = download_expert_weights_standalone(experts_to_test, target_gpu=1)
    torch.cuda.synchronize(1)          # wait for queued CPU→GPU copies to finish
    cpu_to_gpu_end = time.time()
    
    cpu_to_gpu_time = cpu_to_gpu_end - cpu_to_gpu_start
    
    print(f"✅ CPU → GPU 1: {cpu_to_gpu_time:.4f} seconds")
    print(f" Loaded {cpu_to_gpu_result} parameters")
    
    # Check GPU 1 memory after loading
    gpu1_mem_after_load = torch.cuda.memory_allocated(1) / 1e9
    print(f"GPU 1 memory after load: {gpu1_mem_after_load:.2f} GB")
    
    # =========================
    # BENCHMARK 2: GPU 1 → GPU 0  
    # =========================
    print("🚀 BENCHMARK 2: GPU 1 → GPU 0 Transfer")
    print("Streaming expert weights from GPU 1 to GPU 0...")
    
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)   # clear both queues
    gpu_to_gpu_start = time.time()
    gpu_to_gpu_result = transfer_experts_gpu_to_gpu(experts_to_test, src_gpu=1, dest_gpu=0)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)   # wait for peer copies
    gpu_to_gpu_end = time.time()
    
    gpu_to_gpu_time = gpu_to_gpu_end - gpu_to_gpu_start
    
    print(f"✅ GPU 1 → GPU 0: {gpu_to_gpu_time:.4f} seconds")
    print(f"   Transferred {gpu_to_gpu_result} experts")
    print()
    
    # =========================
    # RESULTS COMPARISON
    # =========================
    print("📊 BENCHMARK RESULTS:")
    print("=" * 50)
    print(f"CPU → GPU Transfer:  {cpu_to_gpu_time:.4f} seconds")
    print(f"GPU → GPU Transfer:  {gpu_to_gpu_time:.4f} seconds")
    print()
    
    if gpu_to_gpu_time > 0:
        speedup = cpu_to_gpu_time / gpu_to_gpu_time
        print(f"🚀 GPU→GPU Speedup:   {speedup:.2f}x faster")
        
        if speedup > 1:
            print(f"✅ GPU→GPU is {speedup:.2f}x faster than CPU→GPU!")
        else:
            print(f"⚠️  CPU→GPU is {1/speedup:.2f}x faster than GPU→GPU")
            if not nvlink_available:
                print("   💡 This may be due to lack of NVLink - using PCIe instead")
    else:
        print("❌ GPU→GPU transfer failed")
    
    print("=" * 50)
    
    # Memory usage report
    print("\n💾 Memory Usage Report:")
    gpu0_allocated = torch.cuda.memory_allocated(0) / 1e9
    gpu1_allocated = torch.cuda.memory_allocated(1) / 1e9
    print(f"GPU 0 memory: {gpu0_allocated:.2f} GB")
    print(f"GPU 1 memory: {gpu1_allocated:.2f} GB")
    
    print("\nBenchmark complete! 🎯")

def transfer_experts_gpu_to_gpu(expert_indices: list[int], src_gpu: int, dest_gpu: int) -> int:
    """Transfer expert weights from source GPU to destination GPU"""
    
    for expert_idx in expert_indices:
        # Stream expert from source GPU to destination GPU
        stream_result = stream_expert_between_gpus(expert_idx, dest_gpu)
        if stream_result < 0:
            print(f"Failed to stream expert {expert_idx}")
            exit(1)
    
    return 0

def load_non_moe_weights(gpu_id: int) -> int:
    '''Load the non-moe weights into the model on gpu 0'''
    device = f"cuda:{gpu_id}"
    loaded_count = 0
    
    print(f"Loading non-MoE weights directly to GPU {gpu_id}")
    
    for safetensor_file in safetensor_files:
        print(f"Processing {os.path.basename(safetensor_file)}")

        try:
            with safe_open(safetensor_file, framework="pt", device=device) as f:
                # Use cached keys instead of reading them again
                cached_keys = safetensor_keys_cache[safetensor_file]
                for key in cached_keys:
                    # Load only non-expert weights (skip anything with "experts" in the name)
                    if "experts" not in key:
                        try:
                            # Get tensor directly on GPU
                            tensor = f.get_tensor(key)
                            # Navigate to the parameter in model
                            param = model
                            for attr in key.split('.'):
                                param = getattr(param, attr)
                            # Copy the weight (already on correct device)
                            with torch.no_grad():
                                param.copy_(tensor)
                            
                            loaded_count += 1
                            
                        except Exception as e:
                            print(f"Failed to load {key} directly to GPU: {e}")
                            
        except Exception as e:
            print(f"Direct GPU loading failed for {safetensor_file}, error: {e}")
            print("Exiting now")
            exit(1)
    
    print(f"Successfully loaded {loaded_count} non-MoE parameters to GPU {gpu_id}")
    return 0

def load_moe_weights_cpu_to_gpu(model: AutoModelForCausalLM, gpu_id: int, expert_indices: list[int]) -> int:
    '''Load the moe weights into the model on gpu 0'''
    device = f"cuda:{gpu_id}"
    loaded_count = 0
    
    print(f"Loading MoE experts {expert_indices} to GPU {gpu_id}")
    
    for safetensor_file in safetensor_files:
        print(f"Processing {os.path.basename(safetensor_file)} for experts")
        
        try:
            # Attempt direct GPU loading
            with safe_open(safetensor_file, framework="pt", device=device) as f:
                # Use cached keys and build target list efficiently
                cached_keys = safetensor_keys_cache[safetensor_file]
                target_keys = []
                
                # Build target keys for our specific experts
                for key in cached_keys:
                    for expert_idx in expert_indices:
                        if f"experts.{expert_idx}." in key:
                            target_keys.append(key)
                            break
                
                # Load only the keys we actually need
                for key in target_keys:
                    try:
                        # Get tensor directly on GPU
                        tensor = f.get_tensor(key)
                        # Navigate to the parameter in model
                        param = model
                        for attr in key.split('.'):
                            param = getattr(param, attr)
                        # Copy the weight (already on correct device)
                        with torch.no_grad():
                            param.copy_(tensor)
                        
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"Failed to load {key} directly to GPU: {e}")
                            
        except Exception as e:
            print(f"Direct GPU loading failed for {safetensor_file}, falling back to CPU staging: {e}")
            print("Exiting now")
            exit(1)
    
    loaded_experts.update(expert_indices)
    print(f"Successfully loaded {loaded_count} MoE expert parameters for experts {expert_indices} to GPU {gpu_id}")
    return 0

def check_missing_weights(model: AutoModelForCausalLM) -> list[int]:
    '''Check which weights arent loaded in the model on gpu 0, return an array of ints of the missing weights'''
    return list(all_expert_indices - loaded_experts)

def transfer_weights(src_gpu_id: int, dest_gpu_id: int, missing_weights: list[int]) -> int:
    """Transfer the missing weights from gpu 1 to gpu 0"""
    print(f"Transferring {len(missing_weights)} experts from GPU {src_gpu_id} to GPU {dest_gpu_id}")
    
    transferred_count = 0
    
    for expert_idx in missing_weights:
        print(f"Transferring expert {expert_idx}...")
        
        # Stream expert from source GPU to destination GPU
        stream_result = stream_expert_between_gpus(expert_idx, dest_gpu_id)
        if stream_result > 0:
            # Install expert into model
            install_result = install_expert_to_model(model, expert_idx)
            if install_result > 0:
                transferred_count += 1
                print(f"✅ Expert {expert_idx} transferred successfully")
            else:
                print(f"❌ Failed to install expert {expert_idx}")
        else:
            print(f"❌ Failed to stream expert {expert_idx}")
    
    print(f"Successfully transferred {transferred_count}/{len(missing_weights)} experts")
    return transferred_count

def forward_pass(model) -> int:
    """Perform forward pass with the question about South Korea's capital"""
    question = "What is the capital of South Korea?"
    print(f"Running forward pass with question: '{question}'")
    
    try:
        # Tokenize input
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        print(f"Input tokens: {inputs['input_ids'].shape}")
        
        # Generate response
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated response: {response}")
        
        return 0  # Success
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return -1

def download_expert_weights_standalone(expert_indices: list[int], target_gpu: int) -> int:
    """Download expert weights as standalone tensors to target GPU"""
    device = f"cuda:{target_gpu}"
    loaded_count = 0
    
    print(f"Downloading expert weights {expert_indices} as standalone tensors to GPU {target_gpu}")
    
    for expert_idx in expert_indices:
        expert_tensors_cache[expert_idx] = {}
    
    # Only process files that have expert weights (skip others)
    for safetensor_file in safetensor_files:
        if safetensor_file not in safetensor_keys_cache:
            continue  # Skip files with no expert weights
            
        try:
            with safe_open(safetensor_file, framework="pt", device=device) as f:
                cached_keys = safetensor_keys_cache[safetensor_file]
                
                # Build target keys for our specific experts
                target_keys = []
                for key in cached_keys:
                    for expert_idx in expert_indices:
                        if f"experts.{expert_idx}." in key:
                            target_keys.append((key, expert_idx))
                            break
                
                # Load expert weights as standalone tensors
                for key, expert_idx in target_keys:
                    try:
                        # Load tensor directly to GPU as standalone tensor
                        tensor = f.get_tensor(key)
                        expert_tensors_cache[expert_idx][key] = tensor
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"Failed to download {key}: {e}")
                        
        except Exception as e:
            print(f"Failed to process {safetensor_file}: {e}")
            return -1
    
    print(f"Downloaded {loaded_count} expert tensor parameters for experts {expert_indices} to GPU {target_gpu}")
    torch.cuda.synchronize(target_gpu)     # ensure all CPU→GPU transfers finished
    return loaded_count

def stream_expert_between_gpus(expert_idx: int, dest_gpu: int) -> int:
    src_tensors = expert_tensors_cache[expert_idx]
    # Ensure destination cache entry exists
    dest_key = f"{expert_idx}@{dest_gpu}"
    if dest_key not in expert_tensors_cache:
        expert_tensors_cache[dest_key] = {}
    dest_device  = torch.device(f'cuda:{dest_gpu}')
    torch.cuda.set_device(dest_gpu)

    # ONE stream for all copies → keeps engines busy
    stream = torch.cuda.Stream(device=dest_gpu)
    with torch.cuda.stream(stream):
        for name, src in src_tensors.items():
            # 1. pre-allocate once
            dst = torch.empty_like(src, device=dest_device, memory_format=torch.contiguous_format)
            # 2. peer copy (async)
            dst.copy_(src, non_blocking=True)
            expert_tensors_cache[f"{expert_idx}@{dest_gpu}"][name] = dst

    # 3. let the CPU overlap while the GPU engines run
    stream.synchronize()
    return len(src_tensors)

def install_expert_to_model(model: AutoModelForCausalLM, expert_idx: int) -> int:
    """Install cached expert tensors into model structure"""
    if expert_idx not in expert_tensors_cache:
        print(f"Expert {expert_idx} not found in tensor cache")
        return -1
    
    installed_count = 0
    
    print(f"Installing expert {expert_idx} tensors into model")
    
    for param_name, tensor in expert_tensors_cache[expert_idx].items():
        try:
            # Navigate to model parameter
            param = model
            for attr in param_name.split('.'):
                param = getattr(param, attr)
            
            # Copy cached tensor into model
            with torch.no_grad():
                param.copy_(tensor)
            
            installed_count += 1
            
        except Exception as e:
            print(f"Failed to install {param_name}: {e}")
            return -1
    
    # Update loaded experts tracking
    loaded_experts.add(expert_idx)
    
    print(f"Installed {installed_count} parameters for expert {expert_idx} into model")
    return installed_count

def get_expert_memory_usage(expert_idx: int) -> float:
    """Get memory usage of cached expert in GB"""
    if expert_idx not in expert_tensors_cache:
        return 0.0
    
    total_bytes = 0
    for tensor in expert_tensors_cache[expert_idx].values():
        total_bytes += tensor.element_size() * tensor.numel()
    
    return total_bytes / (1024**3)  # Convert to GB

if __name__ == "__main__":
    main()
