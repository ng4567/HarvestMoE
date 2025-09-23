# Overview

This repo is a research project by [Nikhil Gopal](https://www.linkedin.com/in/nikhil-gopal/) & Professor [Kostis Kaffes](https://www.cs.columbia.edu/~kkaffes/). It is a fork of the [MoE Lightning repository](https://github.com/caoshiyi/artifacts/tree/asplos25).

MoE Lightning was designed as a pipelined inference framework for Mixture of Experts transformers. The goal was to hide the latency of model weight fetches from DRAM. Our additions build upon the original by extending MoE Lightning to multi-GPU setups and enabling dynamic re-allocation of expert weights. [Nvidia NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) provides an interface to transfer model weights between GPUs at much faster speeds compared to fetching from DRAM over PCIE. We added a feature inside the server API to keep track of Expert locations in memory, and to move them from DRAM to a GPU. On another thread, a program can continuously query this API as well as GPU memory usage to determine if more can be allocated into GPU memory. By allocating experts into open pockets of GPU memory on a cluster, we further reduce the impact of cache misses, and allow for the performance of the system to scale dynamically as memory usage evolves.

Note that all testing was done using an Azure Standard_NC80adis_H100_v5 Instance with 2 H100 GPUs on Ubuntu 24.04.3 LTS. 

## Technology Stack

### Core ML Infrastructure
- **PyTorch 2.1.2** - Deep learning framework providing tensor operations and GPU acceleration
- **vLLM (0.2.7-0.4.1)** - High-performance LLM serving library for efficient inference and memory management
- **Triton 2.2.0** - GPU kernel compiler enabling custom CUDA kernels for MoE routing and Flash Attention

### Model & Tokenization
- **HuggingFace Transformers** - Model loading, tokenization, and compatibility with Mixtral/DBRX architectures

### API & Serving Layer
- **FastAPI** - Modern async web framework for REST API endpoints (expert tracking, reallocation)
- **Uvicorn + Uvloop** - High-performance ASGI server with optimized event loop for handling concurrent requests

### System Optimization
- **Intel MKL** - CPU-optimized math operations for fallback computations
- **CUDA/NVLink** - GPU-to-GPU communication for fast expert weight transfers

## To-Do List:

### Completed
- ✅ implement KV Cache size and block size configuration in server API
- ✅ implement paged KV cache infrastructure with configurable block size (use --use-paged-kv-cache flag)

### Paged KV Cache - Missing Components

#### 1. **Block Storage and Retrieval**
   - **Current**: `store_kv_cache()` is a no-op for paged cache - KV data is never actually stored
   - **Fix**: Implement `PagedKVCache.write_kv()` to store keys/values in allocated blocks
   - **Fix**: Track which blocks belong to each sequence (sequence ID → block indices mapping)

#### 2. **CPU Attention Support**
   - **Current**: Uses zero-filled temporary buffers instead of actual KV data
   - **Fix**: Implement block-to-continuous buffer copying for CPU attention
   - **Fix**: Or better: Create block-aware CPU attention kernel that reads directly from blocks
   - **Fix**: Add buffer pooling to avoid repeated allocations

#### 3. **Block Lifecycle Management**
   - **Current**: No actual block allocation per sequence during prefill
   - **Fix**: Allocate blocks in `prepare_for_prefill()` based on sequence length
   - **Fix**: Free blocks when sequences complete or are evicted
   - **Fix**: Implement proper cleanup in `reset()` method

#### 4. **Attention Kernels**
   - **Current**: Standard attention kernels expect continuous KV buffers
   - **Fix**: Implement paged attention kernels (GPU) that can read from non-continuous blocks
   - **Fix**: Implement efficient block gathering for existing kernels
   - **Fix**: Add block prefetching for better memory access patterns

#### 5. **Capacity Management**
   - **Current**: Uses hardcoded capacity (2048) for paged cache
   - **Fix**: Calculate actual available capacity from free blocks
   - **Fix**: Implement admission control based on available blocks
   - **Fix**: Add block eviction policy when capacity is exceeded

#### 6. **Memory Movement**
   - **Current**: No support for moving KV cache blocks between GPU/CPU
   - **Fix**: Implement block swapping between GPU and CPU memory
   - **Fix**: Add async block transfers to hide latency
   - **Fix**: Integrate with expert movement infrastructure

### Implementation Steps

1. **Phase 1: Basic Functionality**
   - Implement actual KV storage in blocks
   - Add sequence-to-block mapping
   - Fix CPU attention to read from blocks (even if inefficient)

2. **Phase 2: Optimization**
   - Create efficient block-aware attention kernels
   - Add buffer pooling and reuse
   - Implement block prefetching

3. **Phase 3: Advanced Features**
   - Block sharing for sequences with common prefixes
   - Dynamic block movement between GPU/CPU
   - Integration with MoE Lightning's pipelining

### Other To-Dos
- add KV Cache location tracking and movement between DRAM/GPUs (for paged cache blocks)
- benchmark SOTA models vs standard MoE Lightning with paged attention

## Pre-Installation:

You will need nvidia drivers and g++

```bash
sudo apt-get install g++
sudo apt-get install -y software-properties-common
sudo apt-get install -y nvidia-driver-535
sudo apt-get install -y nvidia-cuda-toolkit
```

## Installation

```bash
git clone -b asplos-artifact https://github.com/ng4567/moe-lightning-fork.git
cd FastMoE
conda create -n fastmoe python=3.11
conda activate fastmoe
conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge onemkl-sycl-blas
conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge mkl-include
pip install -e .
pip install triton==2.2.0
```

log into hugging face cli: `huggingface-cli login` (make sure to use a read token)

## Launch Inference Server

```bash
python -m fastmoe.serve.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000 --cpu-mem-bdw 76 --avg-prompt-len 77 --gen-len 32
```

## Curl Inference Server:

```bash
curl -s -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":["Hello, world!"],"sampling_params":{"max_new_tokens":32},"batch":true,"stream":false}'
```

## Query Expert Location

```bash
# After starting the server, query expert locations:
curl http://localhost:30000/expert_locations | python -m json.tool

# Get info for a specific layer:
curl http://localhost:30000/expert_locations/layer/0
```

### What It Shows

- **Memory Distribution**: Where experts are located (GPU/CPU/Cache)
- **Capacity Utilization**: How many experts can fit on GPU vs actual usage
- **Layer Statistics**: Per-layer expert distribution

Example output:
```json
{
  "summary": {
    "total_experts": 256,
    "gpu_capacity": 0,      // Experts that can be permanently stored on GPU
    "cache_capacity": 16,   // Dynamic buffer size
    "global_location_distribution": {
      "gpu_persistent": 0,  // Permanently on GPU
      "cpu_memory": 256,    // In CPU memory
      "gpu_cache": 0        // Temporarily on GPU
    }
  }
}
```

## Expert Movement

### Move Experts with cURL
```bash
# Move expert from GPU to CPU
curl -X POST http://localhost:30000/expert_reallocation/request \
  -H "Content-Type: application/json" \
  -d '{"layer_id": 8, "expert_id": 1, "action": "move_to_cpu"}'

# Move expert from CPU to GPU  
curl -X POST http://localhost:30000/expert_reallocation/request \
  -H "Content-Type: application/json" \
  -d '{"layer_id": 8, "expert_id": 3, "action": "move_to_gpu"}'

# Check reallocation stats
curl http://localhost:30000/expert_reallocation/stats
```

### Move Experts with Python
```python
import requests

# Move to CPU
response = requests.post("http://localhost:30000/expert_reallocation/request", 
    json={"layer_id": 8, "expert_id": 1, "action": "move_to_cpu"})

# Move to GPU (if capacity available)
response = requests.post("http://localhost:30000/expert_reallocation/request",
    json={"layer_id": 8, "expert_id": 3, "action": "move_to_gpu"})

# Check memory usage
stats = requests.get("http://localhost:30000/expert_reallocation/stats").json()
print(f"GPU: {stats['memory']['gpu_allocated_gb']}GB, CPU: {stats['memory']['cpu_expert_storage_gb']}GB")
```

**Note**: GPU capacity is limited by the `wg` parameter. Moves are non-blocking and won't interrupt inference.

## KV Cache Configuration

**Note**: By default, the KV cache uses a continuous buffer with size determined by the optimizer policy. To enable the paged KV cache implementation with configurable size and block size, use the `--use-paged-kv-cache` flag.

### Query KV Cache Configuration
```bash
# Get current KV cache size and block size (stored values only)
curl http://localhost:30000/kv_cache_config | python -m json.tool
```

Example output:
```json
{
  "kv_cache_size_gb": 4.0,
  "kv_block_size": 16,
  "kv_cache_size_bytes": 4294967296,
  "use_paged_kv_cache": false,
  "computed_capacity": 2048,
  "override_capacity": null,
  "message": "Current KV cache configuration (Using default continuous buffer, values shown are configuration only)"
}
```

The `computed_capacity` field shows the KV cache capacity in tokens, calculated based on:
- The hierarchical roofline model's optimization (determines optimal batch sizes)
- Available free blocks in the paged KV cache (when using `--use-paged-kv-cache`)
- The smaller of these two values is used as the effective capacity

### Update KV Cache Configuration
```bash
# Update KV cache size to 8GB
curl -X POST http://localhost:30000/kv_cache_config/update \
  -H "Content-Type: application/json" \
  -d '{"kv_cache_size_gb": 8.0}'

# Update block size to 32 (must be power of 2)
curl -X POST http://localhost:30000/kv_cache_config/update \
  -H "Content-Type: application/json" \
  -d '{"kv_block_size": 32}'

# Update both parameters at once
curl -X POST http://localhost:30000/kv_cache_config/update \
  -H "Content-Type: application/json" \
  -d '{"kv_cache_size_gb": 6.0, "kv_block_size": 64}'

# Override the computed capacity (in tokens)
curl -X POST http://localhost:30000/kv_cache_config/update \
  -H "Content-Type: application/json" \
  -d '{"override_capacity": 4096}'
  
# Clear override (use computed capacity)
curl -X POST http://localhost:30000/kv_cache_config/update \
  -H "Content-Type: application/json" \
  -d '{"override_capacity": 0}'
```

### Using Python to Configure KV Cache
```python
import requests

# Query current configuration
response = requests.get("http://localhost:30000/kv_cache_config")
config = response.json()
print(f"Current KV cache: {config['kv_cache_size_gb']}GB, Block size: {config['kv_block_size']}")
print(f"Computed capacity: {config.get('computed_capacity')} tokens")
print(f"Override capacity: {config.get('override_capacity')} tokens")

# Update configuration
response = requests.post("http://localhost:30000/kv_cache_config/update",
    json={"kv_cache_size_gb": 8.0, "kv_block_size": 32})
print(response.json()["message"])

# Override the computed capacity
response = requests.post("http://localhost:30000/kv_cache_config/update",
    json={"override_capacity": 4096})
print(f"Set override capacity to: {response.json().get('override_capacity')} tokens")
```

### Launch Server with Custom KV Cache Settings

#### With Default Continuous Buffer (configuration stored but not actively used):
```bash
python -m fastmoe.serve.launch_server \
  --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --port 30000 \
  --kv-cache-size-gb 8.0 \
  --kv-block-size 32 \
  --cpu-mem-bdw 76 \
  --avg-prompt-len 77 \
  --gen-len 32
```

#### With Paged KV Cache Implementation (actively uses configured values):
```bash
python -m fastmoe.serve.launch_server \
  --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --port 30000 \
  --use-paged-kv-cache \
  --kv-cache-size-gb 8.0 \
  --kv-block-size 32 \
  --kv-cache-gpu-fraction 0.2 \
  --cpu-mem-bdw 76 \
  --avg-prompt-len 77 \
  --gen-len 32
  
# With override capacity (optional)
python -m fastmoe.serve.launch_server \
  --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --port 30000 \
  --use-paged-kv-cache \
  --kv-cache-size-gb 8.0 \
  --kv-block-size 32 \
  --kv-cache-gpu-fraction 0.2 \
  --kv-cache-override-capacity 4096 \
  --cpu-mem-bdw 76 \
  --avg-prompt-len 77 \
  --gen-len 32
```

When using `--use-paged-kv-cache`:
- The KV cache is divided into fixed-size blocks of `kv_block_size` tokens
- Total cache size is limited to `kv_cache_size_gb` GB
- `--kv-cache-gpu-fraction` controls the split (default: 0.2 = 20% GPU, 80% CPU)
- Blocks are dynamically allocated/freed as sequences are processed
- Block size must be a power of 2 (e.g., 16, 32, 64)

## Run Test Bench

```bash
cd benchmarks/mtbench
python bench.py --port 30000 --max-new-tokens 32 --ubs 324 --n-ub 14
```

