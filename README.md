# Overview

This repo is a research project by [Nikhil Gopal](https://www.linkedin.com/in/nikhil-gopal/) & Professor [Kostis Kaffes](https://www.cs.columbia.edu/~kkaffes/). It is a fork of the [MoE Lightning repository](https://github.com/caoshiyi/artifacts/tree/asplos25).

MoE Lightning was designed as a pipeline inference framework for Mixture of Experts transformers. The goal was to hide the latency of model weight fetches from DRAM. Our additions build upon the original by extending MoE Lightning to multi-GPU setups and enabling dynamic re-allocation of expert weights. [Nvidia NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) provides an interface to transfer model weights between GPUs at much faster speeds compared to fetching from DRAM over PCIE. We added a feature inside the server API to keep track of Expert locations in memory, and to move them from DRAM to a GPU. On another thread, a program can continuously query this API as well as GPU memory usage to determine if more can be allocated into GPU memory. By allocating experts into open pockets of GPU memory on a cluster, we further reduce the impact of cache misses, and allow for the performance of the system to scale dynamically as memory usage evolves.

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

- implement KV Cache location tracking and movement to server API
- add ability to move KV cache between DRAM/GPUs
- benchmark SOTA models vs standard MoE Lightning

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
curl http://localhost:8000/expert_locations | python -m json.tool

# Get info for a specific layer:
curl http://localhost:8000/expert_locations/layer/0
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
curl -X POST http://localhost:8000/expert_reallocation/request \
  -H "Content-Type: application/json" \
  -d '{"layer_id": 8, "expert_id": 1, "action": "move_to_cpu"}'

# Move expert from CPU to GPU  
curl -X POST http://localhost:8000/expert_reallocation/request \
  -H "Content-Type: application/json" \
  -d '{"layer_id": 8, "expert_id": 3, "action": "move_to_gpu"}'

# Check reallocation stats
curl http://localhost:8000/expert_reallocation/stats
```

### Move Experts with Python
```python
import requests

# Move to CPU
response = requests.post("http://localhost:8000/expert_reallocation/request", 
    json={"layer_id": 8, "expert_id": 1, "action": "move_to_cpu"})

# Move to GPU (if capacity available)
response = requests.post("http://localhost:8000/expert_reallocation/request",
    json={"layer_id": 8, "expert_id": 3, "action": "move_to_gpu"})

# Check memory usage
stats = requests.get("http://localhost:8000/expert_reallocation/stats").json()
print(f"GPU: {stats['memory']['gpu_allocated_gb']}GB, CPU: {stats['memory']['cpu_expert_storage_gb']}GB")
```

**Note**: GPU capacity is limited by the `wg` parameter. Moves are non-blocking and won't interrupt inference.

## Run Test Bench

```bash
cd benchmarks/mtbench
python bench.py --port 30000 --max-new-tokens 32 --ubs 324 --n-ub 14
```

