# FastMoE Configuration

## Model Configurations

### Qwen3-30B-A3B
- **Model Path**: `Qwen/Qwen3-30B-A3B`
- **Architecture**: Mixture of Experts (MoE) 
- **Parameters**: ~30B total parameters
- **Expert Configuration**: Sparse activation pattern
- **Supported TP Sizes**: 1, 2, 4, 8

## Server Configuration

### Default Parameters
- **Port**: 30000
- **CPU Memory Bandwidth**: 76 GB/s
- **Average Prompt Length**: 77 tokens
- **Generation Length**: 32 tokens
- **Host**: 0.0.0.0 (all interfaces)

### Command Line Usage

#### 2-GPU Mode
```bash
python -m fastmoe.serve.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --port 30000 \
    --cpu-mem-bdw 76 \
    --avg-prompt-len 77 \
    --gen-len 32 \
    --tp-size 2
```

#### 4-GPU Mode  
```bash
python -m fastmoe.serve.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --port 30000 \
    --cpu-mem-bdw 76 \
    --avg-prompt-len 77 \
    --gen-len 32 \
    --tp-size 4
```

## Benchmark Configuration

### Parameters
- **Iterations**: 10 (configurable)
- **Request Timeout**: 30 seconds
- **Sample Prompts**: 5 predefined prompts cycling
- **Output Format**: CSV with performance metrics

### Expected Metrics
- **Latency**: Time from request to response
- **Throughput**: Tokens processed per second
- **Token Counts**: Prompt, completion, and total tokens
- **Success Rate**: Should be 100% for successful benchmarks

## Environment Requirements

### Hardware
- **GPUs**: 2 or 4 compatible GPUs for tensor parallelism
- **Memory**: Sufficient GPU memory for model weights + KV cache
- **CPU**: Multi-core for data preprocessing
- **Storage**: Fast storage for model loading

### Software
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **vLLM**: 0.3.0+
- **Transformers**: 4.30.0+
- **FastAPI**: For REST API server
- **CUDA**: Compatible with PyTorch version