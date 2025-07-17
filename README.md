# MoE Lightning Fork

This repository contains artifacts for benchmarking Mixture of Experts (MoE) models in multi-GPU environments.

## Features

- FastMoE inference server for MoE models
- Support for Qwen3MoE models
- Multi-GPU benchmarking capabilities
- Automated benchmark collection and analysis

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Running Qwen3-30B-A3B Benchmarks

To run comprehensive benchmarks for the Qwen3-30B-A3B model in both 2-GPU and 4-GPU modes:

```bash
./run_qwen3_benchmarks.sh
```

### Manual Server Launch

To manually start the MoE inference server:

```bash
python -m fastmoe.serve.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --port 30000 \
    --cpu-mem-bdw 76 \
    --avg-prompt-len 77 \
    --gen-len 32 \
    --tp-size 2
```

### Manual Benchmarking

To run benchmarks against a running server:

```bash
./benchmarks/mtbench/benchmark.sh localhost 30000 10
```

## Model Support

Currently supported models:
- Qwen/Qwen3-30B-A3B (via Qwen3MoE implementation)

## Benchmark Results

Benchmark results are saved as CSV files with the following format:
- `qwen3_30b_a3b_2gpu_results.csv` - Results for 2-GPU mode
- `qwen3_30b_a3b_4gpu_results.csv` - Results for 4-GPU mode

Each CSV contains:
- iteration: Benchmark iteration number
- prompt_tokens: Number of tokens in the prompt
- completion_tokens: Number of tokens in the completion
- total_tokens: Total tokens processed
- latency_ms: Request latency in milliseconds
- throughput_tokens_per_sec: Throughput in tokens per second
