#!/bin/bash

# Script to run Qwen3-30B-A3B benchmarks in 2-GPU and 4-GPU modes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="Qwen/Qwen3-30B-A3B"
PORT=30000
CPU_MEM_BDW=76
AVG_PROMPT_LEN=77
GEN_LEN=32

echo "Starting Qwen3-30B-A3B benchmarks..."
echo "Project root: $PROJECT_ROOT"

# Function to start server
start_server() {
    local tp_size=$1
    echo "Starting server with TP size: $tp_size"
    
    python -m fastmoe.serve.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$PORT" \
        --cpu-mem-bdw "$CPU_MEM_BDW" \
        --avg-prompt-len "$AVG_PROMPT_LEN" \
        --gen-len "$GEN_LEN" \
        --tp-size "$tp_size" &
    
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    
    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            break
        fi
        echo "Waiting... ($i/30)"
        sleep 2
    done
    
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Error: Server failed to start within timeout"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
}

# Function to stop server
stop_server() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "Server stopped"
    fi
}

# Function to run benchmark
run_benchmark() {
    local tp_size=$1
    local result_file="qwen3_30b_a3b_${tp_size}gpu_results.csv"
    
    echo "Running benchmark for ${tp_size}-GPU mode..."
    
    # Start server
    start_server $tp_size
    
    # Run benchmark
    cd "$PROJECT_ROOT"
    ./benchmarks/mtbench/benchmark.sh localhost $PORT 10
    
    # Rename results file
    if [ -f "benchmarks/mtbench/bench_results.csv" ]; then
        mv "benchmarks/mtbench/bench_results.csv" "benchmarks/mtbench/$result_file"
        echo "Results saved to: benchmarks/mtbench/$result_file"
    else
        echo "Warning: Results file not found"
    fi
    
    # Stop server
    stop_server
    
    # Wait a bit before next test
    sleep 5
}

# Trap to ensure server is stopped on exit
trap stop_server EXIT

# Check if required commands are available
if ! command -v python >/dev/null 2>&1; then
    echo "Error: python command not found"
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl command not found"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
cd "$PROJECT_ROOT"
pip install -r requirements.txt

# Add project to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Starting benchmarks..."

# Run 2-GPU benchmark
echo "========================================="
echo "Running 2-GPU benchmark"
echo "========================================="
run_benchmark 2

# Run 4-GPU benchmark
echo "========================================="
echo "Running 4-GPU benchmark"
echo "========================================="
run_benchmark 4

echo "========================================="
echo "All benchmarks completed!"
echo "========================================="

# Display summary
echo "Results summary:"
for result_file in benchmarks/mtbench/qwen3_30b_a3b_*gpu_results.csv; do
    if [ -f "$result_file" ]; then
        echo "File: $result_file"
        echo "Lines: $(wc -l < "$result_file")"
        echo ""
    fi
done