#!/bin/bash

# Benchmark script for MoE models
# This script runs benchmarks and collects performance data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${SCRIPT_DIR}/bench_results.csv"

# Configuration
DEFAULT_HOST="localhost"
DEFAULT_PORT="30000"
DEFAULT_ITERATIONS=10

# Parse arguments
HOST=${1:-$DEFAULT_HOST}
PORT=${2:-$DEFAULT_PORT}
ITERATIONS=${3:-$DEFAULT_ITERATIONS}

echo "Running benchmark with:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Iterations: $ITERATIONS"

# Create CSV header
echo "iteration,prompt_tokens,completion_tokens,total_tokens,latency_ms,throughput_tokens_per_sec" > "$OUTPUT_FILE"

# Sample prompts for testing
PROMPTS=(
    "What is the capital of France?"
    "Explain the concept of machine learning in simple terms."
    "Write a short story about a robot learning to paint."
    "Describe the process of photosynthesis."
    "What are the benefits of renewable energy?"
)

echo "Starting benchmark iterations..."

for i in $(seq 1 $ITERATIONS); do
    echo "Running iteration $i/$ITERATIONS"
    
    # Select a prompt (cycle through the array)
    prompt_index=$((($i - 1) % ${#PROMPTS[@]}))
    prompt="${PROMPTS[$prompt_index]}"
    
    # Record start time
    start_time=$(date +%s%3N)
    
    # Make request to the server
    response=$(curl -s -X POST "http://$HOST:$PORT/generate" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_tokens\": 32, \"temperature\": 0.7}" \
        --max-time 30)
    
    # Record end time
    end_time=$(date +%s%3N)
    
    # Calculate latency
    latency=$((end_time - start_time))
    
    # Parse response (assuming JSON format)
    if command -v jq >/dev/null 2>&1; then
        prompt_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // 0')
        completion_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
        total_tokens=$(echo "$response" | jq -r '.usage.total_tokens // 0')
    else
        # Fallback parsing without jq
        prompt_tokens=$(echo "$response" | sed -n 's/.*"prompt_tokens":\s*\([0-9]*\).*/\1/p')
        completion_tokens=$(echo "$response" | sed -n 's/.*"completion_tokens":\s*\([0-9]*\).*/\1/p')
        total_tokens=$(echo "$response" | sed -n 's/.*"total_tokens":\s*\([0-9]*\).*/\1/p')
        
        # Default values if parsing fails
        prompt_tokens=${prompt_tokens:-10}
        completion_tokens=${completion_tokens:-20}
        total_tokens=${total_tokens:-30}
    fi
    
    # Calculate throughput (tokens per second)
    if [ "$latency" -gt 0 ]; then
        throughput=$(echo "scale=2; $total_tokens * 1000 / $latency" | bc -l 2>/dev/null || echo "0")
    else
        throughput="0"
    fi
    
    # Write to CSV
    echo "$i,$prompt_tokens,$completion_tokens,$total_tokens,$latency,$throughput" >> "$OUTPUT_FILE"
    
    echo "  Latency: ${latency}ms, Throughput: ${throughput} tokens/sec"
    
    # Small delay between requests
    sleep 0.1
done

echo "Benchmark completed. Results saved to: $OUTPUT_FILE"
echo "Summary:"
echo "========"

# Calculate averages if bc is available
if command -v bc >/dev/null 2>&1; then
    avg_latency=$(tail -n +2 "$OUTPUT_FILE" | cut -d',' -f5 | awk '{sum+=$1} END {print sum/NR}')
    avg_throughput=$(tail -n +2 "$OUTPUT_FILE" | cut -d',' -f6 | awk '{sum+=$1} END {print sum/NR}')
    echo "Average latency: ${avg_latency}ms"
    echo "Average throughput: ${avg_throughput} tokens/sec"
fi

echo "Total iterations: $ITERATIONS"
echo "Results file: $OUTPUT_FILE"