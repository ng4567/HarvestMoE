#!/bin/bash

echo "Stopping existing FastMoE processes..."

# Kill all existing fastmoe processes
ps aux | grep python | grep -E "(fastmoe|mixtral)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Check if processes are killed
sleep 2

echo "Checking if all processes are stopped..."
if ps aux | grep python | grep -E "(fastmoe|mixtral)" | grep -v grep; then
    echo "Warning: Some processes may still be running"
else
    echo "All FastMoE processes stopped successfully"
fi

echo "Starting server with GPU expert allocation enabled (wg=0.25)..."
echo "This will allocate 2 experts per layer to GPU (64 total)"

cd /home/azureuser/main

# Start the server with unbuffered output
source /home/azureuser/anaconda3/bin/activate fastmoe
stdbuf -o0 -e0 python -m fastmoe.serve.launch_server \
    --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --port 8000 \
    --cpu-mem-bdw 76 \
    --avg-prompt-len 77 \
    --gen-len 32 2>&1 | tee server_gpu_experts.log &

echo "Server starting... Check server_gpu_experts.log for progress"
echo ""
echo "Once the server is ready, you can verify GPU experts with:"
echo "  curl http://localhost:8000/expert_locations | jq '.summary.gpu_capacity'"
echo ""
echo "Then test moving experts:"
echo "  curl -X POST http://localhost:8000/expert_reallocation/request \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{"
echo "      \"layer_id\": 8,"
echo "      \"expert_id\": 1,"
echo "      \"action\": \"move_to_gpu\","
echo "      \"priority\": 1"
echo "    }'"
