#!/usr/bin/env python3
"""Test script for expert reallocation feature."""

import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000"

def check_server_ready():
    """Check if the server is ready."""
    try:
        response = requests.get(f"{BASE_URL}/expert_locations", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    while not check_server_ready():
        time.sleep(2)
        print(".", end="", flush=True)
    print("\nServer is ready!")
    
    # 1. Check initial state
    print("\n1. Checking initial expert allocation...")
    response = requests.get(f"{BASE_URL}/expert_locations")
    data = response.json()
    summary = data['summary']
    
    print(f"  GPU capacity: {summary['gpu_capacity']} experts")
    print(f"  GPU persistent: {summary['global_location_distribution']['gpu_persistent']} experts")
    print(f"  CPU memory: {summary['global_location_distribution']['cpu_memory']} experts")
    
    # 2. Find a CPU expert to move
    print("\n2. Finding CPU experts in layer 8...")
    layer_8 = requests.get(f"{BASE_URL}/expert_locations/layer/8").json()
    cpu_experts = layer_8['location_distribution']['cpu_memory']
    print(f"  Layer 8 has {cpu_experts} experts on CPU")
    
    if summary['gpu_capacity'] == 0:
        print("\nERROR: GPU capacity is 0. The wg override didn't work!")
        print("Make sure the execution_engine.py hack is uncommented.")
        sys.exit(1)
    
    # 3. Try to move expert 1 from layer 8 to GPU
    print("\n3. Moving expert 1 from layer 8 to GPU...")
    move_request = {
        "layer_id": 8,
        "expert_id": 1,
        "action": "move_to_gpu",
        "priority": 1
    }
    
    response = requests.post(f"{BASE_URL}/expert_reallocation/request", json=move_request)
    result = response.json()
    request_id = result.get('request_id')
    
    print(f"  Request ID: {request_id}")
    print(f"  Initial status: {result.get('status')}")
    
    # 4. Wait for completion
    print("\n4. Checking reallocation status...")
    time.sleep(1)  # Give it time to process
    
    status_response = requests.get(f"{BASE_URL}/expert_reallocation/status/{request_id}")
    status = status_response.json()
    
    print(f"  Final status: {status.get('status')}")
    if status.get('message'):
        print(f"  Message: {status.get('message')}")
    
    # 5. Verify the move
    print("\n5. Verifying expert location after move...")
    updated_locations = requests.get(f"{BASE_URL}/expert_locations").json()
    updated_summary = updated_locations['summary']
    
    print(f"  GPU persistent: {updated_summary['global_location_distribution']['gpu_persistent']} experts")
    print(f"  CPU memory: {updated_summary['global_location_distribution']['cpu_memory']} experts")
    
    # Check specific expert
    all_experts = updated_locations['all_experts']
    target_expert = next((e for e in all_experts if e['layer_id'] == 8 and e['expert_id'] == 1), None)
    
    if target_expert:
        print(f"\n  Expert L8E1 location: {target_expert['location']}")
        print(f"  Expert L8E1 status: {target_expert['status']}")
        print(f"  Expert L8E1 device: {target_expert['device_id']}")
        print(f"  Expert L8E1 cache slot: {target_expert['cache_slot']}")
    
    # 6. Check memory stats
    print("\n6. Checking memory statistics...")
    stats = requests.get(f"{BASE_URL}/expert_reallocation/stats").json()
    if 'memory' in stats:
        mem = stats['memory']
        print(f"  GPU allocated: {mem['gpu_allocated_gb']} GB")
        print(f"  GPU free: {mem['gpu_free_gb']} GB")
        print(f"  Pending transfers: {mem['pending_transfers']}")

if __name__ == "__main__":
    main()
