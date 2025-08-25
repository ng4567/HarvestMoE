#!/usr/bin/env python3
"""Test expert reallocation by swapping experts between GPU and CPU."""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def wait_for_request(request_id, timeout=5):
    """Wait for a reallocation request to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(f"{BASE_URL}/expert_reallocation/status/{request_id}")
        status = response.json()
        if status['status'] in ['completed', 'failed']:
            return status
        time.sleep(0.1)
    return status

def main():
    print("=== EXPERT REALLOCATION SWAP TEST ===\n")
    
    # 1. Check initial state
    print("1. Checking initial state...")
    response = requests.get(f"{BASE_URL}/expert_locations")
    data = response.json()
    summary = data['summary']
    
    print(f"   GPU capacity: {summary['gpu_capacity']}")
    print(f"   GPU persistent: {summary['global_location_distribution']['gpu_persistent']}")
    print(f"   Available slots: {summary['gpu_capacity'] - summary['global_location_distribution']['gpu_persistent']}")
    
    # 2. Check layer 8 specifically
    print("\n2. Layer 8 expert locations:")
    layer_8 = requests.get(f"{BASE_URL}/expert_locations/layer/8").json()
    layer_8_experts = [e for e in data['all_experts'] if e['layer_id'] == 8]
    
    for expert in sorted(layer_8_experts, key=lambda x: x['expert_id']):
        print(f"   Expert {expert['expert_id']}: {expert['location']}")
    
    # 3. Move expert 0 from layer 8 to CPU (free up a slot)
    print("\n3. Moving expert 0 from layer 8 to CPU...")
    move_request = {
        "layer_id": 8,
        "expert_id": 0,
        "action": "move_to_cpu",
        "priority": 1
    }
    
    response = requests.post(f"{BASE_URL}/expert_reallocation/request", json=move_request)
    result = response.json()
    request_id = result.get('request_id')
    print(f"   Request ID: {request_id}")
    print(f"   Initial status: {result.get('status')}")
    
    # Wait for completion
    final_status = wait_for_request(request_id)
    print(f"   Final status: {final_status.get('status')}")
    if final_status.get('message'):
        print(f"   Message: {final_status.get('message')}")
    
    # 4. Verify expert 0 moved to CPU
    time.sleep(0.5)
    print("\n4. Verifying expert 0 moved to CPU...")
    updated_data = requests.get(f"{BASE_URL}/expert_locations").json()
    expert_0 = next((e for e in updated_data['all_experts'] if e['layer_id'] == 8 and e['expert_id'] == 0), None)
    
    if expert_0:
        print(f"   Expert 0 location: {expert_0['location']}")
    
    summary = updated_data['summary']
    print(f"   GPU persistent: {summary['global_location_distribution']['gpu_persistent']}")
    print(f"   Available slots: {summary['gpu_capacity'] - summary['global_location_distribution']['gpu_persistent']}")
    
    # 5. Now move expert 2 from layer 8 to GPU
    print("\n5. Moving expert 2 from layer 8 to GPU...")
    move_request = {
        "layer_id": 8,
        "expert_id": 2,
        "action": "move_to_gpu",
        "priority": 1
    }
    
    response = requests.post(f"{BASE_URL}/expert_reallocation/request", json=move_request)
    result = response.json()
    request_id = result.get('request_id')
    print(f"   Request ID: {request_id}")
    print(f"   Initial status: {result.get('status')}")
    
    # Wait for completion
    final_status = wait_for_request(request_id)
    print(f"   Final status: {final_status.get('status')}")
    if final_status.get('message'):
        print(f"   Message: {final_status.get('message')}")
    
    # 6. Final verification
    time.sleep(0.5)
    print("\n6. Final expert locations in layer 8:")
    final_data = requests.get(f"{BASE_URL}/expert_locations").json()
    layer_8_experts = [e for e in final_data['all_experts'] if e['layer_id'] == 8]
    
    for expert in sorted(layer_8_experts, key=lambda x: x['expert_id']):
        loc_str = f"{expert['location']}"
        if expert['location'] == 'gpu_persistent':
            loc_str += f" (slot {expert['cache_slot']})"
        print(f"   Expert {expert['expert_id']}: {loc_str}")
    
    # 7. Memory stats
    print("\n7. Memory statistics:")
    stats = requests.get(f"{BASE_URL}/expert_reallocation/stats").json()
    if 'memory' in stats:
        mem = stats['memory']
        print(f"   GPU allocated: {mem['gpu_allocated_gb']} GB")
        print(f"   CPU expert storage: {mem['cpu_expert_storage_gb']} GB")

if __name__ == "__main__":
    main()
