#!/usr/bin/env python3
"""Debug script to understand expert allocation issues."""

import requests
import json

BASE_URL = "http://localhost:8000"

def main():
    # 1. Get full expert locations
    print("=== FULL EXPERT LOCATION DATA ===")
    response = requests.get(f"{BASE_URL}/expert_locations")
    data = response.json()
    
    print(f"\nGPU Capacity: {data['summary']['gpu_capacity']}")
    print(f"Cache Capacity: {data['summary']['cache_capacity']}")
    print(f"GPU Persistent count: {data['summary']['global_location_distribution']['gpu_persistent']}")
    
    # 2. Check layer 8 specifically
    print("\n=== LAYER 8 EXPERT DETAILS ===")
    layer_8_experts = [e for e in data['all_experts'] if e['layer_id'] == 8]
    
    for expert in sorted(layer_8_experts, key=lambda x: x['expert_id']):
        print(f"Expert {expert['expert_id']}: location={expert['location']}, "
              f"status={expert['status']}, device={expert['device_id']}, "
              f"slot={expert['cache_slot']}")
    
    # 3. Find which experts are actually on CPU in layer 8
    cpu_experts = [e for e in layer_8_experts if e['location'] == 'cpu_memory']
    print(f"\nCPU experts in layer 8: {[e['expert_id'] for e in cpu_experts]}")
    
    # 4. Check reallocation stats
    print("\n=== REALLOCATION STATS ===")
    stats = requests.get(f"{BASE_URL}/expert_reallocation/stats").json()
    print(json.dumps(stats, indent=2))
    
    # 5. Try to understand GPU slot allocation pattern
    print("\n=== GPU SLOT ALLOCATION PATTERN ===")
    gpu_experts = [e for e in data['all_experts'] if e['location'] == 'gpu_persistent']
    
    # Group by layer
    by_layer = {}
    for e in gpu_experts:
        layer = e['layer_id']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append((e['expert_id'], e['cache_slot']))
    
    for layer in sorted(by_layer.keys())[:5]:  # Show first 5 layers
        experts = sorted(by_layer[layer])
        print(f"Layer {layer}: experts {[e[0] for e in experts]} at slots {[e[1] for e in experts]}")
    
    # 6. Calculate expected vs actual
    print("\n=== EXPECTED VS ACTUAL ===")
    num_layers = 32
    experts_per_layer = 8
    wg = 0.25
    expected_gpu_per_layer = int(experts_per_layer * wg)  # Should be 2
    expected_total_gpu = num_layers * expected_gpu_per_layer  # Should be 64
    
    print(f"Expected GPU experts per layer: {expected_gpu_per_layer}")
    print(f"Expected total GPU experts: {expected_total_gpu}")
    print(f"Actual total GPU experts: {len(gpu_experts)}")
    print(f"GPU capacity field value: {data['summary']['gpu_capacity']}")
    
    # 7. Try to move a CPU expert
    if cpu_experts:
        print(f"\n=== ATTEMPTING TO MOVE CPU EXPERT ===")
        target = cpu_experts[0]
        print(f"Moving expert {target['expert_id']} from layer 8...")
        
        move_request = {
            "layer_id": 8,
            "expert_id": target['expert_id'],
            "action": "move_to_gpu",
            "priority": 1
        }
        
        response = requests.post(f"{BASE_URL}/expert_reallocation/request", json=move_request)
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()
