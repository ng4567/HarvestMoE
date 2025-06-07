#!/usr/bin/env python3
"""
GPU Memory Bandwidth Measurement Script
Run this to determine actual GPU memory bandwidth for H100 configuration.
"""

import torch
import time
import numpy as np

def measure_gpu_bandwidth():
    device = torch.cuda.current_device()
    print(f"Measuring bandwidth for: {torch.cuda.get_device_name(device)}")
    
    # Test different sizes
    sizes_gb = [1, 2, 4, 8]  # Test with different memory sizes
    
    for size_gb in sizes_gb:
        # Create large tensors
        size_elements = int(size_gb * 1024**3 // 4)  # 4 bytes per float32
        
        try:
            # Allocate memory
            a = torch.randn(size_elements, dtype=torch.float32, device='cuda')
            b = torch.randn(size_elements, dtype=torch.float32, device='cuda')
            
            # Warm up
            for _ in range(10):
                c = a + b
            torch.cuda.synchronize()
            
            # Measure bandwidth
            num_trials = 50
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_trials):
                c = a + b  # This is memory bandwidth limited
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate bandwidth
            total_time = end_time - start_time
            bytes_transferred = size_gb * 1024**3 * 3 * num_trials  # Read a, read b, write c
            bandwidth_gbs = bytes_transferred / total_time / (1024**3)
            
            print(f"Size: {size_gb} GB, Bandwidth: {bandwidth_gbs:.2f} GB/s")
            
            del a, b, c
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory for {size_gb} GB test")
            break

def measure_compute_throughput():
    """Measure FLOPS performance"""
    device = torch.cuda.current_device()
    print(f"\nMeasuring compute throughput for: {torch.cuda.get_device_name(device)}")
    
    # Test matrix multiplication (good proxy for FLOPS)
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        try:
            a = torch.randn(size, size, dtype=torch.float16, device='cuda')
            b = torch.randn(size, size, dtype=torch.float16, device='cuda')
            
            # Warm up
            for _ in range(10):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            
            # Measure
            num_trials = 100
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_trials):
                c = torch.mm(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            # FLOPS = 2 * size^3 operations per matrix multiply
            flops_per_trial = 2 * size**3
            total_flops = flops_per_trial * num_trials
            tflops = total_flops / total_time / 1e12
            
            print(f"Matrix size: {size}x{size}, Performance: {tflops:.2f} TFLOPS")
            
            del a, b, c
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory for {size}x{size} matrix")
            break

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    print("=== GPU Hardware Measurement ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Basic GPU info
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"Memory: {props.total_memory / (1024**3):.1f} GB")
    print(f"SMs: {props.multi_processor_count}")
    
    measure_gpu_bandwidth()
    measure_compute_throughput() 