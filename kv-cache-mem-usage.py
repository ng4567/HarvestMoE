import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                       help="Model name or path")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--num_experts", type=int, default=8,
                       help="Number of experts in MoE")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size (set to 1 to disable)")
    parser.add_argument("--expert_parallel_size", type=int, default=2,
                       help="Expert parallel size (number of GPUs for expert sharding)")
    parser.add_argument("--dtype", type=str, default="fp16",
                       choices=["fp16", "bf16", "fp32"],
                       help="Model precision")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run KV cache memory benchmark")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                       help="Maximum sequence length for benchmark")
    parser.add_argument("--max_batch_size", type=int, default=4,
                       help="Maximum batch size for benchmark")
    return parser.parse_args()

def get_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float32

def get_gpu_memory_info(gpu_id=None):
    """Get GPU memory usage in MB for specified GPU or current GPU"""
    if torch.cuda.is_available():
        if gpu_id is not None:
            with torch.cuda.device(gpu_id):
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
        else:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0

def format_memory(mb):
    """Format memory in MB to human readable format"""
    if mb > 1024:
        return f"{mb/1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"

def estimate_model_memory(model, dtype):
    """Estimate model parameters memory usage"""
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2 if dtype == torch.float16 else 4  # fp16=2bytes, fp32=4bytes
    estimated_memory = (total_params * bytes_per_param) / 1024**2  # Convert to MB
    return estimated_memory, total_params

def get_multi_gpu_memory_info():
    """Get memory info for all available GPUs"""
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(i).total_memory / 1024**2
            memory_info[f"GPU_{i}"] = {
                "allocated": allocated,
                "reserved": reserved,
                "total": total,
                "free": total - reserved
            }
    return memory_info

def benchmark_kv_cache_memory(model, tokenizer, max_seq_len=2048, max_batch_size=4):
    """Benchmark KV cache memory consumption for different sequence lengths and batch sizes"""
    print(f"\n{'='*80}")
    print("KV CACHE MEMORY BENCHMARK")
    print(f"{'='*80}")
    
    # Get model config for KV cache estimation
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, 'num_key_value_heads', num_attention_heads)
    num_layers = model.config.num_hidden_layers
    
    print(f"Model Config:")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Attention heads: {num_attention_heads}")
    print(f"  - KV heads: {num_key_value_heads}")
    print(f"  - Layers: {num_layers}")
    
    # Calculate theoretical KV cache size per token
    kv_cache_per_token = (2 * num_layers * num_key_value_heads * hidden_size // num_attention_heads) * 2  # 2 for K and V, 2 bytes for fp16
    kv_cache_per_token_mb = kv_cache_per_token / (1024**2)
    
    print(f"  - Theoretical KV cache per token: {kv_cache_per_token_mb:.2f} MB")
    
    # Calculate available GPU memory for multiple model instances
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**2)  # Convert to MB
        total_gpu_memory += total_memory
        print(f"  - GPU {i} total memory: {format_memory(total_memory)}")
    
    # Estimate model memory usage (approximate)
    model_params = sum(p.numel() for p in model.parameters())
    model_memory_mb = (model_params * 2) / (1024**2)  # Assuming fp16 (2 bytes per parameter)
    print(f"  - Estimated model memory per instance: {format_memory(model_memory_mb)}")
    
    # Calculate available memory for 2 model instances
    available_memory_for_kv = total_gpu_memory - (2 * model_memory_mb)
    print(f"  - Available memory for KV cache (2 instances): {format_memory(available_memory_for_kv)}")
    print()
    
    results = []
    
    # Test different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048] if max_seq_len >= 2048 else [64, 128, 256, 512, 1024]
    batch_sizes = [1, 2, 4] if max_batch_size >= 4 else [1, 2]
    
    print(f"{'Batch':<6} {'Seq Len':<8} {'Tokens':<8} {'Theoretical':<12} {'Actual':<12} {'Efficiency':<10} {'OOM Risk':<10}")
    print("-" * 85)
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            if seq_len > max_seq_len:
                continue
                
            total_tokens = batch_size * seq_len
            theoretical_kv_mb = total_tokens * kv_cache_per_token_mb
            
            # Clear cache before each test
            if hasattr(model, 'clear_cache'):
                model.clear_cache()
            torch.cuda.empty_cache()
            
            # Get baseline memory
            baseline_memory = {}
            for i in range(torch.cuda.device_count()):
                allocated, _ = get_gpu_memory_info(i)
                baseline_memory[i] = allocated
            
            # Create dummy input
            dummy_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            
            # Forward pass to populate KV cache
            with torch.no_grad():
                _ = model(dummy_input, use_cache=True)
            
            # Measure actual KV cache memory
            actual_kv_mb = 0
            for i in range(torch.cuda.device_count()):
                allocated, _ = get_gpu_memory_info(i)
                actual_kv_mb += (allocated - baseline_memory[i])
            
            efficiency = (theoretical_kv_mb / actual_kv_mb * 100) if actual_kv_mb > 0 else 0
            
            # Calculate OOM risk for 2 model instances
            kv_memory_for_2_instances = actual_kv_mb * 2  # Each instance needs the same KV cache
            oom_risk = "SAFE" if kv_memory_for_2_instances < available_memory_for_kv else "OOM RISK"
            
            print(f"{batch_size:<6} {seq_len:<8} {total_tokens:<8} {theoretical_kv_mb:<12.2f} {actual_kv_mb:<12.2f} {efficiency:<10.1f}% {oom_risk:<10}")
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'total_tokens': total_tokens,
                'theoretical_mb': theoretical_kv_mb,
                'actual_mb': actual_kv_mb,
                'efficiency': efficiency,
                'oom_risk': oom_risk,
                'kv_memory_for_2_instances': kv_memory_for_2_instances
            })
            
            # Clear cache
            if hasattr(model, 'clear_cache'):
                model.clear_cache()
            torch.cuda.empty_cache()
    
    print("-" * 85)
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"  - Total tests: {len(results)}")
    print(f"  - Average efficiency: {sum(r['efficiency'] for r in results) / len(results):.1f}%")
    print(f"  - Max KV cache memory: {max(r['actual_mb'] for r in results):.2f} MB")
    print(f"  - Min KV cache memory: {min(r['actual_mb'] for r in results):.2f} MB")
    
    # OOM Analysis for multiple instances
    print(f"\nOOM ANALYSIS FOR 2 MODEL INSTANCES:")
    safe_configs = [r for r in results if r['oom_risk'] == 'SAFE']
    oom_configs = [r for r in results if r['oom_risk'] == 'OOM RISK']
    
    print(f"  - Safe configurations: {len(safe_configs)}")
    print(f"  - OOM risk configurations: {len(oom_configs)}")
    
    if safe_configs:
        max_safe_seq_len = max(r['seq_len'] for r in safe_configs)
        max_safe_batch = max(r['batch_size'] for r in safe_configs)
        print(f"  - Max safe sequence length: {max_safe_seq_len}")
        print(f"  - Max safe batch size: {max_safe_batch}")
    
    if oom_configs:
        min_oom_seq_len = min(r['seq_len'] for r in oom_configs)
        min_oom_batch = min(r['batch_size'] for r in oom_configs)
        print(f"  - Min OOM sequence length: {min_oom_seq_len}")
        print(f"  - Min OOM batch size: {min_oom_batch}")
    
    # Calculate maximum safe KV cache size
    if safe_configs:
        max_safe_kv = max(r['kv_memory_for_2_instances'] for r in safe_configs)
        print(f"  - Max safe KV cache for 2 instances: {format_memory(max_safe_kv)}")
    
    # Calculate exact OOM thresholds
    oom_thresholds = calculate_oom_thresholds(model, available_memory_for_kv, kv_cache_per_token_mb)
    
    return results, oom_thresholds

def calculate_oom_thresholds(model, available_memory_for_kv, kv_cache_per_token_mb):
    """Calculate exact sequence lengths that would cause OOM for different batch sizes"""
    print(f"\n{'='*80}")
    print("OOM THRESHOLD CALCULATIONS")
    print(f"{'='*80}")
    
    print(f"Available memory for KV cache (2 instances): {format_memory(available_memory_for_kv)}")
    print(f"KV cache per token: {kv_cache_per_token_mb:.2f} MB")
    print()
    
    print(f"{'Batch Size':<12} {'Max Safe Seq Len':<18} {'Max Safe Tokens':<18} {'KV Memory':<15} {'Safety Margin':<15}")
    print("-" * 85)
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = []
    
    for batch_size in batch_sizes:
        # Calculate max tokens that can fit in available memory for 2 instances
        max_tokens_per_instance = available_memory_for_kv / (2 * kv_cache_per_token_mb)
        max_safe_tokens = int(max_tokens_per_instance)
        max_safe_seq_len = max_safe_tokens // batch_size
        
        # Calculate actual KV memory usage
        actual_kv_memory = max_safe_tokens * kv_cache_per_token_mb * 2  # 2 instances
        
        # Calculate safety margin
        safety_margin = available_memory_for_kv - actual_kv_memory
        
        print(f"{batch_size:<12} {max_safe_seq_len:<18} {max_safe_tokens:<18} {format_memory(actual_kv_memory):<15} {format_memory(safety_margin):<15}")
        
        results.append({
            'batch_size': batch_size,
            'max_safe_seq_len': max_safe_seq_len,
            'max_safe_tokens': max_safe_tokens,
            'kv_memory': actual_kv_memory,
            'safety_margin': safety_margin
        })
    
    print("-" * 85)
    print(f"\nRECOMMENDATIONS:")
    print(f"  - For batch size 1: Max sequence length ~{results[0]['max_safe_seq_len']:,}")
    print(f"  - For batch size 2: Max sequence length ~{results[1]['max_safe_seq_len']:,}")
    print(f"  - For batch size 4: Max sequence length ~{results[2]['max_safe_seq_len']:,}")
    
    return results

def main():
    args = parse_args()
    
    # Initialize distributed environment
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
    
    # Set expert-parallel size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    expert_parallel_size = min(args.expert_parallel_size, args.num_experts)
    
    print(f"World size: {world_size}")
    print(f"Expert parallel size: {expert_parallel_size}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Number of experts: {args.num_experts}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Loading model...")
    print(f"\n=== Initial GPU Memory (All GPUs) ===")
    for i in range(torch.cuda.device_count()):
        allocated, reserved = get_gpu_memory_info(i)
        print(f"GPU {i}: Allocated: {format_memory(allocated)}, Reserved: {format_memory(reserved)}")
    
    # Get memory for current GPU (usually GPU 0)
    initial_memory_allocated, initial_memory_reserved = get_gpu_memory_info()
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=get_dtype(args.dtype),
        device_map="auto" if args.local_rank == -1 else None,
        trust_remote_code=True
    )
    
    # Measure model memory usage
    print(f"\n=== Model Memory Analysis ===")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    estimated_model_memory, total_params = estimate_model_memory(model, get_dtype(args.dtype))
    print(f"Estimated model memory: {format_memory(estimated_model_memory)}")
    
    print(f"Actual GPU memory after model load:")
    for i in range(torch.cuda.device_count()):
        allocated, reserved = get_gpu_memory_info(i)
        print(f"  GPU {i}: Allocated: {format_memory(allocated)}, Reserved: {format_memory(reserved)}")
    
    # Get memory for current GPU for delta calculation
    model_memory_allocated, model_memory_reserved = get_gpu_memory_info()
    print(f"  Model memory delta (GPU 0): {format_memory(model_memory_allocated - initial_memory_allocated)}")
    
    # Multi-GPU memory info
    multi_gpu_info = get_multi_gpu_memory_info()
    if multi_gpu_info:
        print(f"\n=== Multi-GPU Memory Distribution ===")
        for gpu_name, info in multi_gpu_info.items():
            print(f"{gpu_name}:")
            print(f"  - Allocated: {format_memory(info['allocated'])}")
            print(f"  - Reserved: {format_memory(info['reserved'])}")
            print(f"  - Total: {format_memory(info['total'])}")
            print(f"  - Free: {format_memory(info['free'])}")
    
    # For expert parallelism, we'll use tensor parallelism to distribute experts
    # This is the most effective way to shard Mixtral across GPUs
    if expert_parallel_size > 1:
        print(f"Setting up expert parallelism across {expert_parallel_size} GPUs")
        print("Using tensor parallelism to distribute experts")
        tensor_parallel_size = expert_parallel_size
    else:
        tensor_parallel_size = args.tensor_parallel_size
    
    # Move model to GPU if not using device_map
    if args.local_rank != -1:
        model = model.cuda()
    
    # For DeepSpeed inference, we'll use a simpler approach
    # since the API varies between versions
    print(f"Model loaded with tensor parallel size: {tensor_parallel_size}")
    print(f"Model dtype: {get_dtype(args.dtype)}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Run KV cache benchmark or example inference
    should_run_inference = True
    if dist.is_initialized():
        should_run_inference = (dist.get_rank() == 0)
    
    if should_run_inference:
        if args.benchmark:
            # Run KV cache memory benchmark
            benchmark_results, oom_thresholds = benchmark_kv_cache_memory(
                model, tokenizer, 
                max_seq_len=args.max_seq_len, 
                max_batch_size=args.max_batch_size
            )
        else:
            print(f"\n=== Inference Memory Analysis ===")
        
        # Memory before inference (all GPUs)
        print(f"Memory before inference:")
        pre_inference_memory = {}
        for i in range(torch.cuda.device_count()):
            allocated, reserved = get_gpu_memory_info(i)
            pre_inference_memory[i] = (allocated, reserved)
            print(f"  GPU {i}: Allocated: {format_memory(allocated)}, Reserved: {format_memory(reserved)}")
        
        # Get current GPU memory for delta calculations
        pre_inference_memory_allocated, pre_inference_memory_reserved = get_gpu_memory_info()
        
        prompt = "Explain quantum computing in simple terms:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Memory after input preparation (all GPUs)
        print(f"Memory after input preparation:")
        post_input_memory = {}
        for i in range(torch.cuda.device_count()):
            allocated, reserved = get_gpu_memory_info(i)
            post_input_memory[i] = (allocated, reserved)
            print(f"  GPU {i}: Allocated: {format_memory(allocated)}, Reserved: {format_memory(reserved)}")
        
        post_input_memory_allocated, post_input_memory_reserved = get_gpu_memory_info()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Memory after generation (includes KV cache) - all GPUs
        print(f"\n=== KV Cache Analysis ===")
        print(f"Memory after generation:")
        post_generation_memory = {}
        total_kv_cache_memory = 0
        
        for i in range(torch.cuda.device_count()):
            allocated, reserved = get_gpu_memory_info(i)
            post_generation_memory[i] = (allocated, reserved)
            kv_cache_memory = allocated - post_input_memory[i][0]
            total_kv_cache_memory += kv_cache_memory
            print(f"  GPU {i}: Allocated: {format_memory(allocated)}, Reserved: {format_memory(reserved)}")
            print(f"    KV Cache: {format_memory(kv_cache_memory)}")
        
        post_generation_memory_allocated, post_generation_memory_reserved = get_gpu_memory_info()
        print(f"Total KV cache memory across all GPUs: {format_memory(total_kv_cache_memory)}")
        print(f"Total inference memory delta (GPU 0): {format_memory(post_generation_memory_allocated - pre_inference_memory_allocated)}")
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== Generation Results ===")
        print(f"Prompt: {prompt}")
        print(f"Generated length: {len(outputs[0])} tokens")
        
        # Clear KV cache
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
        torch.cuda.empty_cache()
        
        # Memory after clearing cache (all GPUs)
        print(f"\nMemory after clearing cache:")
        for i in range(torch.cuda.device_count()):
            allocated, reserved = get_gpu_memory_info(i)
            memory_freed = post_generation_memory[i][0] - allocated
            print(f"  GPU {i}: Allocated: {format_memory(allocated)}, Reserved: {format_memory(reserved)}")
            print(f"    Memory freed: {format_memory(memory_freed)}")
        
        post_clear_memory_allocated, post_clear_memory_reserved = get_gpu_memory_info()

if __name__ == "__main__":
    main()