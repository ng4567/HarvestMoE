import psutil
import random
import socket
import sys
import time
import traceback
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from fastmoe.utils.port_utils import allocate_ports_with_retry, is_port_free

def get_available_gpu_memory(gpu_id, distributed=True):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    assert gpu_id < num_gpus

    if torch.cuda.current_device() != gpu_id:
        print(
            f"WARN: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
            "which may cause useless memory allocation for torch CUDA context.",
        )

    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)

def get_available_cpu_memory():
    return psutil.virtual_memory().available / (1 << 30)


def set_random_seed(seed: int) -> None:
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alloc_usable_network_port(num, used_list=()):
    port_list = []
    for port in range(10000, 65536):
        if port in used_list:
            continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                port_list.append(port)
            except socket.error:
                pass

            if len(port_list) == num:
                return port_list
    return None


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except socket.error:
            return False


def handle_port_init(
    port: Optional[int] = None,
    additional_ports: Optional[List[int]] = None,
    tp_size: int = 1,
):
    port = 30000 if port is None else port
    additional_ports = [] if additional_ports is None else additional_ports
    additional_ports = (
        [additional_ports] if isinstance(additional_ports, int) else additional_ports
    )
    
    # Check and allocate main server port
    if not is_port_free(port):
        try:
            # Try to find a free port starting from the requested port
            for offset in range(100):
                test_port = port + offset
                if is_port_free(test_port):
                    print(f"Port {port} is not available, using {test_port} instead.")
                    port = test_port
                    break
            else:
                raise RuntimeError(f"Could not find free port near {port}")
        except Exception as e:
            print(f"Error finding free port: {e}")
            port = random.randint(30000, 40000)

    # Allocate additional ports
    num_required_ports = 4 + tp_size
    
    # Filter existing additional_ports for those that are actually free
    valid_ports = [p for p in additional_ports if p != port and is_port_free(p)]
    
    if len(valid_ports) < num_required_ports:
        # Need more ports - allocate them dynamically
        num_needed = num_required_ports - len(valid_ports)
        excluded_ports = [port] + valid_ports
        
        try:
            new_ports = allocate_ports_with_retry(
                num_ports=num_needed,
                start_port=10000,
                end_port=50000,
                max_retries=3
            )
            # Ensure new ports don't overlap with excluded ones
            new_ports = [p for p in new_ports if p not in excluded_ports]
            valid_ports.extend(new_ports[:num_needed])
        except Exception as e:
            print(f"Warning: Could not allocate all required ports: {e}")
            # Fallback to sequential allocation
            for start in [10000, 20000, 30000, 40000]:
                for p in range(start, start + 10000):
                    if p not in excluded_ports and is_port_free(p):
                        valid_ports.append(p)
                        excluded_ports.append(p)
                        if len(valid_ports) >= num_required_ports:
                            break
                if len(valid_ports) >= num_required_ports:
                    break
    
    additional_ports = valid_ports[:num_required_ports]
    print(f"Allocated ports - Main: {port}, Additional: {additional_ports}")
    return port, additional_ports


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def get_int_token_logit_bias(tokenizer, vocab_size):
    from transformers import LlamaTokenizer, LlamaTokenizerFast

    # a bug when model's vocab size > tokenizer.vocab_size
    vocab_size = tokenizer.vocab_size
    logit_bias = np.zeros(vocab_size, dtype=np.float32)
    for t_id in range(vocab_size):
        ss = tokenizer.decode([t_id]).strip()
        if not (ss.isdigit() or len(ss) == 0 or t_id == tokenizer.eos_token_id):
            logit_bias[t_id] = -1e5
        # else:
        #    print(ss, t_id)

    return logit_bias


def wrap_kernel_launcher(kernel):
    """A faster launcher for triton kernels."""
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    kernels = kernel.cache[rank].values()
    kernel = next(iter(kernels))

    # Different trition versions use different low-level names
    if hasattr(kernel, "cu_function"):
        kfunction = kernel.cu_function
    else:
        kfunction = kernel.function

    if hasattr(kernel, "c_wrapper"):
        run = kernel.c_wrapper
    else:
        run = kernel.run

    add_cluster_dim = True

    def ret_func(grid, num_warps, *args):
        nonlocal add_cluster_dim

        try:
            if add_cluster_dim:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    1,
                    1,
                    1,
                    1,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
            else:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
        except TypeError:
            add_cluster_dim = not add_cluster_dim
            ret_func(grid, num_warps, *args)

    return ret_func