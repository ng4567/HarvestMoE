"""Dynamic port allocation utilities for FastMoE."""

import socket
import random
import time
from typing import List, Optional


def find_free_port(start_port: int = 10000, end_port: int = 65535, exclude_ports: Optional[List[int]] = None) -> int:
    """
    Find a free port in the given range.
    
    Args:
        start_port: Starting port number to search from
        end_port: Ending port number to search to
        exclude_ports: List of ports to exclude from search
        
    Returns:
        A free port number
        
    Raises:
        RuntimeError: If no free port is found
    """
    exclude_ports = exclude_ports or []
    exclude_set = set(exclude_ports)
    
    # Try random ports first for better distribution
    for _ in range(100):
        port = random.randint(start_port, end_port)
        if port in exclude_set:
            continue
            
        if is_port_free(port):
            return port
    
    # If random search fails, do sequential search
    for port in range(start_port, end_port + 1):
        if port in exclude_set:
            continue
            
        if is_port_free(port):
            return port
    
    raise RuntimeError(f"No free port found in range {start_port}-{end_port}")


def is_port_free(port: int, host: str = '') -> bool:
    """
    Check if a port is free on the given host.
    
    Args:
        port: Port number to check
        host: Host to check on (default: '' for all interfaces)
        
    Returns:
        True if port is free, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.close()
            return True
        except (OSError, socket.error):
            return False


def allocate_ports_with_retry(num_ports: int, start_port: int = 10000, 
                             end_port: int = 65535, max_retries: int = 3) -> List[int]:
    """
    Allocate multiple free ports with retry logic.
    
    Args:
        num_ports: Number of ports to allocate
        start_port: Starting port number to search from
        end_port: Ending port number to search to
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of allocated port numbers
        
    Raises:
        RuntimeError: If unable to allocate required ports
    """
    for attempt in range(max_retries):
        try:
            allocated_ports = []
            for _ in range(num_ports):
                port = find_free_port(start_port, end_port, allocated_ports)
                allocated_ports.append(port)
            
            # Verify all ports are still free
            all_free = all(is_port_free(port) for port in allocated_ports)
            if all_free:
                return allocated_ports
            
            # If not all free, wait and retry
            time.sleep(0.1 * (attempt + 1))
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to allocate {num_ports} ports after {max_retries} attempts: {e}")
            time.sleep(0.1 * (attempt + 1))
    
    raise RuntimeError(f"Failed to allocate {num_ports} ports after {max_retries} attempts")


def find_free_nccl_port() -> int:
    """
    Find a free port specifically for NCCL communication.
    Uses a higher port range to avoid conflicts with common services.
    """
    return find_free_port(start_port=20000, end_port=30000)
