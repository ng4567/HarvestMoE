"""Dynamic expert reallocation system for FastMoE.

This module provides functionality to dynamically move experts between
different memory locations (GPU/CPU) based on runtime decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Any
import time
import threading
from queue import Queue, Empty
import torch
import logging

from fastmoe.backend.expert_tracker import MemoryLocation, ExpertStatus

logger = logging.getLogger(__name__)


class ReallocationAction(str, Enum):
    """Types of expert reallocation actions."""
    MOVE_TO_GPU = "move_to_gpu"        # Move expert from CPU to GPU persistent
    MOVE_TO_CPU = "move_to_cpu"        # Move expert from GPU to CPU
    PRELOAD_TO_CACHE = "preload_cache" # Preload expert into GPU cache
    EVICT_FROM_CACHE = "evict_cache"   # Evict expert from GPU cache


class ReallocationStatus(str, Enum):
    """Status of a reallocation request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExpertReallocationRequest:
    """Request to reallocate a specific expert."""
    request_id: str
    layer_id: int
    expert_id: int
    action: ReallocationAction
    priority: int = 0  # Higher priority processed first
    timestamp: float = field(default_factory=time.time)
    status: ReallocationStatus = ReallocationStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BatchReallocationRequest:
    """Request to reallocate multiple experts."""
    request_id: str
    requests: List[ExpertReallocationRequest]
    atomic: bool = True  # If True, all succeed or all fail
    timestamp: float = field(default_factory=time.time)
    status: ReallocationStatus = ReallocationStatus.PENDING
    completed_count: int = 0
    failed_count: int = 0


class ReallocationManager:
    """Manages expert reallocation requests and execution."""
    
    def __init__(self, max_concurrent_moves: int = 2):
        self.max_concurrent_moves = max_concurrent_moves
        self.request_queue = Queue()
        self.active_requests: Dict[str, ExpertReallocationRequest] = {}
        self.completed_requests: Dict[str, ExpertReallocationRequest] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # Track which experts are currently being moved
        self.experts_in_motion: Set[Tuple[int, int]] = set()  # (layer_id, expert_id)
        
    def submit_request(self, request: ExpertReallocationRequest) -> str:
        """Submit a reallocation request."""
        with self._lock:
            # Check if expert is already being moved
            expert_key = (request.layer_id, request.expert_id)
            if expert_key in self.experts_in_motion:
                request.status = ReallocationStatus.FAILED
                request.error_message = "Expert is already being reallocated"
                return request.request_id
            
            # Add to queue
            self.request_queue.put(request)
            self.active_requests[request.request_id] = request
            
        return request.request_id
    
    def submit_batch(self, batch: BatchReallocationRequest) -> str:
        """Submit a batch reallocation request."""
        with self._lock:
            # Check all experts in batch
            for req in batch.requests:
                expert_key = (req.layer_id, req.expert_id)
                if expert_key in self.experts_in_motion:
                    if batch.atomic:
                        batch.status = ReallocationStatus.FAILED
                        return batch.request_id
            
            # Add all requests to queue
            for req in batch.requests:
                self.request_queue.put(req)
                self.active_requests[req.request_id] = req
                
        return batch.request_id
    
    def get_request_status(self, request_id: str) -> Optional[ExpertReallocationRequest]:
        """Get the status of a reallocation request."""
        with self._lock:
            if request_id in self.active_requests:
                return self.active_requests[request_id]
            elif request_id in self.completed_requests:
                return self.completed_requests[request_id]
        return None
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request."""
        with self._lock:
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                if request.status == ReallocationStatus.PENDING:
                    request.status = ReallocationStatus.CANCELLED
                    return True
        return False
    
    def get_reallocation_stats(self) -> Dict:
        """Get statistics about reallocation requests."""
        with self._lock:
            pending_count = sum(1 for r in self.active_requests.values() 
                              if r.status == ReallocationStatus.PENDING)
            in_progress_count = sum(1 for r in self.active_requests.values() 
                                  if r.status == ReallocationStatus.IN_PROGRESS)
            completed_count = len(self.completed_requests)
            
            return {
                "pending_requests": pending_count,
                "in_progress_requests": in_progress_count,
                "completed_requests": completed_count,
                "experts_in_motion": len(self.experts_in_motion),
                "queue_size": self.request_queue.qsize()
            }


class ExpertMover:
    """Handles the actual movement of experts between memory locations."""
    
    def __init__(self, execution_engine, expert_tracker, device_id: int = 0):
        self.execution_engine = execution_engine
        self.expert_tracker = expert_tracker
        self.model_config = execution_engine.model_config
        self.context = execution_engine.context
        self.device_id = device_id
        self._lock = threading.Lock()
        
        # Set CUDA device
        torch.cuda.set_device(self.device_id)
        
        # Create CUDA streams for non-blocking operations
        self.copy_stream = torch.cuda.Stream(device=self.device_id)
        self.sync_events = {}  # Track copy completion events
        
        # Cache expert dimensions
        self.intermediate_size = self.model_config.intermediate_size // execution_engine.hardware_config.tp_size
        self.hidden_size = self.model_config.hidden_size
        self.expert_size_elements = 3 * self.intermediate_size * self.hidden_size
        self.dtype = torch.get_default_dtype()
        
    def can_move_to_gpu(self, layer_id: int, expert_id: int) -> Tuple[bool, Optional[str]]:
        """Check if an expert can be moved to GPU."""
        # Check current location
        expert_loc = self.expert_tracker.get_expert_location(layer_id, expert_id)
        if not expert_loc:
            return False, "Expert not found"
            
        if expert_loc.location == MemoryLocation.GPU_PERSISTENT:
            return False, "Expert already on GPU"
        
        # Check if there's available GPU capacity globally
        total_gpu_experts = sum(
            1 for layer in self.expert_tracker.layers.values()
            for e in layer.experts.values()
            if e.location == MemoryLocation.GPU_PERSISTENT
        )
        
        if total_gpu_experts >= self.expert_tracker.gpu_capacity:
            return False, "GPU expert capacity full"
        
        # Also check per-layer limit
        num_gpu_experts_per_layer = int(self.model_config.num_local_experts * self.context.policy.wg)
        current_layer_gpu_experts = sum(1 for e in self.expert_tracker.layers[layer_id].experts.values()
                                      if e.location == MemoryLocation.GPU_PERSISTENT)
        
        if current_layer_gpu_experts >= num_gpu_experts_per_layer:
            return False, f"Layer {layer_id} already has maximum {num_gpu_experts_per_layer} experts on GPU"
            
        return True, None
    
    def move_expert_to_gpu(self, layer_id: int, expert_id: int, target_device_id: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Move an expert from CPU to GPU persistent storage."""
        if target_device_id is None:
            target_device_id = self.device_id
            
        with self._lock:
            can_move, error = self.can_move_to_gpu(layer_id, expert_id)
            if not can_move:
                return False, error
            
            try:
                # Get the expert's current location
                expert_loc = self.expert_tracker.get_expert_location(layer_id, expert_id)
                if not expert_loc:
                    return False, "Expert not found in tracker"
                
                # Find a free GPU slot
                gpu_slot = self._find_free_gpu_slot(layer_id)
                if gpu_slot is None:
                    return False, "No free GPU slots available"
                
                # Set the target device
                with torch.cuda.device(target_device_id):
                    try:
                        # Get expert weights from CPU
                        # The experts are stored in the model's expert modules
                        expert_cpu = self._get_expert_weights_from_cpu(layer_id, expert_id)
                        if expert_cpu is None:
                            return False, "Failed to locate expert weights in CPU"
                        
                        # Allocate GPU memory in the expert cache
                        # The experts_cache is 2D: [num_slots, expert_size]
                        # We just need to specify which slot (row) to use
                        gpu_cache_slice = slice(gpu_slot, gpu_slot + 1)
                        
                        # Perform non-blocking copy
                        with torch.cuda.stream(self.copy_stream):
                            # Copy expert weights to GPU cache
                            # experts_cache[slot] is a 1D tensor of size expert_size_elements
                            target = self.context.experts_cache[gpu_slot]
                            source = expert_cpu.view(-1)
                            
                            if target.shape != source.shape:
                                logger.error(f"Shape mismatch: target {target.shape} vs source {source.shape}")
                                return False, f"Shape mismatch when copying to GPU"
                            
                            target.copy_(source, non_blocking=True)
                            
                            # Record event for synchronization
                            event_key = f"gpu_{layer_id}_{expert_id}"
                            if event_key not in self.sync_events:
                                self.sync_events[event_key] = torch.cuda.Event()
                            self.sync_events[event_key].record(self.copy_stream)
                        
                        # Update tracking (this happens immediately)
                        self.expert_tracker.update_expert_location(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            location=MemoryLocation.GPU_PERSISTENT,
                            status=ExpertStatus.CACHED,
                            device_id=target_device_id,
                            cache_slot=gpu_slot
                        )
                        
                        # Update experts_mapping tensor
                        self.execution_engine.experts_mapping[layer_id][expert_id] = gpu_slot
                        
                        logger.info(f"Successfully initiated move of expert L{layer_id}E{expert_id} to GPU slot {gpu_slot}")
                        return True, None
                        
                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"GPU out of memory when moving expert L{layer_id}E{expert_id}")
                        return False, "GPU out of memory"
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            return False, "GPU memory allocation failed"
                        raise
                        
            except Exception as e:
                logger.error(f"Failed to move expert to GPU: {str(e)}")
                return False, f"Failed to move expert: {str(e)}"
    
    def move_expert_to_cpu(self, layer_id: int, expert_id: int) -> Tuple[bool, Optional[str]]:
        """Move an expert from GPU to CPU memory."""
        with self._lock:
            expert_loc = self.expert_tracker.get_expert_location(layer_id, expert_id)
            if not expert_loc:
                return False, "Expert not found"
                
            if expert_loc.location != MemoryLocation.GPU_PERSISTENT:
                return False, "Expert not on GPU"
            
            try:
                # Get the current GPU slot
                gpu_slot = expert_loc.cache_slot
                if gpu_slot < 0:
                    return False, "Invalid GPU cache slot"
                
                # Wait for any pending GPU operations on this expert
                event_key = f"gpu_{layer_id}_{expert_id}"
                if event_key in self.sync_events:
                    self.sync_events[event_key].synchronize()
                
                with torch.cuda.device(expert_loc.device_id):
                    try:
                        # Note: experts_cache is 2D tensor, we access by slot index
                        
                        # Allocate CPU memory if needed
                        cpu_storage = self._get_or_allocate_cpu_storage(layer_id, expert_id)
                        if cpu_storage is None:
                            return False, "Failed to allocate CPU memory"
                        
                        # Perform non-blocking copy from GPU to CPU
                        with torch.cuda.stream(self.copy_stream):
                            # Get the GPU tensor (single row from experts_cache)
                            gpu_tensor = self.context.experts_cache[gpu_slot]
                            
                            if gpu_tensor.numel() == 0:
                                logger.error(f"GPU cache slot {gpu_slot} for expert L{layer_id}E{expert_id} is empty")
                                return False, "GPU cache slot is empty"
                            
                            if gpu_tensor.shape != cpu_storage.shape:
                                logger.error(f"Shape mismatch: GPU {gpu_tensor.shape} vs CPU {cpu_storage.shape}")
                                return False, f"Shape mismatch when copying to CPU"
                            
                            # Copy directly 
                            cpu_storage.copy_(gpu_tensor, non_blocking=True)
                            
                            # Record event for synchronization
                            event_key = f"cpu_{layer_id}_{expert_id}"
                            if event_key not in self.sync_events:
                                self.sync_events[event_key] = torch.cuda.Event()
                            self.sync_events[event_key].record(self.copy_stream)
                        
                        # Free the GPU slot (mark as available)
                        self._mark_gpu_slot_free(gpu_slot)
                        
                        # Update tracking
                        self.expert_tracker.update_expert_location(
                            layer_id=layer_id,
                            expert_id=expert_id,
                            location=MemoryLocation.CPU_MEMORY,
                            status=ExpertStatus.IDLE,
                            device_id=-1,
                            cache_slot=-1
                        )
                        
                        # Update experts_mapping to indicate CPU location
                        # Use a large negative value to indicate CPU storage
                        self.execution_engine.experts_mapping[layer_id][expert_id] = -expert_id - 1
                        
                        logger.info(f"Successfully initiated move of expert L{layer_id}E{expert_id} to CPU")
                        return True, None
                        
                    except MemoryError:
                        logger.error(f"CPU out of memory when moving expert L{layer_id}E{expert_id}")
                        return False, "CPU out of memory"
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            return False, "Memory allocation failed"
                        raise
                        
            except Exception as e:
                logger.error(f"Failed to move expert to CPU: {str(e)}")
                return False, f"Failed to move expert: {str(e)}"
    
    def _find_free_gpu_slot(self, layer_id: int) -> Optional[int]:
        """Find a free GPU slot for the given layer."""
        num_gpu_experts = int(self.model_config.num_local_experts * self.context.policy.wg)
        if num_gpu_experts == 0:
            return None
            
        # Check layer's expert allocation
        base_slot = layer_id * num_gpu_experts
        
        # Look for a free slot in this layer's allocation
        for i in range(num_gpu_experts):
            slot = base_slot + i
            if self._is_slot_free(layer_id, slot):
                return slot
        
        # If no slots available in permanent allocation, check dynamic cache
        # This would require more complex logic in production
        return None
    
    def _is_slot_free(self, layer_id: int, slot: int) -> bool:
        """Check if a GPU slot is free."""
        # Check if any expert in this layer is using this slot
        layer_experts = self.expert_tracker.layers.get(layer_id)
        if not layer_experts:
            return True
            
        for expert_id, expert_loc in layer_experts.experts.items():
            if expert_loc.cache_slot == slot and expert_loc.location == MemoryLocation.GPU_PERSISTENT:
                return False
        return True
    
    def _mark_gpu_slot_free(self, slot: int) -> None:
        """Mark a GPU slot as free for reuse."""
        # In production, this would update a slot allocation table
        # For now, the tracker update is sufficient
        pass
    
    def _get_expert_weights_from_cpu(self, layer_id: int, expert_id: int) -> Optional[torch.Tensor]:
        """Get expert weights from CPU memory."""
        try:
            # Access the model's expert weights
            # This assumes the model has a structure like model.layers[layer_id].experts[expert_id]
            # The actual structure depends on the model implementation
            
            # For FastMoE, experts are typically stored in a format like:
            # model.layers[layer_id].mlp.experts[expert_id]
            # We need to get the concatenated weights (w1, w2, w3)
            
            # Check if we have CPU storage for this expert
            if hasattr(self.execution_engine, 'cpu_expert_storage'):
                storage_key = f"L{layer_id}E{expert_id}"
                if storage_key in self.execution_engine.cpu_expert_storage:
                    return self.execution_engine.cpu_expert_storage[storage_key]
            
            # Otherwise, get from the model's original weights
            # This is a placeholder - actual implementation depends on model structure
            model = self.execution_engine.model_runner.model
            if hasattr(model, 'get_expert_weights'):
                return model.get_expert_weights(layer_id, expert_id)
            
            # Fallback: create dummy weights for testing
            logger.warning(f"Using dummy weights for expert L{layer_id}E{expert_id}")
            return torch.randn(self.expert_size_elements, dtype=self.dtype, device='cpu')
            
        except Exception as e:
            logger.error(f"Failed to get expert weights: {str(e)}")
            return None
    
    def _get_or_allocate_cpu_storage(self, layer_id: int, expert_id: int) -> Optional[torch.Tensor]:
        """Get or allocate CPU storage for an expert."""
        try:
            # Ensure we have a CPU storage dictionary
            if not hasattr(self.execution_engine, 'cpu_expert_storage'):
                self.execution_engine.cpu_expert_storage = {}
            
            storage_key = f"L{layer_id}E{expert_id}"
            
            # Check if storage already exists
            if storage_key in self.execution_engine.cpu_expert_storage:
                return self.execution_engine.cpu_expert_storage[storage_key]
            
            # Allocate new CPU storage
            try:
                cpu_tensor = torch.empty(
                    self.expert_size_elements,
                    dtype=self.dtype,
                    device='cpu',
                    pin_memory=True  # Use pinned memory for faster transfers
                )
                self.execution_engine.cpu_expert_storage[storage_key] = cpu_tensor
                return cpu_tensor
                
            except (RuntimeError, MemoryError) as e:
                logger.error(f"Failed to allocate CPU memory: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get/allocate CPU storage: {str(e)}")
            return None
    
    def wait_for_completion(self, layer_id: int, expert_id: int, target: str = "gpu") -> bool:
        """Wait for a specific expert transfer to complete."""
        event_key = f"{target}_{layer_id}_{expert_id}"
        if event_key in self.sync_events:
            try:
                self.sync_events[event_key].synchronize()
                return True
            except Exception as e:
                logger.error(f"Failed to synchronize expert transfer: {str(e)}")
                return False
        return True  # No pending transfer
    
    def wait_all_transfers(self) -> bool:
        """Wait for all pending transfers to complete."""
        try:
            for event in self.sync_events.values():
                event.synchronize()
            return True
        except Exception as e:
            logger.error(f"Failed to synchronize all transfers: {str(e)}")
            return False
    
    def is_transfer_in_progress(self, layer_id: int, expert_id: int) -> bool:
        """Check if an expert transfer is currently in progress."""
        gpu_key = f"gpu_{layer_id}_{expert_id}"
        cpu_key = f"cpu_{layer_id}_{expert_id}"
        
        for key in [gpu_key, cpu_key]:
            if key in self.sync_events:
                event = self.sync_events[key]
                if not event.query():  # Returns False if event hasn't occurred yet
                    return True
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            with torch.cuda.device(self.device_id):
                gpu_allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)  # GB
                gpu_reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)   # GB
                gpu_total = torch.cuda.get_device_properties(self.device_id).total_memory / (1024**3)  # GB
                
            cpu_storage_size = 0
            if hasattr(self.execution_engine, 'cpu_expert_storage'):
                for tensor in self.execution_engine.cpu_expert_storage.values():
                    cpu_storage_size += tensor.element_size() * tensor.nelement() / (1024**3)  # GB
            
            return {
                "gpu_device_id": self.device_id,
                "gpu_allocated_gb": round(gpu_allocated, 2),
                "gpu_reserved_gb": round(gpu_reserved, 2),
                "gpu_total_gb": round(gpu_total, 2),
                "gpu_free_gb": round(gpu_total - gpu_allocated, 2),
                "cpu_expert_storage_gb": round(cpu_storage_size, 2),
                "pending_transfers": sum(1 for key in self.sync_events if not self.sync_events[key].query())
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {str(e)}")
            return {}
    
    def batch_move_to_gpu(self, expert_list: List[Tuple[int, int]], 
                         target_device_id: Optional[int] = None) -> List[Tuple[bool, Optional[str]]]:
        """Move multiple experts to GPU in batch for efficiency."""
        results = []
        
        # Pre-allocate GPU memory for all experts
        total_size = len(expert_list) * self.expert_size_elements
        
        try:
            # Check if we have enough GPU memory
            with torch.cuda.device(target_device_id or self.device_id):
                free_memory = torch.cuda.get_device_properties(self.device_id).total_memory - torch.cuda.memory_allocated()
                required_memory = total_size * self._get_dtype_size(self.dtype)
                
                if required_memory > free_memory * 0.9:  # Leave 10% buffer
                    return [(False, "Insufficient GPU memory for batch operation")] * len(expert_list)
            
            # Process each expert
            for layer_id, expert_id in expert_list:
                result = self.move_expert_to_gpu(layer_id, expert_id, target_device_id)
                results.append(result)
                
        except Exception as e:
            logger.error(f"Batch move to GPU failed: {str(e)}")
            # Return failure for remaining experts
            while len(results) < len(expert_list):
                results.append((False, str(e)))
        
        return results
    
    def batch_move_to_cpu(self, expert_list: List[Tuple[int, int]]) -> List[Tuple[bool, Optional[str]]]:
        """Move multiple experts to CPU in batch."""
        results = []
        
        for layer_id, expert_id in expert_list:
            result = self.move_expert_to_cpu(layer_id, expert_id)
            results.append(result)
        
        return results
    
    def _get_dtype_size(self, dtype: torch.dtype) -> int:
        """Get the size in bytes of a torch dtype."""
        dummy = torch.empty(1, dtype=dtype)
        return dummy.element_size()
    
    def cleanup_completed_events(self) -> int:
        """Clean up completed sync events to prevent memory leaks."""
        completed_keys = []
        for key, event in self.sync_events.items():
            if event.query():  # Event has completed
                completed_keys.append(key)
        
        for key in completed_keys:
            del self.sync_events[key]
        
        return len(completed_keys)

