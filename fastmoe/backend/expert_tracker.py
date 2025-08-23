"""Expert location tracking system for FastMoE.

This module provides data structures and utilities to track where experts
are located in memory (GPU, CPU, pinned memory) and their status.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import time
import torch
from threading import Lock


class MemoryLocation(str, Enum):
    """Enum for different memory locations where experts can reside."""
    GPU_PERSISTENT = "gpu_persistent"      # Permanently on GPU
    GPU_CACHE = "gpu_cache"               # In GPU cache (can be evicted)
    CPU_MEMORY = "cpu_memory"             # In CPU memory
    CPU_PINNED = "cpu_pinned"             # In CPU pinned memory (transfer buffer)
    NOT_LOADED = "not_loaded"             # Not loaded yet


class ExpertStatus(str, Enum):
    """Status of an expert."""
    ACTIVE = "active"                     # Currently being used
    CACHED = "cached"                     # In cache, ready to use
    LOADING = "loading"                   # Being loaded/transferred
    IDLE = "idle"                        # Available but not active
    NOT_AVAILABLE = "not_available"       # Not loaded


@dataclass
class ExpertLocation:
    """Tracks the location and status of a single expert."""
    expert_id: int
    layer_id: int
    location: MemoryLocation
    status: ExpertStatus
    device_id: int = -1                   # GPU device ID (-1 for CPU)
    cache_slot: int = -1                  # Index in cache (if applicable)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0                   # Size in bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass 
class LayerExpertMap:
    """Tracks all experts in a single layer."""
    layer_id: int
    num_experts: int
    experts: Dict[int, ExpertLocation] = field(default_factory=dict)
    
    def get_experts_by_location(self, location: MemoryLocation) -> List[int]:
        """Get list of expert IDs at a specific location."""
        return [
            exp_id for exp_id, exp_loc in self.experts.items()
            if exp_loc.location == location
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this layer."""
        location_counts = {}
        for location in MemoryLocation:
            location_counts[location.value] = len(self.get_experts_by_location(location))
        
        return {
            "layer_id": self.layer_id,
            "total_experts": self.num_experts,
            "location_distribution": location_counts,
            "loaded_experts": self.num_experts - location_counts.get(MemoryLocation.NOT_LOADED.value, 0)
        }


class ExpertTracker:
    """Main class for tracking expert locations across all layers."""
    
    def __init__(self, num_layers: int, num_experts_per_layer: int, 
                 num_gpu_experts: int, cache_size: int):
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.gpu_capacity = num_gpu_experts  # Using gpu_capacity instead of num_gpu_experts for clarity
        self.cache_size = cache_size
        self.layers: Dict[int, LayerExpertMap] = {}
        self._lock = Lock()
        
        # Initialize layer maps
        for layer_id in range(num_layers):
            self.layers[layer_id] = LayerExpertMap(
                layer_id=layer_id,
                num_experts=num_experts_per_layer
            )
            
            # Initialize all experts as not loaded
            for expert_id in range(num_experts_per_layer):
                self.layers[layer_id].experts[expert_id] = ExpertLocation(
                    expert_id=expert_id,
                    layer_id=layer_id,
                    location=MemoryLocation.NOT_LOADED,
                    status=ExpertStatus.NOT_AVAILABLE
                )
    
    def update_expert_location(self, layer_id: int, expert_id: int,
                             location: MemoryLocation, 
                             status: ExpertStatus = ExpertStatus.IDLE,
                             device_id: int = -1,
                             cache_slot: int = -1) -> None:
        """Update the location of a specific expert."""
        with self._lock:
            if layer_id in self.layers and expert_id in self.layers[layer_id].experts:
                expert = self.layers[layer_id].experts[expert_id]
                expert.location = location
                expert.status = status
                expert.device_id = device_id
                expert.cache_slot = cache_slot
                expert.last_accessed = time.time()
    
    def mark_expert_accessed(self, layer_id: int, expert_id: int) -> None:
        """Mark that an expert was accessed."""
        with self._lock:
            if layer_id in self.layers and expert_id in self.layers[layer_id].experts:
                expert = self.layers[layer_id].experts[expert_id]
                expert.last_accessed = time.time()
                expert.access_count += 1
    
    def get_expert_location(self, layer_id: int, expert_id: int) -> Optional[ExpertLocation]:
        """Get the location info for a specific expert."""
        with self._lock:
            if layer_id in self.layers and expert_id in self.layers[layer_id].experts:
                return self.layers[layer_id].experts[expert_id]
        return None
    
    def get_layer_summary(self, layer_id: int) -> Optional[Dict[str, Any]]:
        """Get summary for a specific layer."""
        with self._lock:
            if layer_id in self.layers:
                return self.layers[layer_id].get_summary()
        return None
    
    def get_global_summary(self) -> Dict[str, Any]:
        """Get global summary of all expert locations."""
        with self._lock:
            total_location_counts = {}
            for location in MemoryLocation:
                total_location_counts[location.value] = 0
            
            layer_summaries = []
            total_loaded = 0
            total_accessed = 0
            
            for layer_id, layer_map in self.layers.items():
                summary = layer_map.get_summary()
                layer_summaries.append(summary)
                
                # Aggregate location counts
                for loc, count in summary["location_distribution"].items():
                    total_location_counts[loc] += count
                
                total_loaded += summary["loaded_experts"]
                
                # Count accessed experts
                for expert in layer_map.experts.values():
                    if expert.access_count > 0:
                        total_accessed += 1
            
            return {
                "num_layers": self.num_layers,
                "experts_per_layer": self.num_experts_per_layer,
                "total_experts": self.num_layers * self.num_experts_per_layer,
                "total_loaded": total_loaded,
                "total_accessed": total_accessed,
                "gpu_capacity": self.gpu_capacity,
                "cache_capacity": self.cache_size,
                "global_location_distribution": total_location_counts,
                "layer_summaries": layer_summaries
            }
    
    def get_all_expert_locations(self) -> List[Dict[str, Any]]:
        """Get detailed location info for all experts."""
        with self._lock:
            all_experts = []
            for layer_id, layer_map in self.layers.items():
                for expert_id, expert_loc in layer_map.experts.items():
                    all_experts.append({
                        "layer_id": expert_loc.layer_id,
                        "expert_id": expert_loc.expert_id,
                        "location": expert_loc.location.value,
                        "status": expert_loc.status.value,
                        "device_id": expert_loc.device_id,
                        "cache_slot": expert_loc.cache_slot,
                        "last_accessed": expert_loc.last_accessed,
                        "access_count": expert_loc.access_count,
                        "size_bytes": expert_loc.size_bytes
                    })
            return all_experts
    
    def sync_with_experts_mapping(self, layer_id: int, 
                                experts_mapping: torch.Tensor,
                                num_gpu_experts: int) -> None:
        """Sync tracker state with the actual experts_mapping tensor."""
        with self._lock:
            if layer_id not in self.layers:
                return
                
            mapping = experts_mapping.cpu().numpy()
            
            for expert_id in range(len(mapping)):
                cache_idx = int(mapping[expert_id])
                
                # Determine location based on cache index
                if expert_id < num_gpu_experts:
                    # Persistent GPU expert
                    location = MemoryLocation.GPU_PERSISTENT
                    device_id = 0
                elif cache_idx >= 0:
                    # In GPU cache
                    location = MemoryLocation.GPU_CACHE
                    device_id = 0
                else:
                    # On CPU
                    location = MemoryLocation.CPU_MEMORY
                    device_id = -1
                
                self.update_expert_location(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    location=location,
                    status=ExpertStatus.CACHED,
                    device_id=device_id,
                    cache_slot=cache_idx if cache_idx >= 0 else -1
                )
