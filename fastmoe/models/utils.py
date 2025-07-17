"""Utility functions for model implementations."""
import re
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from vllm.sequence import IntermediateTensors


def extract_layer_index(prefix: str) -> int:
    """Extract layer index from prefix string."""
    match = re.search(r'\.layers\.(\d+)', prefix)
    if match:
        return int(match.group(1))
    return 0


def is_pp_missing_parameter(name: str, model: nn.Module) -> bool:
    """Check if parameter is missing for pipeline parallelism."""
    # For now, assume all parameters are present
    return False


def make_empty_intermediate_tensors_factory(
    names: List[str],
    hidden_size: int
) -> Callable[[int, torch.dtype, torch.device], IntermediateTensors]:
    """Create a factory for empty intermediate tensors."""
    def factory(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> IntermediateTensors:
        tensors = {}
        for name in names:
            tensors[name] = torch.zeros(
                (batch_size, hidden_size),
                dtype=dtype,
                device=device
            )
        return IntermediateTensors(tensors)
    return factory


def make_layers(
    num_layers: int,
    layer_factory: Callable[[str], nn.Module],
    prefix: str = ""
) -> Tuple[int, int, nn.ModuleList]:
    """Create a list of layers."""
    start_layer = 0
    end_layer = num_layers
    layers = nn.ModuleList([
        layer_factory(f"{prefix}.{i}")
        for i in range(start_layer, end_layer)
    ])
    return start_layer, end_layer, layers


def maybe_prefix(prefix: str, name: str) -> str:
    """Add prefix to name if prefix is not empty."""
    if prefix:
        return f"{prefix}.{name}"
    return name


class AutoWeightsLoader:
    """Automatic weights loader for models."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def load_weights(self, weights) -> set[str]:
        """Load weights into the model."""
        return self.model.load_weights(weights)