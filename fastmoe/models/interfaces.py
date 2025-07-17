"""Interfaces for model implementations."""
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from vllm.sequence import IntermediateTensors


class SupportsPP(ABC):
    """Interface for models that support pipeline parallelism."""
    
    @abstractmethod
    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """Create empty intermediate tensors for pipeline parallelism."""
        pass