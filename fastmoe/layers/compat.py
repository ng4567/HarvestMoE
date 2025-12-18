"""Compatibility layer for vLLM API changes."""
import os
from typing import Optional, Generator, Tuple
import torch
from huggingface_hub import snapshot_download

from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    safetensors_weights_iterator,
    pt_weights_iterator,
)


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
    fall_back_to_pt: bool = True,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Compatibility wrapper for the old hf_model_weights_iterator function.
    Downloads model weights from HuggingFace and iterates over them.
    """
    # Download the model if it's a HF model ID
    if os.path.isdir(model_name_or_path):
        model_path = model_name_or_path
    else:
        # Download from HuggingFace Hub
        allow_patterns = ["*.safetensors", "*.bin"] if fall_back_to_pt else ["*.safetensors"]
        model_path = snapshot_download(
            model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            allow_patterns=allow_patterns,
        )
    
    # Find weight files
    import glob
    safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    pt_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
    
    # Prefer safetensors if available
    if safetensor_files:
        yield from safetensors_weights_iterator(
            safetensor_files,
            use_tqdm_on_load=True,
        )
    elif pt_files and fall_back_to_pt:
        yield from pt_weights_iterator(pt_files)
    else:
        raise ValueError(f"No weight files found in {model_path}")


# Re-export for convenience
__all__ = ['hf_model_weights_iterator', 'default_weight_loader']

