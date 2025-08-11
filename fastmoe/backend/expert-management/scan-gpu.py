import torch
from fastmoe.utils.model_config import ModelConfig
from dataclasses import dataclass, field
from threading import Lock, Event
from enum import Enum, auto
import time

class Loc(Enum): CPU=auto(); PIN=auto(); GPU_PERMA=auto(); GPU_SLOT=auto(); GPU_ADHOC=auto()
class Status(Enum): MISSING=auto(); LOADING=auto(); RESIDENT=auto(); EVICTING=auto()

@dataclass
class ExpertRec:
    layer: int
    idx: int
    status: Status = Status.MISSING
    loc: Loc = Loc.CPU
    device: int = -1
    slot: int = -1          # index into experts_cache rows
    pins: int = 0
    last_used: float = field(default_factory=time.time)
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    ready: Event = field(default_factory=Event, repr=False, compare=False)


class ExpertDirectory:
    def __init__(self, model_cfg, context):
        self.model_cfg = model_cfg
        self.cx = context
        self.records: dict[tuple[int,int], ExpertRec] = {}
        # free list of dynamic GPU slots (indexes into cx.experts_cache)
        self.free_gpu_slots: list[int] = self._init_free_slots()

    def _init_free_slots(self) -> list[int]:
        # Build the list of dynamic rows in cx.experts_cache that are not perma
        # You already compute these ranges in init_gpu_experts(); reuse that logic:
        # e.g., take the rows that correspond to the 2 * page_size buffer region.
        start = self.cx.get_ecache_size() - self.cx.weights_prefetch_num_pages_gpu * (self.cx.page_size)
        return list(range(start, self.cx.get_ecache_size()))

    def get(self, layer, idx) -> ExpertRec:
        k = (layer, idx)
        if k not in self.records:
            self.records[k] = ExpertRec(layer, idx)
        return self.records[k]

num_gpus = torch.cuda.device_count()

mc = ModelConfig("mistralai/Mixtral-8x7B-Instruct-v0.1")

# If you add TP later, set this accordingly
tp_size = 1


dtype = torch.float16
bytes_per_elem = torch.tensor([], dtype=dtype).element_size()

# Per expert per layer (per TP rank)
elem_dim_per_layer = 3 * (mc.intermediate_size // tp_size) * mc.hidden_size
bytes_per_expert_per_layer = elem_dim_per_layer * bytes_per_elem
# Full expert across all layers on this rank
bytes_per_expert_all_layers = mc.num_hidden_layers * bytes_per_expert_per_layer


def get_num_experts_that_fit(gpu_id: int, full_expert: bool = True, safety: float = 0.8) -> int:
    # Query free memory on the target device
    with torch.cuda.device(gpu_id):
        free_bytes, _ = torch.cuda.mem_get_info()
    budget = int(free_bytes * safety)
    unit = bytes_per_expert_all_layers if full_expert else bytes_per_expert_per_layer
    return budget // unit


if __name__ == "__main__":
    for gpu in range(num_gpus):
        full = get_num_experts_that_fit(gpu, full_expert=True)
        per_layer = get_num_experts_that_fit(gpu, full_expert=False)
        print(f"GPU {gpu}: full experts={full}, per-layer experts={per_layer}")
