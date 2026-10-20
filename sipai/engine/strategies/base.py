from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Protocol
import torch
from sipai.engine.distributed import DistributedContext

class TrainingStrategy(Protocol):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    scheduler: Any | None
    ema: Any | None
    distributed_context: DistributedContext
    checkpoint_backend: str

    @property
    def is_main_process(self) -> bool:
        ...

    @property
    def checkpoint_on_all_ranks(self) -> bool:
        ...

    @property
    def optimizer_step_was_performed(self) -> bool:
        ...

    @property
    def optimizer_step_boundary_was_reached(self) -> bool:
        ...

    def train_mode(self) -> None:
        ...

    def zero_grad(self) -> None:
        ...

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def backward(self, loss: torch.Tensor) -> None:
        ...

    def step(self) -> None:
        ...

    def save_checkpoint(self, path: str | Path, *, step: int=0, extra: Mapping[str, Any] | None=None) -> Path:
        ...

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        ...

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model

def set_model_compute_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    setter = getattr(model, 'set_compute_dtype', None)
    if callable(setter):
        setter(dtype)
