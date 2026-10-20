from __future__ import annotations
from typing import Any, Mapping
import torch
from sipai.engine.distributed import DistributedContext
from sipai.engine.strategies.base import TrainingStrategy, unwrap_model
from sipai.engine.strategies.fsdp import FSDPStrategy

def _plain_strategy_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, 'items'):
        return {key: value for (key, value) in cfg.items()}
    return dict(cfg)

def build_training_strategy(cfg: Mapping[str, Any] | None=None, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | Mapping[str, Any] | None=None, scheduler: Any | Mapping[str, Any] | None=None, device: str | torch.device | None=None, distributed_context: DistributedContext | None=None, move_model_to_device: bool=True) -> TrainingStrategy:
    strategy_cfg = _plain_strategy_cfg(cfg)
    strategy_type = str(strategy_cfg.pop('type', 'FSDPStrategy'))
    if strategy_type == 'FSDPStrategy':
        strategy_cfg = {key: value for (key, value) in strategy_cfg.items() if value is not None}
        strategy_cfg.setdefault('move_model_to_device', move_model_to_device)
        return FSDPStrategy(model=model, optimizer=optimizer, scheduler=scheduler, device=device, distributed_context=distributed_context, **strategy_cfg)
    raise ValueError(f'Unsupported training strategy type: {strategy_type}')
__all__ = ['FSDPStrategy', 'TrainingStrategy', 'build_training_strategy', 'unwrap_model']
