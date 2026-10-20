from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import torch
from sipai.engine.distributed import DistributedContext
from sipai.engine.strategies import TrainingStrategy, build_training_strategy
from sipai.engine.trainers import SimpleTrainer
from sipai.registry import RUNNERS

@RUNNERS.register_module()
class TrainRunner:

    def __init__(self, *, model: torch.nn.Module, dataloader: Iterable[Mapping[str, Any]], optimizer: torch.optim.Optimizer | Mapping[str, Any] | None=None, scheduler: Any | Mapping[str, Any] | None=None, max_steps: int=1, hooks: Sequence[Any] | None=None, checkpoint_dir: str | Path | None=None, save_interval: int | None=None, resume_from: str | Path | None=None, device: str | torch.device | None=None, distributed_context: DistributedContext | None=None, strategy: TrainingStrategy | Mapping[str, Any] | None=None, logging_frequency: int=25, checkpoint_extra: Mapping[str, Any] | None=None, defer_final_interval_checkpoint: bool=False, checkpoint_saved_callback: Callable[[Path, int], None] | None=None):
        distributed_context = distributed_context or DistributedContext(enabled=False)
        training_strategy = strategy if hasattr(strategy, 'forward') and hasattr(strategy, 'backward') else build_training_strategy(strategy if isinstance(strategy, Mapping) else None, model=model, optimizer=optimizer, scheduler=scheduler, device=device, distributed_context=distributed_context)
        self.trainer = SimpleTrainer(model=training_strategy.model, dataloader=dataloader, optimizer=training_strategy.optimizer, scheduler=training_strategy.scheduler, max_steps=max_steps, hooks=hooks, checkpoint_dir=checkpoint_dir, save_interval=save_interval, resume_from=resume_from, device=device, distributed_context=training_strategy.distributed_context, move_model_to_device=False, strategy=training_strategy, logging_frequency=logging_frequency, checkpoint_extra=checkpoint_extra, defer_final_interval_checkpoint=defer_final_interval_checkpoint, checkpoint_saved_callback=checkpoint_saved_callback)

    def train(self) -> dict[str, Any]:
        return self.trainer.train()
