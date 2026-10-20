from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import torch
import torch.distributed as dist
from sipai.engine.distributed import DistributedContext, barrier
from sipai.engine.hooks import Hook
from sipai.engine.strategies import TrainingStrategy, build_training_strategy
from sipai.registry import TRAINERS
from sipai.utils import ensure_shared_artifact_directory, normalize_shared_artifact_permissions, normalize_shared_artifact_tree

@TRAINERS.register_module()
class SimpleTrainer:

    def __init__(self, *, model: torch.nn.Module, dataloader: Iterable[Mapping[str, Any]], optimizer: torch.optim.Optimizer, scheduler: Any | None=None, max_steps: int=1, hooks: Sequence[Hook] | None=None, checkpoint_dir: str | Path | None=None, save_interval: int | None=None, resume_from: str | Path | None=None, device: str | torch.device | None=None, loss_key: str='action_loss', distributed_context: DistributedContext | None=None, move_model_to_device: bool=True, strategy: TrainingStrategy | None=None, logging_frequency: int=25, checkpoint_extra: Mapping[str, Any] | None=None, defer_final_interval_checkpoint: bool=False, checkpoint_saved_callback: Callable[[Path, int], None] | None=None):
        self.strategy = strategy or build_training_strategy(None, model=model, optimizer=optimizer, scheduler=scheduler, device=device, distributed_context=distributed_context, move_model_to_device=move_model_to_device)
        self.model = self.strategy.model
        self.dataloader = dataloader
        self.optimizer = self.strategy.optimizer
        self.scheduler = self.strategy.scheduler
        self.max_steps = int(max_steps)
        self.hooks = list(hooks or [])
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.save_interval = int(save_interval) if save_interval else None
        self.device = torch.device(device) if device is not None else None
        self.loss_key = loss_key
        self.distributed_context = self.strategy.distributed_context
        self.completed_steps = 0
        self.last_outputs: Mapping[str, Any] | None = None
        self.metrics_path = self._default_metrics_path()
        self.logging_frequency = max(0, int(logging_frequency or 0))
        self.checkpoint_extra = dict(checkpoint_extra or {})
        self.defer_final_interval_checkpoint = bool(defer_final_interval_checkpoint)
        self.checkpoint_saved_callback = checkpoint_saved_callback
        self._train_started_at = 0.0
        self._window_records: list[dict[str, Any]] = []
        self._pending_scalar_sums: dict[str, torch.Tensor] = {}
        self._pending_scalar_count = 0
        if resume_from is not None:
            self.load_checkpoint(resume_from)

    def train(self) -> dict[str, Any]:
        self.strategy.train_mode()
        self._call('before_train')
        self._train_started_at = time.perf_counter()
        epoch = self._current_epoch()
        self._set_sampler_epoch(epoch)
        data_iter = iter(self.dataloader)
        data_iter = self._skip_seen_batches(data_iter)
        data_time_accum = 0.0
        model_time_accum = 0.0
        pending_final_interval_checkpoint = False
        while self.completed_steps < self.max_steps:
            data_started_at = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                self._set_sampler_epoch(epoch)
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            data_time_accum += time.perf_counter() - data_started_at
            model_started_at = time.perf_counter()
            outputs = self.train_step(batch)
            model_time_accum += time.perf_counter() - model_started_at
            self.last_outputs = outputs
            if self.logging_frequency > 0:
                self._accumulate_scalar_outputs(outputs)
            step_boundary = bool(getattr(self.strategy, 'optimizer_step_boundary_was_reached', self.strategy.optimizer_step_was_performed))
            if step_boundary:
                if self.strategy.optimizer_step_was_performed:
                    self.completed_steps += 1
                    if self.logging_frequency > 0:
                        self._record_step_metrics(self._pending_scalar_averages(), data_seconds=data_time_accum, model_seconds=model_time_accum)
                    if self._should_defer_final_interval_checkpoint():
                        pending_final_interval_checkpoint = True
                    elif self._should_save_interval_checkpoint():
                        self.save_checkpoint()
                self._clear_pending_scalar_outputs()
                data_time_accum = 0.0
                model_time_accum = 0.0
        try:
            self._call('after_train')
        except BaseException as after_train_error:
            if pending_final_interval_checkpoint:
                try:
                    self.save_checkpoint()
                except BaseException as checkpoint_error:
                    note = f'The deferred final interval checkpoint also failed to save: {type(checkpoint_error).__name__}: {checkpoint_error}'
                    add_note = getattr(after_train_error, 'add_note', None)
                    if callable(add_note):
                        add_note(note)
                    raise after_train_error from checkpoint_error
            raise
        return {'completed_steps': self.completed_steps, 'last_outputs': self.last_outputs}

    def train_step(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        self._call('before_step', batch)
        self.strategy.zero_grad()
        batch = self._move_batch_to_device(batch)
        outputs = self.strategy.forward(batch)
        if self.loss_key not in outputs:
            raise KeyError(f'Model output is missing loss key `{self.loss_key}`')
        loss = outputs[self.loss_key]
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss, dtype=torch.float32)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Loss `{self.loss_key}` is not finite: {loss}')
        self.strategy.backward(loss)
        self.strategy.step()
        self._call('after_step', batch, outputs)
        return outputs

    def save_checkpoint(self, path: str | Path | None=None, *, extra: Mapping[str, Any] | None=None) -> Path:
        checkpoint_on_all_ranks = bool(getattr(self.strategy, 'checkpoint_on_all_ranks', False))
        if not self.distributed_context.is_main_process and (not checkpoint_on_all_ranks):
            if path is None:
                path = self._checkpoint_path()
            barrier(self.distributed_context)
            return Path(path)
        if path is None:
            path = self._checkpoint_path()
        merged_extra = {**self.checkpoint_extra, **dict(extra or {})}
        if 'state_action_representation' in merged_extra:
            raise ValueError('Normalization statistics must not be embedded in checkpoint metadata; save config.yaml and state_action_statistics.json beside the checkpoint instead.')
        checkpoint_path = self.strategy.save_checkpoint(path, step=self.completed_steps, extra=merged_extra or None)
        barrier(self.distributed_context)
        if self.distributed_context.is_main_process:
            ensure_shared_artifact_directory(checkpoint_path.parent)
            checkpoint_backend = str(getattr(self.strategy, 'checkpoint_backend', 'torch_full_state'))
            if checkpoint_path.is_dir() and checkpoint_backend != 'torch_dcp':
                normalize_shared_artifact_tree(checkpoint_path)
            elif checkpoint_path.is_file():
                normalize_shared_artifact_permissions(checkpoint_path)
            if self.checkpoint_saved_callback is not None:
                self.checkpoint_saved_callback(checkpoint_path, self.completed_steps)
        return checkpoint_path

    def _checkpoint_path(self) -> Path:
        if self.checkpoint_dir is None:
            raise ValueError('checkpoint_dir is required when path is omitted')
        backend = str(getattr(self.strategy, 'checkpoint_backend', 'torch_full_state'))
        suffix = '.pt' if backend in {'torch_full_state'} else ''
        return self.checkpoint_dir / f'step_{self.completed_steps}{suffix}'

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        state = self.strategy.load_checkpoint(path)
        self.completed_steps = int(state.get('step', 0))
        return state

    def _call(self, hook_name: str, *args: Any) -> None:
        for hook in self.hooks:
            getattr(hook, hook_name)(self, *args)

    def _should_save_interval_checkpoint(self) -> bool:
        if self._should_defer_final_interval_checkpoint():
            return False
        return self._is_interval_checkpoint_step()

    def _should_defer_final_interval_checkpoint(self) -> bool:
        return self.defer_final_interval_checkpoint and self.completed_steps >= self.max_steps and self._is_interval_checkpoint_step()

    def _is_interval_checkpoint_step(self) -> bool:
        if self.save_interval is None or self.save_interval <= 0:
            return False
        if self.checkpoint_dir is None:
            return False
        return self.completed_steps > 0 and self.completed_steps % self.save_interval == 0

    def _default_metrics_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir.parent / 'metrics.jsonl'

    def _record_step_metrics(self, scalar_outputs: Mapping[str, float], *, data_seconds: float, model_seconds: float) -> None:
        if self.logging_frequency <= 0:
            return
        step_record = self._build_step_record(scalar_outputs, data_seconds=data_seconds, model_seconds=model_seconds)
        self._window_records.append(step_record)
        if not self._should_log_step():
            return
        record = self._build_log_record(step_record)
        if self.distributed_context.is_main_process:
            if self.metrics_path is not None:
                ensure_shared_artifact_directory(self.metrics_path.parent)
                with self.metrics_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, sort_keys=True) + '\n')
                    f.flush()
                normalize_shared_artifact_permissions(self.metrics_path)
            self._call('after_log', record)
            print(self._format_log_line(record), flush=True)
        self._window_records.clear()

    def _build_step_record(self, scalar_outputs: Mapping[str, float], *, data_seconds: float, model_seconds: float) -> dict[str, Any]:
        record: dict[str, Any] = {'step': self.completed_steps, 'data_seconds': float(data_seconds), 'model_seconds': float(model_seconds), 'step_seconds': float(data_seconds + model_seconds)}
        record.update({str(key): float(value) for (key, value) in scalar_outputs.items()})
        return record

    def _build_log_record(self, latest: Mapping[str, Any]) -> dict[str, Any]:
        window_size = max(1, len(self._window_records))
        local_values = {'timing/data': sum((float(item['data_seconds']) for item in self._window_records)) / window_size, 'timing/model': sum((float(item['model_seconds']) for item in self._window_records)) / window_size, 'timing/step': sum((float(item['step_seconds']) for item in self._window_records)) / window_size}
        reserved = {'step', 'data_seconds', 'model_seconds', 'step_seconds'}
        metric_keys = sorted((key for (key, value) in latest.items() if key not in reserved and isinstance(value, (int, float))))
        for key in metric_keys:
            values = [float(item[key]) for item in self._window_records if key in item]
            local_values[key] = sum(values) / len(values)
            local_values[f'step/{key}'] = float(latest[key])
        reduced_values = self._global_mean_values(local_values)
        step_avg = float(reduced_values['timing/step'])
        elapsed_seconds = max(0.0, time.perf_counter() - self._train_started_at)
        eta_seconds = max(0.0, (self.max_steps - self.completed_steps) * step_avg)
        record: dict[str, Any] = {'step': self.completed_steps, 'max_steps': self.max_steps, 'epoch': self._estimate_loader_epoch(), 'lr': self._current_lr(), **reduced_values, 'logging/window_steps': window_size, 'logging/world_size': max(1, int(self.distributed_context.world_size)), 'elapsed_seconds': elapsed_seconds, 'eta_seconds': eta_seconds}
        if self.loss_key in record:
            record['loss'] = record[self.loss_key]
            record['step/loss'] = record[f'step/{self.loss_key}']
        return record

    def _should_log_step(self) -> bool:
        if self.logging_frequency <= 0:
            return False
        return len(self._window_records) >= self.logging_frequency or self.completed_steps >= self.max_steps

    def _format_log_line(self, record: Mapping[str, Any]) -> str:
        step = int(record.get('step', 0))
        max_steps = max(1, int(record.get('max_steps', self.max_steps)))
        progress = min(1.0, max(0.0, step / max_steps))
        percent = int(progress * 100)
        bar = self._progress_bar(progress)
        elapsed = self._format_duration(float(record.get('elapsed_seconds', 0.0)))
        eta = self._format_duration(float(record.get('eta_seconds', 0.0)))
        loss_items = self._format_loss_items(record)
        step_time = float(record.get('timing/step', 0.0))
        data_time = float(record.get('timing/data', 0.0))
        model_time = float(record.get('timing/model', 0.0))
        epoch = float(record.get('epoch', 0.0))
        parts = [f'{percent:3d}%|{bar}| {step}/{max_steps} [{elapsed}<{eta}, {step_time:.2f}s/it', f'data_times={data_time:.3f}', f'model_times={model_time:.3f}', f'epoch={epoch:.2f}']
        if loss_items:
            parts.append(loss_items)
        lr = record.get('lr', None)
        if lr is not None:
            parts.append(f'lr={float(lr):.2e}')
        return ', '.join(parts) + ']'

    def _format_loss_items(self, record: Mapping[str, Any]) -> str:
        loss_keys = [key for key in record if 'loss' in str(key).lower()]
        loss_keys.sort(key=lambda key: (key != 'loss', key != 'step/loss', key != self.loss_key, str(key)))
        items = []
        for key in loss_keys:
            value = record[key]
            if isinstance(value, (int, float)):
                items.append(f'{key}={float(value):.4f}')
        return ', '.join(items)

    def _accumulate_scalar_outputs(self, outputs: Mapping[str, Any]) -> None:
        loss = outputs[self.loss_key]
        if torch.is_tensor(loss):
            metric_device = loss.device
        elif self.device is not None:
            metric_device = self.device
        else:
            metric_device = torch.device('cpu')
        values: dict[str, torch.Tensor] = {}
        for (key, value) in outputs.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                values[str(key)] = value.detach().to(device=metric_device, dtype=torch.float32).reshape(())
            elif isinstance(value, (int, float)):
                values[str(key)] = torch.tensor(float(value), device=metric_device, dtype=torch.float32)
        if self._pending_scalar_sums and values.keys() != self._pending_scalar_sums.keys():
            raise RuntimeError('Scalar model outputs changed within one optimizer step.')
        if not self._pending_scalar_sums:
            self._pending_scalar_sums = {key: value.clone() for (key, value) in values.items()}
        else:
            for (key, value) in values.items():
                self._pending_scalar_sums[key].add_(value)
        self._pending_scalar_count += 1

    def _pending_scalar_averages(self) -> dict[str, float]:
        if self._pending_scalar_count <= 0:
            raise RuntimeError('No scalar metrics were accumulated for the optimizer step.')
        return {key: float((value / self._pending_scalar_count).cpu()) for (key, value) in self._pending_scalar_sums.items()}

    def _clear_pending_scalar_outputs(self) -> None:
        self._pending_scalar_sums.clear()
        self._pending_scalar_count = 0

    def _global_mean_values(self, values: Mapping[str, float]) -> dict[str, float]:
        world_size = max(1, int(self.distributed_context.world_size))
        if world_size == 1:
            return dict(values)
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError('Distributed metric reduction requires an initialized process group.')
        backend = self.distributed_context.backend or str(dist.get_backend())
        if backend == 'nccl':
            device = torch.device('cuda', self.distributed_context.local_rank)
        else:
            device = torch.device('cpu')
        keys = list(values)
        gathered_keys: list[tuple[str, ...] | None] = [None] * world_size
        dist.all_gather_object(gathered_keys, tuple(keys))
        if any((rank_keys != gathered_keys[0] for rank_keys in gathered_keys[1:])):
            raise RuntimeError(f'Scalar metric keys differ across global ranks: {gathered_keys}.')
        packed = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
        dist.all_reduce(packed)
        packed.div_(world_size)
        reduced = packed.cpu().tolist()
        return {key: float(value) for (key, value) in zip(keys, reduced, strict=True)}

    def _progress_bar(self, progress: float, width: int=10) -> str:
        filled = int(progress * width)
        if filled >= width:
            return '█' * width
        partials = '▏▎▍▌▋▊▉'
        remainder = progress * width - filled
        partial = '' if remainder <= 0 else partials[min(len(partials) - 1, max(0, int(remainder * len(partials))))]
        bar = '█' * filled + partial
        return bar + ' ' * max(0, width - len(bar))

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        (hours, rem) = divmod(seconds, 3600)
        (minutes, secs) = divmod(rem, 60)
        if hours > 0:
            return f'{hours}:{minutes:02d}:{secs:02d}'
        return f'{minutes:02d}:{secs:02d}'

    def _current_lr(self) -> float | None:
        if self.optimizer is None:
            return None
        param_groups = getattr(self.optimizer, 'param_groups', None)
        if not param_groups:
            return None
        return float(param_groups[0].get('lr', 0.0))

    def _skip_seen_batches(self, data_iter: Any) -> Any:
        if self.completed_steps <= 0:
            return data_iter
        try:
            dataloader_len = len(self.dataloader)
        except TypeError:
            dataloader_len = None
        seen_micro_batches = self.completed_steps * int(getattr(self.strategy, 'gradient_accumulation_steps', 1))
        skip = seen_micro_batches if not dataloader_len else seen_micro_batches % int(dataloader_len)
        for _ in range(skip):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
        return data_iter

    def _current_epoch(self) -> int:
        return int(self._estimate_loader_epoch())

    def _estimate_epoch(self) -> float:
        raw_dataset_len = self._raw_dataset_len()
        per_device_batch_size = self._per_device_batch_size()
        if raw_dataset_len is not None and raw_dataset_len > 0 and (per_device_batch_size is not None):
            grad_accum = int(getattr(self.strategy, 'gradient_accumulation_steps', 1))
            world_size = max(1, int(getattr(self.distributed_context, 'world_size', 1)))
            global_seen_samples = self.completed_steps * grad_accum * world_size * per_device_batch_size
            return global_seen_samples / float(raw_dataset_len)
        try:
            dataloader_len = len(self.dataloader)
        except TypeError:
            return 0.0
        if dataloader_len <= 0:
            return 0.0
        seen_micro_batches = self.completed_steps * int(getattr(self.strategy, 'gradient_accumulation_steps', 1))
        return seen_micro_batches / float(dataloader_len)

    def _estimate_loader_epoch(self) -> float:
        try:
            dataloader_len = len(self.dataloader)
        except TypeError:
            return 0.0
        if dataloader_len <= 0:
            return 0.0
        seen_micro_batches = self.completed_steps * int(getattr(self.strategy, 'gradient_accumulation_steps', 1))
        return seen_micro_batches / float(dataloader_len)

    def _raw_dataset_len(self) -> int | None:
        dataset = getattr(self.dataloader, 'dataset', None)
        if dataset is None:
            return None
        try:
            return int(len(dataset))
        except TypeError:
            return None

    def _per_device_batch_size(self) -> int | None:
        batch_size = getattr(self.dataloader, 'batch_size', None)
        if batch_size is None:
            batch_sampler = getattr(self.dataloader, 'batch_sampler', None)
            batch_size = getattr(batch_sampler, 'batch_size', None)
        if batch_size is None:
            return None
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            return None
        return batch_size if batch_size > 0 else None

    def _set_sampler_epoch(self, epoch: int) -> None:
        sampler = getattr(self.dataloader, 'sampler', None)
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(int(epoch))

    def _move_batch_to_device(self, batch: Any) -> Any:
        if self.device is None:
            return batch
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, Mapping):
            return {key: self._move_batch_to_device(value) for (key, value) in batch.items()}
        if isinstance(batch, tuple):
            return tuple((self._move_batch_to_device(value) for value in batch))
        if isinstance(batch, list):
            return [self._move_batch_to_device(value) for value in batch]
        return batch
