from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import torch
from sipai.engine.distributed import DistributedContext
from sipai.engine.ema import ExponentialMovingAverage
from sipai.engine.optim import build_optimizer, build_scheduler, materialize_optimizer_state
from sipai.engine.strategies.base import set_model_compute_dtype

class FSDPStrategy:
    checkpoint_backend = 'torch_dcp'

    def __init__(self, *, model: torch.nn.Module, optimizer: Mapping[str, Any] | None=None, scheduler: Mapping[str, Any] | None=None, device: str | torch.device | None=None, distributed_context: DistributedContext | None=None, mode: str='hybrid_shard', precision: str='fp32', wrap_policy: str='root', min_num_params: int=100000000, module_classes: Sequence[str | type[torch.nn.Module]] | str | type[torch.nn.Module] | None=None, shard_size: str | int='auto', move_model_to_device: bool=True, gradient_accumulation_steps: int=1, max_grad_norm: float | None=None, ema_decay: float | None=None) -> None:
        if optimizer is not None and (not isinstance(optimizer, Mapping)):
            raise TypeError('FSDPStrategy requires an optimizer config (or None), because the optimizer must be built after fully_shard().')
        if scheduler is not None and (not isinstance(scheduler, Mapping)):
            raise TypeError('FSDPStrategy requires a scheduler config (or None), because its optimizer is built after fully_shard().')
        self.distributed_context = distributed_context or DistributedContext(enabled=False)
        self.device = torch.device(device) if device is not None else None
        self.mode = _normalize_mode(mode)
        self.precision = _normalize_precision(precision)
        self.wrap_policy = _normalize_wrap_policy(wrap_policy)
        self.min_num_params = int(min_num_params)
        if self.min_num_params <= 0:
            raise ValueError(f'min_num_params must be positive, got {self.min_num_params}.')
        self.module_classes = _normalize_module_classes(module_classes)
        if self.wrap_policy == 'module_classes' and (not self.module_classes):
            raise ValueError("module_classes must not be empty when wrap_policy='module_classes'.")
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f'max_grad_norm must be positive, got {max_grad_norm!r}.')
        self._accumulation_index = 0
        self._optimizer_step_was_performed = False
        self._optimizer_step_boundary_was_reached = False
        _validate_materialized_parameters(model)
        (fully_shard, mixed_precision_policy_cls, init_device_mesh) = _load_fsdp2_api()
        dtype = _precision_dtype(self.precision)
        set_model_compute_dtype(model, dtype)
        self.mixed_precision_policy = mixed_precision_policy_cls(param_dtype=dtype, reduce_dtype=dtype)
        self.device_mesh = self._build_device_mesh(init_device_mesh, shard_size)
        del move_model_to_device
        for module in _modules_to_wrap(model, policy=self.wrap_policy, min_num_params=self.min_num_params, module_classes=self.module_classes):
            fully_shard(module, mesh=self.device_mesh, mp_policy=self.mixed_precision_policy)
        wrapped_model = fully_shard(model, mesh=self.device_mesh, mp_policy=self.mixed_precision_policy)
        self.model = wrapped_model if wrapped_model is not None else model
        self.optimizer = build_optimizer(self.model, optimizer)
        materialize_optimizer_state(self.optimizer)
        self.scheduler = build_scheduler(self.optimizer, scheduler) if scheduler is not None else None
        self._grad_scaler = _build_grad_scaler(self.precision, self.device)
        self.ema = ExponentialMovingAverage(self.model, ema_decay) if ema_decay is not None else None

    @property
    def is_main_process(self) -> bool:
        return self.distributed_context.is_main_process

    @property
    def checkpoint_on_all_ranks(self) -> bool:
        return True

    @property
    def optimizer_step_was_performed(self) -> bool:
        return self._optimizer_step_was_performed

    @property
    def optimizer_step_boundary_was_reached(self) -> bool:
        return self._optimizer_step_boundary_was_reached

    def train_mode(self) -> None:
        self.model.train()

    def zero_grad(self) -> None:
        if self._accumulation_index == 0:
            self.optimizer.zero_grad()

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.model(batch)

    def backward(self, loss: torch.Tensor) -> None:
        scaled_loss = loss / self.gradient_accumulation_steps
        if self._grad_scaler is not None:
            self._grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def step(self) -> None:
        is_boundary = self._accumulation_index + 1 >= self.gradient_accumulation_steps
        self._optimizer_step_boundary_was_reached = is_boundary
        self._optimizer_step_was_performed = False
        if not is_boundary:
            self._accumulation_index += 1
            return
        if self._grad_scaler is not None and self.max_grad_norm is not None:
            self._grad_scaler.unscale_(self.optimizer)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self._grad_scaler is not None:
            previous_scale = float(self._grad_scaler.get_scale())
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
            self._optimizer_step_was_performed = float(self._grad_scaler.get_scale()) >= previous_scale
        else:
            self.optimizer.step()
            self._optimizer_step_was_performed = True
        if self.scheduler is not None and self._optimizer_step_was_performed:
            self.scheduler.step()
        if self.ema is not None and self._optimizer_step_was_performed:
            self.ema.update()
        self._accumulation_index = 0

    def save_checkpoint(self, path: str | Path, *, step: int=0, extra: Mapping[str, Any] | None=None) -> Path:
        from sipai.engine.checkpoint import save_dcp_checkpoint
        return save_dcp_checkpoint(path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, step=step, extra=extra, distributed_context=self.distributed_context, checkpoint_backend='hsdp', ema=self.ema)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        from sipai.engine.checkpoint import load_dcp_checkpoint
        return load_dcp_checkpoint(path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, distributed_context=self.distributed_context, ema=self.ema)

    def _build_device_mesh(self, init_device_mesh: Callable[..., Any], shard_size: str | int) -> Any | None:
        context = self.distributed_context
        resolved_shard_size = context.local_world_size if str(shard_size).lower() == 'auto' else int(shard_size)
        if resolved_shard_size <= 0:
            raise ValueError(f'shard_size must be positive, got {resolved_shard_size}.')
        if context.world_size % resolved_shard_size != 0:
            raise ValueError(f'HSDP world_size ({context.world_size}) must be divisible by shard_size ({resolved_shard_size}).')
        if context.local_world_size % resolved_shard_size != 0:
            raise ValueError(f'HSDP local_world_size ({context.local_world_size}) must be divisible by shard_size ({resolved_shard_size}).')
        device_type = self.device.type if self.device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        return init_device_mesh(device_type, mesh_shape=(context.world_size // resolved_shard_size, resolved_shard_size), mesh_dim_names=('replicate', 'shard'))

def _load_fsdp2_api() -> tuple[Callable[..., Any], type[Any], Callable[..., Any]]:
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except (ImportError, AttributeError) as exc:
        raise ImportError('FSDPStrategy requires a PyTorch build with the FSDP2 fully_shard API.') from exc
    return (fully_shard, MixedPrecisionPolicy, init_device_mesh)

def _normalize_mode(mode: str) -> str:
    value = str(mode).lower()
    aliases = {'hsdp': 'hybrid_shard'}
    value = aliases.get(value, value)
    if value != 'hybrid_shard':
        raise ValueError(f"Unsupported FSDP mode: {mode}. Expected 'hybrid_shard'.")
    return value

def _normalize_precision(precision: str) -> str:
    value = str(precision).lower()
    aliases = {'float32': 'fp32', 'float16': 'fp16', 'bfloat16': 'bf16'}
    value = aliases.get(value, value)
    if value not in {'fp32', 'bf16', 'fp16'}:
        raise ValueError(f'Unsupported FSDP precision: {precision}. Expected fp32, bf16, or fp16.')
    return value

def _precision_dtype(precision: str) -> torch.dtype:
    return {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[precision]

def _normalize_wrap_policy(policy: str) -> str:
    value = str(policy).lower()
    if value not in {'root', 'size_based', 'module_classes'}:
        raise ValueError(f'Unsupported FSDP wrap_policy: {policy}. Expected root, size_based, or module_classes.')
    return value

def _normalize_module_classes(module_classes: Sequence[str | type[torch.nn.Module]] | str | type[torch.nn.Module] | None) -> tuple[str | type[torch.nn.Module], ...]:
    if module_classes is None:
        return ()
    if isinstance(module_classes, (str, type)):
        return (module_classes,)
    return tuple(module_classes)

def _modules_to_wrap(root: torch.nn.Module, *, policy: str, min_num_params: int, module_classes: tuple[str | type[torch.nn.Module], ...]) -> list[torch.nn.Module]:
    if policy == 'root':
        return []
    if policy == 'module_classes':
        matches = [(index, name, module) for (index, (name, module)) in enumerate(root.named_modules()) if name and _matches_module_class(module, module_classes)]
        matches.sort(key=lambda item: (-item[1].count('.'), item[0]))
        return [module for (_, _, module) in matches]
    selected: list[torch.nn.Module] = []

    def visit(module: torch.nn.Module, *, is_root: bool=False) -> bool:
        descendant_selected = False
        for child in module.children():
            descendant_selected = visit(child) or descendant_selected
        trainable_numel = sum((parameter.numel() for parameter in module.parameters() if parameter.requires_grad))
        if not is_root and trainable_numel >= min_num_params and (not descendant_selected):
            selected.append(module)
            return True
        return descendant_selected
    visit(root, is_root=True)
    return selected

def _matches_module_class(module: torch.nn.Module, module_classes: tuple[str | type[torch.nn.Module], ...]) -> bool:
    cls = type(module)
    qualified_name = f'{cls.__module__}.{cls.__qualname__}'
    for candidate in module_classes:
        if isinstance(candidate, str):
            if candidate in {cls.__name__, cls.__qualname__, qualified_name}:
                return True
        elif isinstance(module, candidate):
            return True
    return False

def _build_grad_scaler(precision: str, device: torch.device | None) -> Any | None:
    if precision != 'fp16' or (device is not None and device.type != 'cuda'):
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()

def _validate_materialized_parameters(model: torch.nn.Module) -> None:
    meta_parameters = [name for (name, parameter) in model.named_parameters() if parameter.is_meta]
    if meta_parameters:
        preview = ', '.join(meta_parameters[:4])
        raise ValueError(f'FSDPStrategy requires meta parameters to be materialized before fully_shard; load the initialization checkpoint with assign=True first. Remaining meta parameters: {preview}.')
__all__ = ['FSDPStrategy']
