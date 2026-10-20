from __future__ import annotations
import math
from typing import Any, Mapping
import torch

def materialize_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    if isinstance(optimizer, torch.optim.LBFGS):
        raise RuntimeError('materialize_optimizer_state does not support LBFGS')
    saved_lrs = [group.get('lr') for group in optimizer.param_groups]
    saved_grads: dict[torch.Tensor, torch.Tensor | None] = {}
    parameters: list[torch.Tensor] = []
    for group in optimizer.param_groups:
        group['lr'] = 0.0
        for parameter in group.get('params', []):
            if parameter is None:
                continue
            parameters.append(parameter)
            saved_grads[parameter] = parameter.grad
            if parameter.grad is None and (not parameter.is_meta):
                try:
                    parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                except (RuntimeError, TypeError):
                    parameter.grad = parameter.detach().new_zeros(parameter.shape)
    try:
        with torch.no_grad():
            optimizer.step()
    finally:
        for (group, lr) in zip(optimizer.param_groups, saved_lrs, strict=True):
            if lr is None:
                group.pop('lr', None)
            else:
                group['lr'] = lr
        for parameter in parameters:
            parameter.grad = saved_grads[parameter]
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        step = state.get('step')
        if torch.is_tensor(step):
            step.zero_()
        elif step is not None:
            state['step'] = 0

def build_optimizer(model: torch.nn.Module, cfg: Mapping[str, Any] | None=None) -> torch.optim.Optimizer:
    cfg = dict(cfg or {})
    optim_type = str(cfg.pop('type', 'AdamW')).lower()
    lr = float(cfg.pop('lr', 0.0001))
    weight_decay = float(cfg.pop('weight_decay', 0.0))
    paramwise_lrs = dict(cfg.pop('paramwise_lrs', {}) or {})
    params = _build_param_groups(model, base_lr=lr, paramwise_lrs=paramwise_lrs)
    if optim_type == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **cfg)
    if optim_type == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **cfg)
    raise ValueError(f'Unsupported optimizer type: {optim_type}')

def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any] | None=None) -> Any | None:
    if cfg is None:
        return None
    cfg = dict(cfg)
    scheduler_type = str(cfg.pop('type', 'StepLR')).lower()
    if scheduler_type == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, **cfg)
    if scheduler_type == 'constant':
        return None
    if scheduler_type in {'cosine_with_min_lr', 'cosinewithminlrscheduler'}:
        num_training_steps = int(cfg.pop('num_training_steps'))
        num_warmup_steps = int(cfg.pop('num_warmup_steps', 0))
        min_lr = float(cfg.pop('min_lr', 0.0))
        if cfg:
            unknown = ', '.join(sorted(cfg))
            raise ValueError(f'Unsupported CosineWithMinLRScheduler arguments: {unknown}')
        return CosineWithMinLRScheduler(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps, min_lr=min_lr)
    raise ValueError(f'Unsupported scheduler type: {scheduler_type}')

def _build_param_groups(model: torch.nn.Module, *, base_lr: float, paramwise_lrs: Mapping[str, float]) -> list[dict[str, Any]]:
    if not paramwise_lrs:
        return [{'params': [param for param in model.parameters() if param.requires_grad], 'lr': base_lr}]
    groups: list[dict[str, Any]] = []
    assigned: set[int] = set()
    named_params = list(model.named_parameters())
    for (module_name, module_lr) in paramwise_lrs.items():
        params = [param for (name, param) in named_params if param.requires_grad and name.startswith(f'{module_name}.')]
        if not params:
            continue
        groups.append({'params': params, 'lr': float(module_lr), 'name': module_name})
        assigned.update((id(param) for param in params))
    default_params = [param for (_, param) in named_params if param.requires_grad and id(param) not in assigned]
    if default_params:
        groups.insert(0, {'params': default_params, 'lr': base_lr, 'name': 'default'})
    return groups

class CosineWithMinLRScheduler(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer: torch.optim.Optimizer, *, num_training_steps: int, num_warmup_steps: int, min_lr: float) -> None:
        if num_training_steps <= 0:
            raise ValueError('num_training_steps must be positive for CosineWithMinLRScheduler.')

        def make_lambda(base_lr: float):
            min_ratio = min(max(min_lr / base_lr, 0.0), 1.0) if base_lr > 0 else 0.0

            def lr_lambda(current_step: int) -> float:
                if num_warmup_steps > 0 and current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine
            return lr_lambda
        super().__init__(optimizer, [make_lambda(float(group['lr'])) for group in optimizer.param_groups])
