from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model

class ExponentialMovingAverage:

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.model = model
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f'EMA decay must be between 0 and 1, got {decay!r}.')
        self.num_updates = 0
        self.shadow = {name: parameter.detach().to(dtype=torch.float32).clone() for (name, parameter) in _unwrap_model(model).named_parameters() if parameter.requires_grad}

    @torch.no_grad()
    def update(self) -> None:
        parameters = {name: parameter for (name, parameter) in _unwrap_model(self.model).named_parameters() if parameter.requires_grad}
        if parameters.keys() != self.shadow.keys():
            raise RuntimeError('Model parameters changed after EMA initialization.')
        one_minus_decay = 1.0 - self.decay
        for (name, parameter) in parameters.items():
            self.shadow[name].mul_(self.decay).add_(parameter.detach().to(dtype=torch.float32), alpha=one_minus_decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {'decay': torch.tensor(self.decay, dtype=torch.float64), 'num_updates': torch.tensor(self.num_updates, dtype=torch.int64), 'shadow': dict(self.shadow)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        saved_decay = float(torch.as_tensor(state_dict['decay']).item())
        if saved_decay != self.decay:
            raise ValueError(f'EMA decay mismatch: checkpoint={saved_decay}, configured={self.decay}.')
        saved_shadow = state_dict.get('shadow')
        if not isinstance(saved_shadow, Mapping) or saved_shadow.keys() != self.shadow.keys():
            raise ValueError('EMA checkpoint parameters do not match the model.')
        with torch.no_grad():
            for (name, value) in saved_shadow.items():
                target = self.shadow[name]
                if value.shape != target.shape:
                    raise ValueError(f'EMA shape mismatch for {name!r}: checkpoint={tuple(value.shape)}, model={tuple(target.shape)}.')
                target.copy_(value)
        self.num_updates = int(torch.as_tensor(state_dict['num_updates']).item())

    def model_state_dict(self) -> dict[str, Any]:
        state = dict(_unwrap_model(self.model).state_dict())
        missing = sorted(set(self.shadow) - set(state))
        if missing:
            raise RuntimeError(f'EMA parameters are absent from the model state_dict: {missing[:4]}.')
        state.update(self.shadow)
        return state
__all__ = ['ExponentialMovingAverage']
