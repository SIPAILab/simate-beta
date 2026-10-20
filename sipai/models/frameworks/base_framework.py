from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from sipai.data.schema import CanonicalBatch

class SiPAIFramework(torch.nn.Module):
    module_name_mapping: Mapping[str, str] = {}

    def forward(self, batch: CanonicalBatch, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError(f'{type(self).__name__} must implement forward(batch: CanonicalBatch).')

    def predict_action(self, batch: CanonicalBatch, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError(f'{type(self).__name__} must implement predict_action(batch: CanonicalBatch).')

    def supports_training_tag(self, tag: str) -> bool:
        return tag == 'vla'

    def freeze(self, trainable_patterns: list[str] | tuple[str, ...] | None=None) -> None:
        if trainable_patterns is None:
            for parameter in self.parameters():
                parameter.requires_grad = False
            return
        patterns = tuple(trainable_patterns)
        for (name, parameter) in self.named_parameters():
            parameter.requires_grad = any((pattern in name for pattern in patterns))

    def parameter_summary(self) -> dict[str, int]:
        total = 0
        trainable = 0
        for parameter in self.parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
        return {'total': total, 'trainable': trainable}

    def trainable_parameter_names(self) -> list[str]:
        return [name for (name, parameter) in self.named_parameters() if parameter.requires_grad]

    def save(self, path: str | Path, **metadata: Any) -> None:
        payload = {'state_dict': self.state_dict(), 'metadata': metadata}
        torch.save(payload, path)

    def load(self, path: str | Path, *, strict: bool=True, map_location: str | torch.device='cpu') -> Any:
        payload = torch.load(path, map_location=map_location)
        state_dict = payload.get('state_dict', payload) if isinstance(payload, Mapping) else payload
        return self.load_state_dict(state_dict, strict=strict)

    def remap_state_dict_keys(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        if not self.module_name_mapping:
            return dict(state_dict)
        remapped: dict[str, Any] = {}
        for (key, value) in state_dict.items():
            new_key = key
            for (old_prefix, new_prefix) in self.module_name_mapping.items():
                if key == old_prefix or key.startswith(f'{old_prefix}.'):
                    new_key = f'{new_prefix}{key[len(old_prefix):]}'
                    break
            remapped[new_key] = value
        return remapped

class BaseFramework(SiPAIFramework):
    pass
