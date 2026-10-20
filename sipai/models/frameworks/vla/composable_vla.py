from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
from sipai.data.schema import CanonicalBatch
from sipai.models.builders import build_action_head, build_backbone
from sipai.models.frameworks.base_framework import SiPAIFramework
from sipai.registry import FRAMEWORKS, MODELS

@FRAMEWORKS.register_module(name='ComposableVLAFramework', force=True)
@MODELS.register_module(name='ComposableVLAFramework', force=True)
class ComposableVLAFramework(SiPAIFramework):

    def __init__(self, *, backbone: Any | None=None, action_head: Any | None=None, backbone_cfg: Mapping[str, Any] | str | None=None, action_head_cfg: Mapping[str, Any] | str | None=None) -> None:
        super().__init__()
        if backbone is None and backbone_cfg is None:
            raise ValueError('ComposableVLAFramework requires `backbone` or `backbone_cfg`.')
        if action_head is None and action_head_cfg is None:
            raise ValueError('ComposableVLAFramework requires `action_head` or `action_head_cfg`.')
        self.backbone = backbone if backbone is not None else build_backbone(backbone_cfg)
        self.action_head = action_head if action_head is not None else build_action_head(action_head_cfg)

    def encode_inputs(self, batch: CanonicalBatch | Sequence[dict[str, Any]], **kwargs: Any) -> Any:
        if hasattr(self.backbone, 'encode'):
            return self.backbone.encode(batch, **kwargs)
        return self.backbone(batch, **kwargs)

    def set_compute_dtype(self, dtype: Any) -> None:
        for component in (self.backbone, self.action_head):
            setter = getattr(component, 'set_compute_dtype', None)
            if callable(setter):
                setter(dtype)

    def forward(self, batch: CanonicalBatch | Sequence[dict[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
        encoded = self.encode_inputs(batch, **kwargs)
        if hasattr(self.action_head, 'compute_loss'):
            return self.action_head.compute_loss(encoded, batch=batch, **kwargs)
        return self.action_head(encoded, batch=batch, **kwargs)

    def predict_action(self, batch: CanonicalBatch | Sequence[dict[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
        encoded = self.encode_inputs(batch, mode='predict', **kwargs)
        if hasattr(self.action_head, 'predict_action'):
            return self.action_head.predict_action(encoded, batch=batch, **kwargs)
        if hasattr(self.action_head, 'predict'):
            return self.action_head.predict(encoded, batch=batch, **kwargs)
        raise NotImplementedError(f'{type(self.action_head).__name__} must implement predict_action or predict.')

    def remap_state_dict_keys(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        if self.module_name_mapping:
            return super().remap_state_dict_keys(state_dict)
        return dict(state_dict)
