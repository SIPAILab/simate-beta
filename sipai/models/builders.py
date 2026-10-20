from __future__ import annotations
import importlib
import inspect
from collections.abc import Mapping
from typing import Any
from sipai.config import normalize_sipai_config
from sipai.registry import ACTION_HEADS, BACKBONES, FRAMEWORKS, MODELS, build_from_cfg
_REGISTRY_IMPORTS = {'model': {'ComposableVLAFramework': 'sipai.models.frameworks.vla.composable_vla'}, 'framework': {'ComposableVLAFramework': 'sipai.models.frameworks.vla.composable_vla'}, 'backbone': {'SimateBetaPrefixBackbone': 'sipai.models.backbones.simate_beta.prefix'}, 'action_head': {'SimateBetaFlowMatchingActionHead': 'sipai.models.action_heads.simate_beta'}}

def _ensure_registered(registry, cfg):
    kind = cfg if isinstance(cfg, str) else cfg.get('type')
    if isinstance(kind, str) and kind not in registry:
        module = _REGISTRY_IMPORTS.get(registry.name, {}).get(kind)
        if module is not None:
            importlib.import_module(module)

def _build(registry, cfg, *args, **kwargs):
    _ensure_registered(registry, cfg)
    return build_from_cfg(cfg, registry, *args, **kwargs)

def build_model(cfg, *args, **kwargs):
    return _build(MODELS, cfg, *args, **kwargs)

def build_framework(cfg, *args, **kwargs):
    return _build(FRAMEWORKS, cfg, *args, **kwargs)

def build_backbone(cfg, *args, **kwargs):
    return _build(BACKBONES, cfg, *args, **kwargs)

def build_action_head(cfg, *args, **kwargs):
    return _build(ACTION_HEADS, cfg, *args, **kwargs)

def _filtered(registry, node, cfg=None):
    raw = dict(node)
    _ensure_registered(registry, raw)
    signature = inspect.signature(registry.get(raw['type']).__init__)
    accepts_extra = any((p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()))
    result = {k: v for (k, v) in raw.items() if k == 'type' or accepts_extra or k in signature.parameters}
    if cfg is not None and (accepts_extra or 'cfg' in signature.parameters):
        result['cfg'] = cfg
    if result.get('input_cameras', object()) is None:
        result.pop('input_cameras')
    return result

def build_model_from_sipai_config(cfg: Any):
    cfg = normalize_sipai_config(cfg)
    framework = cfg.framework
    if framework.name != 'ComposableVLAFramework':
        raise ValueError('This release supports only ComposableVLAFramework.')
    backbone = build_backbone(_filtered(BACKBONES, framework.backbone, cfg))
    head = build_action_head(_filtered(ACTION_HEADS, framework.action_model))
    return build_framework({'type': framework.name, 'backbone': backbone, 'action_head': head})
