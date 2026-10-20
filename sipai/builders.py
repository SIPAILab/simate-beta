from __future__ import annotations
import importlib
from collections.abc import Mapping
from typing import Any
from sipai.registry import DATALOADERS, DATASETS, MODELS, RUNNERS, build_from_cfg
_MODEL_IMPORTS = {'ComposableVLAFramework': 'sipai.models.frameworks.vla.composable_vla'}

def _cfg_type(cfg: Mapping[str, Any] | str) -> Any:
    if isinstance(cfg, str):
        return cfg
    if isinstance(cfg, Mapping):
        return cfg.get('type')
    return None

def _ensure_model_registered(cfg: Mapping[str, Any] | str) -> None:
    obj_type = _cfg_type(cfg)
    if not isinstance(obj_type, str) or obj_type in MODELS:
        return
    module_name = _MODEL_IMPORTS.get(obj_type)
    if module_name is not None:
        importlib.import_module(module_name)

def build_dataset(cfg: Mapping[str, Any] | str, *args: Any, **kwargs: Any) -> Any:
    return build_from_cfg(cfg, DATASETS, *args, **kwargs)

def build_dataloader(cfg: Mapping[str, Any] | str, *args: Any, **kwargs: Any) -> Any:
    return build_from_cfg(cfg, DATALOADERS, *args, **kwargs)

def build_model(cfg: Mapping[str, Any] | str, *args: Any, **kwargs: Any) -> Any:
    _ensure_model_registered(cfg)
    return build_from_cfg(cfg, MODELS, *args, **kwargs)

def build_runner(cfg: Mapping[str, Any] | str, *args: Any, **kwargs: Any) -> Any:
    return build_from_cfg(cfg, RUNNERS, *args, **kwargs)
