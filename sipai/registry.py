from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, TypeVar
T = TypeVar('T')

class Registry:

    def __init__(self, name: str):
        self.name = name
        self._module_dict: dict[str, Any] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __repr__(self) -> str:
        return f'Registry(name={self.name!r}, items={list(self._module_dict)})'

    @property
    def module_dict(self) -> dict[str, Any]:
        return dict(self._module_dict)

    def get(self, key: str) -> Any:
        if key not in self._module_dict:
            available = ', '.join(sorted(self._module_dict)) or '<empty>'
            raise KeyError(f'{key!r} is not registered in {self.name}. Available: {available}')
        return self._module_dict[key]

    def register_module(self, module: T | None=None, *, name: str | None=None, force: bool=False):

        def _register(obj: T) -> T:
            module_name = name or getattr(obj, '__name__', None)
            if not module_name:
                raise ValueError(f'Cannot infer registry name for object {obj!r}')
            if not force and module_name in self._module_dict:
                raise KeyError(f'{module_name!r} is already registered in {self.name}')
            self._module_dict[module_name] = obj
            return obj
        if module is None:
            return _register
        return _register(module)

    def build(self, cfg: Mapping[str, Any] | str, *args: Any, **kwargs: Any) -> Any:
        return build_from_cfg(cfg, self, *args, **kwargs)

def _to_plain_dict(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if hasattr(cfg, 'items'):
        return {k: v for (k, v) in cfg.items()}
    raise TypeError(f'Expected mapping config, got {type(cfg)!r}')

def build_from_cfg(cfg: Mapping[str, Any] | str, registry: Registry, *args: Any, default_args: Mapping[str, Any] | None=None, **kwargs: Any) -> Any:
    if isinstance(cfg, str):
        cfg_dict: dict[str, Any] = {'type': cfg}
    elif isinstance(cfg, Mapping):
        cfg_dict = _to_plain_dict(cfg)
    else:
        raise TypeError(f'cfg must be a mapping or string, got {type(cfg)!r}')
    obj_type = cfg_dict.pop('type', None)
    if obj_type is None:
        raise KeyError(f"Missing required key 'type' for registry {registry.name}")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"cfg['type'] must be a string or callable, got {type(obj_type)!r}")
    init_kwargs = {}
    if default_args:
        init_kwargs.update(default_args)
    init_kwargs.update(cfg_dict)
    init_kwargs.update(kwargs)
    if isinstance(obj_cls, type):
        return obj_cls(*args, **init_kwargs)
    if isinstance(obj_cls, Callable):
        return obj_cls(*args, **init_kwargs)
    raise TypeError(f'Registered object {obj_cls!r} in {registry.name} is not callable')
DATASETS = Registry('dataset')
READERS = Registry('reader')
SAMPLERS = Registry('sampler')
COLLATORS = Registry('collator')
DATALOADERS = Registry('dataloader')
MODELS = Registry('model')
CHECKPOINT_LOADERS = Registry('checkpoint_loader')
FRAMEWORKS = Registry('framework')
BACKBONES = Registry('backbone')
ACTION_HEADS = Registry('action_head')
RUNNERS = Registry('runner')
TRAINERS = Registry('trainer')
