from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any, Mapping
from omegaconf import OmegaConf
SIPAI_CONFIG_SCHEMA_VERSION = 'v0.2'
PUBLIC_CONFIG_REQUIRED_SECTIONS = ('experiment', 'model', 'data', 'train_dataloader', 'optim_wrapper', 'train_cfg')
RUNTIME_CONFIG_SECTIONS = ('framework', 'datasets', 'trainer')
SUPPORTED_DATALOADER_OUTPUT_FORMATS = ('canonical',)
SUPPORTED_TRAINING_PRECISIONS = ('fp32', 'bf16', 'fp16')
SUPPORTED_EXPORT_FINAL_WEIGHTS = ('ema', 'raw')
TRAINING_STRATEGY_PRESETS = {'fsdp2_hsdp': {'type': 'FSDPStrategy', 'mode': 'hybrid_shard'}}
LEGACY_TOP_LEVEL_FIELDS = ('run_id', 'run_root_dir', 'seed', 'max_steps', 'num_warmup_steps', 'save_interval', 'eval_interval', 'max_grad_norm', 'gradient_clipping', 'gradient_accumulation_steps', 'logging_frequency')
PUBLIC_ACTION_HEAD_FIELDS = {'type', 'variant', 'action_dim', 'state_dim', 'horizon', 'action_horizon'}
PUBLIC_ACTION_HEAD_FIELDS_BY_TYPE = {'SimateBetaFlowMatchingActionHead': PUBLIC_ACTION_HEAD_FIELDS | {'transformer_variant', 'num_inference_steps', 'action_normalization', 'action_loss_weight', 'counterfactual_pair_flow', 'dtype', 'gradient_checkpointing'}}

class ConfigValidationError(ValueError):
    pass

def load_sipai_config_file(config_path: str | Path, overrides: list[str] | None=None) -> Any:
    path = Path(config_path).expanduser().resolve(strict=False)
    cfg = _load_inherited_config(path, stack=())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg

def _load_inherited_config(path: Path, *, stack: tuple[Path, ...]) -> Any:
    if path in stack:
        cycle = ' -> '.join((str(item) for item in (*stack, path)))
        raise ConfigValidationError(f'Config inheritance cycle detected: {cycle}')
    if not path.is_file():
        raise FileNotFoundError(f'SiPAI config does not exist: {path}')
    loaded = OmegaConf.load(path)
    if not OmegaConf.is_dict(loaded):
        raise ConfigValidationError(f'SiPAI config root must be a mapping: {path}')
    authored = OmegaConf.to_container(loaded, resolve=False)
    if not isinstance(authored, dict):
        raise ConfigValidationError(f'SiPAI config root must be a mapping: {path}')
    raw_bases = authored.pop('_base_', None)
    if raw_bases is None:
        base_refs: list[str] = []
    elif isinstance(raw_bases, str):
        base_refs = [raw_bases]
    elif isinstance(raw_bases, list) and all((isinstance(item, str) for item in raw_bases)):
        base_refs = list(raw_bases)
    else:
        raise ConfigValidationError(f'{path}: _base_ must be a path string or a list of path strings')
    if any((not item.strip() for item in base_refs)):
        raise ConfigValidationError(f'{path}: _base_ paths must be non-empty')
    merged = OmegaConf.create({})
    next_stack = (*stack, path)
    for reference in base_refs:
        base_path = Path(reference).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_path = base_path.resolve(strict=False)
        merged = OmegaConf.merge(merged, _load_inherited_config(base_path, stack=next_stack))
    base_value = OmegaConf.to_container(merged, resolve=False)
    if not isinstance(base_value, dict):
        raise ConfigValidationError(f'Inherited SiPAI config root must be a mapping: {path}')
    return OmegaConf.create(_merge_authored_config(base_value, authored))

def _merge_authored_config(base: Mapping[str, Any], authored: Mapping[str, Any]) -> dict[str, Any]:
    if authored.get('_replace_') is True:
        return {str(key): copy.deepcopy(value) for (key, value) in authored.items() if key != '_replace_'}
    if '_replace_' in authored:
        raise ConfigValidationError('Config mapping `_replace_` must be exactly true when present')
    result = copy.deepcopy(dict(base))
    for (key, value) in authored.items():
        inherited = result.get(key)
        if isinstance(value, Mapping) and isinstance(inherited, Mapping):
            result[key] = _merge_authored_config(inherited, value)
        elif isinstance(value, Mapping):
            result[key] = _merge_authored_config({}, value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def get_config_value(cfg: Any, key: str, default: Any=None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def get_nested_config(cfg: Any, path: str, default: Any=None) -> Any:
    cur = cfg
    for part in path.split('.'):
        cur = get_config_value(cur, part, None)
        if cur is None:
            return default
    return cur

def _plain(node: Any) -> Any:
    if node is None:
        return None
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    if isinstance(node, Mapping):
        return {key: _plain(value) for (key, value) in node.items()}
    return node

def _pop_any(data: dict[str, Any], *keys: str, default: Any=None) -> Any:
    for key in keys:
        if key in data:
            return data.pop(key)
    return default

def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _normalize_data_sources(data_train: dict[str, Any], manifest_cfg: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    sources = data_train.pop('sources', None)
    data_roots = data_train.pop('data_roots', None)
    catalog = data_train.pop('catalog', None)
    mixture = data_train.pop('mixture', None)
    sampling = data_train.pop('sampling', None)
    configured_inputs = sum((value is not None for value in (sources, data_roots, catalog)))
    if configured_inputs > 1:
        raise ConfigValidationError('Configure only one of `data.train.catalog`, `data.train.sources`, or legacy `data.train.data_roots`.')
    split = manifest_cfg.get('split', data_train.get('manifest_split', None))
    if catalog is not None:
        raise ConfigValidationError('This release uses explicit data.train.sources, not catalogs.')
    if sources is None and data_roots is not None:
        sources = [{'name': str(root).rstrip('/').split('/')[-1] or f'source_{index}', 'root': root, 'format': 'simate_unified', 'split': split, 'ratio': 1.0, 'sampling_weight': 1.0, 'weight': 1.0} for (index, root) in enumerate(_as_list(data_roots))]
    if sources is None:
        return None
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for (index, raw_source) in enumerate(_as_list(sources)):
        if isinstance(raw_source, str):
            source = {'name': raw_source.rstrip('/').split('/')[-1] or f'source_{index}', 'root': raw_source}
        elif isinstance(raw_source, Mapping):
            source = dict(_plain(raw_source) or {})
        else:
            raise ConfigValidationError(f'data.train.sources[{index}] must be a mapping or string.')
        source_root = source.get('root')
        if not source_root:
            raise ConfigValidationError(f'data.train.sources[{index}] must define `root`.')
        name = str(source.get('name') or str(source_root).rstrip('/').split('/')[-1] or f'source_{index}')
        if name in names:
            raise ConfigValidationError(f'Duplicate data.train.sources name: {name!r}.')
        names.add(name)
        fmt = str(source.get('format', 'simate_unified')).lower()
        if fmt != 'simate_unified':
            raise ConfigValidationError(f"Unsupported data.train.sources[{index}].format={fmt!r}; expected 'simate_unified'.")
        raw_weight = source.get('sampling_weight', source.get('ratio', source.get('weight', 1.0)))
        weight = float(raw_weight)
        if weight <= 0:
            raise ConfigValidationError(f'data.train.sources[{index}] ratio/weight must be > 0, got {weight}.')
        normalized_source = dict(source)
        normalized_source.update({'name': name, 'root': source_root, 'format': fmt, 'ratio': float(source.get('ratio', weight)), 'sampling_weight': weight, 'weight': weight})
        normalized_source.setdefault('split', split)
        normalized.append(normalized_source)
    return normalized

def _normalized_precision(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    precision = str(value).lower()
    if precision not in SUPPORTED_TRAINING_PRECISIONS:
        supported = ', '.join(SUPPORTED_TRAINING_PRECISIONS)
        raise ConfigValidationError(f'Unsupported {path}={value!r}; expected one of: {supported}.')
    return precision

def _normalize_engine_cfg(cfg: Any) -> dict[str, Any] | None:
    raw_engine = get_config_value(cfg, 'engine', None)
    if raw_engine is None:
        return None
    engine = dict(_plain(raw_engine) or {})
    engine_precision = _normalized_precision(engine.get('precision'), path='engine.precision')
    if engine_precision is not None:
        engine['precision'] = engine_precision
    raw_strategy = engine.get('strategy')
    if raw_strategy is None:
        if engine_precision is not None:
            engine['strategy'] = {'type': 'FSDPStrategy', 'mode': 'hybrid_shard', 'precision': engine_precision}
        return engine
    if not isinstance(raw_strategy, Mapping):
        raise ConfigValidationError('`engine.strategy` must be a mapping.')
    strategy = dict(raw_strategy)
    preset = strategy.pop('preset', None)
    raw_options = strategy.pop('options', None)
    if raw_options is None:
        options: dict[str, Any] = {}
    elif isinstance(raw_options, Mapping):
        options = dict(raw_options)
    else:
        raise ConfigValidationError('`engine.strategy.options` must be a mapping.')
    for key in strategy.keys() & options.keys():
        if strategy[key] != options[key]:
            raise ConfigValidationError(f'Conflicting engine.strategy.{key} and engine.strategy.options.{key} values.')
    expanded: dict[str, Any] = {}
    if preset is not None:
        preset_name = str(preset).lower()
        if preset_name not in TRAINING_STRATEGY_PRESETS:
            supported = ', '.join(TRAINING_STRATEGY_PRESETS)
            raise ConfigValidationError(f'Unsupported engine.strategy.preset={preset!r}; expected one of: {supported}.')
        preset_cfg = TRAINING_STRATEGY_PRESETS[preset_name]
        identity_keys = {'type', 'zero_stage', 'mode'}
        for key in identity_keys & preset_cfg.keys():
            for (source_name, source) in (('engine.strategy', strategy), ('engine.strategy.options', options)):
                if key in source and source[key] != preset_cfg[key]:
                    raise ConfigValidationError(f'{source_name}.{key}={source[key]!r} conflicts with preset {preset_name!r}.')
        expanded.update(preset_cfg)
    expanded.update(strategy)
    expanded.update(options)
    strategy_precision = _normalized_precision(expanded.get('precision'), path='engine.strategy.precision')
    if engine_precision is not None and strategy_precision is not None and (engine_precision != strategy_precision):
        raise ConfigValidationError('engine.precision conflicts with engine.strategy precision; precision must be configured independently at `engine.precision`.')
    if engine_precision is not None:
        expanded['precision'] = engine_precision
    elif strategy_precision is not None:
        expanded['precision'] = strategy_precision
    engine['strategy'] = expanded
    return engine

def _action_horizon_value(action_model_cfg: Mapping[str, Any]) -> int | None:
    value = action_model_cfg.get('action_horizon')
    if value is None:
        value = action_model_cfg.get('future_action_window_size')
        if value is not None:
            return int(value) + 1
    return int(value) if value is not None else None

def _apply_action_head_defaults(action_model_cfg: dict[str, Any]) -> None:
    return None

def is_normalized_runtime_config(cfg: Any) -> bool:
    return all((get_config_value(cfg, section, None) is not None for section in RUNTIME_CONFIG_SECTIONS))

def is_public_sipai_config(cfg: Any) -> bool:
    return get_config_value(cfg, 'model', None) is not None and (not is_normalized_runtime_config(cfg))

def _require_section(cfg: Any, section: str) -> Any:
    value = get_config_value(cfg, section, None)
    if value is None:
        raise ConfigValidationError(f'SiPAI config is missing required top-level section `{section}`.')
    return value

def _require_nested(cfg: Any, path: str) -> Any:
    value = get_nested_config(cfg, path, None)
    if value is None:
        raise ConfigValidationError(f'SiPAI config is missing required field `{path}`.')
    return value

def _reject_legacy_simate_beta_component_names(cfg: Any, *, backbone_type_path: str, action_head_type_path: str) -> None:
    migrations = ((backbone_type_path, 'SimateBetaBackbone', 'SimateBetaPrefixBackbone'), (action_head_type_path, 'SimateBetaActionHead', 'SimateBetaFlowMatchingActionHead'))
    for (path, legacy_name, canonical_name) in migrations:
        if get_nested_config(cfg, path, None) == legacy_name:
            raise ConfigValidationError(f'SiPAI config no longer supports `{path}: {legacy_name}`; use `{canonical_name}`.')

def _uses_state_action_representation(state_cfg: Any, sampling_cfg: Any, action_cfg: Any) -> bool:
    if state_cfg is not None:
        return True
    if isinstance(sampling_cfg, Mapping) and 'anchor_stride' in sampling_cfg:
        return True
    return _has_action_representation_markers(action_cfg)

def _has_action_representation_markers(action_cfg: Any) -> bool:
    if not isinstance(action_cfg, Mapping):
        return False
    return any((key in action_cfg for key in ('fields', 'representation', 'rotation', 'target_start_offset', 'target_step_interval', 'tail_padding')))

def _select_train_action_config(data_train: Mapping[str, Any], pipeline: Mapping[str, Any]) -> Any:
    has_legacy_action = 'action' in data_train
    has_pipeline_action = 'action' in pipeline
    if has_legacy_action and has_pipeline_action:
        raise ConfigValidationError('`data.train.pipeline.action` cannot be combined with legacy `data.train.action`; keep exactly one action configuration source.')
    if has_legacy_action:
        legacy_action = data_train.get('action')
        if _has_action_representation_markers(legacy_action):
            raise ConfigValidationError('`data.train.action` is the legacy action owner and cannot contain new representation fields; move this action block to `data.train.pipeline.action`.')
        state_selects_new_path = pipeline.get('state') is not None
        sampling = pipeline.get('sampling')
        sampling_selects_new_path = isinstance(sampling, Mapping) and 'anchor_stride' in sampling
        if state_selects_new_path or sampling_selects_new_path:
            raise ConfigValidationError('New state/action representation config cannot be combined with legacy `data.train.action`; move the action block to `data.train.pipeline.action` or remove the new pipeline markers.')
        return legacy_action
    return pipeline.get('action') if has_pipeline_action else None

def _positive_config_int(value: Any, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigValidationError(f'`{path}` must be an integer >= 1, got {value!r}.')
    return int(value)

def _validate_history_offsets(value: Any, *, path: str) -> list[int]:
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 1 or isinstance(value[0], bool) or (not isinstance(value[0], int)) or (value[0] >= 0):
        raise ConfigValidationError(f'`{path}` must contain exactly one negative integer, got {value!r}.')
    return [int(value[0])]

def _validate_pose_delta_axes(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or value not in {'parent', 'anchor', 'parent_position_anchor_rotation'}:
        raise ConfigValidationError(f'`{path}` must be one of: parent, anchor, parent_position_anchor_rotation; got {value!r}.')
    return value

def _validate_export_final_weights(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_EXPORT_FINAL_WEIGHTS:
        supported = ', '.join(SUPPORTED_EXPORT_FINAL_WEIGHTS)
        raise ConfigValidationError(f'`{path}` must be one of: {supported}; got {value!r}.')
    return value

def _validate_representation_fields(fields: Any, *, path: str, role: str, stored_typed: bool=False, allow_numeric_action_select: bool=False) -> None:
    fields = _plain(fields)
    if not isinstance(fields, (list, tuple)) or not fields:
        raise ConfigValidationError(f'`{path}` must be a non-empty ordered list.')
    names: set[str] = set()
    for (index, raw_field) in enumerate(fields):
        field_path = f'{path}[{index}]'
        if not isinstance(raw_field, Mapping):
            raise ConfigValidationError(f'`{field_path}` must be a mapping.')
        allowed_field_keys = {'name', 'source', 'kind', 'metadata', 'select'}
        if stored_typed:
            allowed_field_keys.add('anchor_source')
        unknown = sorted(set(raw_field) - allowed_field_keys)
        if unknown:
            raise ConfigValidationError(f"`{field_path}` has unsupported key(s): {', '.join(unknown)}.")
        name = raw_field.get('name')
        source = raw_field.get('source')
        kind = raw_field.get('kind')
        if not isinstance(name, str) or not name.strip():
            raise ConfigValidationError(f'`{field_path}.name` must be a non-empty string.')
        if name in names:
            raise ConfigValidationError(f'`{path}` contains duplicate field name {name!r}.')
        names.add(name)
        if not isinstance(source, str) or not source.strip():
            raise ConfigValidationError(f'`{field_path}.source` must be a non-empty string.')
        if kind not in {'pose', 'joint', 'gripper'}:
            raise ConfigValidationError(f'`{field_path}.kind` must be pose, joint, or gripper, got {kind!r}.')
        if 'select' in raw_field:
            selector = raw_field.get('select')
            if stored_typed:
                if role != 'action':
                    raise ConfigValidationError(f'`{field_path}.select` typed layout is action-only.')
                if not isinstance(selector, Mapping):
                    raise ConfigValidationError(f'`{field_path}.select` must be a mapping.')
                selector_keys = set(selector)
                if selector_keys == {'layout_fields'}:
                    layout_fields = selector.get('layout_fields')
                    if not isinstance(layout_fields, (list, tuple)) or not layout_fields or any((not isinstance(item, str) or not item.strip() for item in layout_fields)):
                        raise ConfigValidationError(f'`{field_path}.select.layout_fields` must be a non-empty string list.')
                elif selector_keys == {'start', 'stop'}:
                    start = selector.get('start')
                    stop = selector.get('stop')
                    if isinstance(start, bool) or not isinstance(start, int) or isinstance(stop, bool) or (not isinstance(stop, int)) or (start < 0) or (stop < 0) or (start >= stop):
                        raise ConfigValidationError(f'`{field_path}.select` requires integer bounds satisfying 0 <= start < stop.')
                else:
                    raise ConfigValidationError(f'`{field_path}.select` requires exactly `layout_fields` or `start` and `stop`.')
                if source != 'values':
                    raise ConfigValidationError(f'`{field_path}.source` must be `values` for typed stored actions.')
            elif role != 'state' and (not (role == 'action' and allow_numeric_action_select)):
                raise ConfigValidationError(f'`{field_path}.select` is only supported for state fields.')
            elif not isinstance(selector, Mapping):
                raise ConfigValidationError(f'`{field_path}.select` must be a mapping.')
            elif sorted(set(selector) - {'start', 'stop'}):
                selector_unknown = sorted(set(selector) - {'start', 'stop'})
                raise ConfigValidationError(f"`{field_path}.select` has unsupported key(s): {', '.join(selector_unknown)}.")
            elif set(selector) != {'start', 'stop'}:
                raise ConfigValidationError(f'`{field_path}.select` requires exactly `start` and `stop`.')
            else:
                start = selector.get('start')
                stop = selector.get('stop')
                if isinstance(start, bool) or not isinstance(start, int) or isinstance(stop, bool) or (not isinstance(stop, int)) or (start < 0) or (stop < 0) or (start >= stop):
                    raise ConfigValidationError(f'`{field_path}.select` requires integer bounds satisfying 0 <= start < stop.')
        elif stored_typed:
            raise ConfigValidationError(f'`{field_path}.select` is required for typed stored actions.')
        metadata = raw_field.get('metadata')
        if metadata is not None and (not isinstance(metadata, Mapping)):
            raise ConfigValidationError(f'`{field_path}.metadata` must be a mapping.')

def _validate_rotation_config(value: Any, *, path: str, has_pose: bool) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f'`{path}` must be a mapping.')
    unknown = sorted(set(value) - {'encoding'})
    if unknown:
        raise ConfigValidationError(f"`{path}` has unsupported key(s): {', '.join(unknown)}.")
    encoding = value.get('encoding', 'quat_xyzw')
    if encoding not in {'quat_xyzw', 'quat_wxyz', 'rotvec', 'rot6d'}:
        raise ConfigValidationError(f'Unsupported `{path}.encoding`={encoding!r}.')
    if not has_pose:
        raise ConfigValidationError(f'`{path}` is only valid when the block contains a pose field.')

def _validate_normalization_config(value: Any, *, path: str, has_shared_statistics: bool=False) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f'`{path}` must be a mapping.')
    if 'companion' in value:
        raise ConfigValidationError(f'`{path}.companion` has been removed; configure the single shared path `data.train.pipeline.state_action_statistics` instead.')
    unknown = sorted(set(value) - {'mode', 'statistics'})
    if unknown:
        raise ConfigValidationError(f"`{path}` has unsupported key(s): {', '.join(unknown)}.")
    mode = str(value.get('mode', 'none')).lower()
    if mode not in {'none', 'mean_std', 'min_max', 'q01_q99'}:
        raise ConfigValidationError(f'Unsupported `{path}.mode`={mode!r}.')
    statistics = value.get('statistics')
    if statistics is not None:
        raise ConfigValidationError(f'`{path}.statistics` is an internal resolved field; authored configs must use `data.train.pipeline.state_action_statistics`.')
    if mode != 'none' and (not has_shared_statistics):
        raise ConfigValidationError(f'`{path}` mode {mode!r} requires `data.train.pipeline.state_action_statistics`.')

def _validate_statistics_generation_config(value: Any) -> None:
    if value is None:
        return
    path = 'data.train.pipeline.state_action_statistics_generation'
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f'`{path}` must be a mapping.')
    allowed = {'quantile_estimator', 'quantile_bins', 'histogram_count_dtype', 'accumulator_dtype', 'batch_size', 'num_workers', 'prefetch_factor', 'persistent_workers'}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigValidationError(f"`{path}` has unsupported key(s): {', '.join(unknown)}.")
    estimator = str(value.get('quantile_estimator', 'histogram_two_pass')).lower()
    if estimator not in {'exact', 'histogram_two_pass'}:
        raise ConfigValidationError(f'Unsupported `{path}.quantile_estimator`={estimator!r}.')
    bins = value.get('quantile_bins', 5000)
    if isinstance(bins, bool) or not isinstance(bins, int) or (not 2 <= bins <= 1000000):
        raise ConfigValidationError(f'`{path}.quantile_bins` must be an integer in [2, 1000000].')
    count_dtype = str(value.get('histogram_count_dtype', 'uint64')).lower()
    if count_dtype != 'uint64':
        raise ConfigValidationError(f"`{path}.histogram_count_dtype` currently only supports 'uint64'.")
    accumulator_dtype = str(value.get('accumulator_dtype', 'float64')).lower()
    if accumulator_dtype != 'float64':
        raise ConfigValidationError(f"`{path}.accumulator_dtype` currently only supports 'float64'.")
    for (key, default, minimum) in (('batch_size', 32, 1), ('num_workers', 4, 0), ('prefetch_factor', 2, 1)):
        setting = value.get(key, default)
        if isinstance(setting, bool) or not isinstance(setting, int) or setting < minimum:
            raise ConfigValidationError(f'`{path}.{key}` must be an integer >= {minimum}.')
    persistent_workers = value.get('persistent_workers', True)
    if not isinstance(persistent_workers, bool):
        raise ConfigValidationError(f'`{path}.persistent_workers` must be boolean.')

def _validate_action_supervision_config(value: Any) -> bool:
    if value is None:
        return False
    path = 'data.train.pipeline.action_supervision'
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f'`{path}` must be a mapping.')
    value = _plain(value)
    unknown = sorted(set(value) - {'gripper', 'target_overrides', 'hold_current_state', 'source_overrides', 'statistics_phase_rebind'})
    if unknown:
        raise ConfigValidationError(f"`{path}` has unsupported key(s): {', '.join(unknown)}.")
    enabled = False
    source_overrides = value.get('source_overrides')
    if source_overrides is not None:
        if not isinstance(source_overrides, (list, tuple)) or not source_overrides:
            raise ConfigValidationError(f'`{path}.source_overrides` must be a non-empty sequence.')
        configured_sources: set[str] = set()
        for (index, override) in enumerate(source_overrides):
            override_path = f'{path}.source_overrides[{index}]'
            if not isinstance(override, Mapping):
                raise ConfigValidationError(f'`{override_path}` must be a mapping.')
            unknown_override_keys = sorted(set(override) - {'source_names', 'action_source', 'disable_gripper_compensation', 'expected_episodes'})
            if unknown_override_keys:
                raise ConfigValidationError(f"`{override_path}` has unsupported key(s): {', '.join(unknown_override_keys)}.")
            source_names = override.get('source_names')
            if not isinstance(source_names, (list, tuple)) or not source_names or any((not isinstance(item, str) or not item.strip() for item in source_names)) or (len(set(source_names)) != len(source_names)):
                raise ConfigValidationError(f'`{override_path}.source_names` must be a unique non-empty string sequence.')
            duplicates = configured_sources.intersection(map(str, source_names))
            if duplicates:
                raise ConfigValidationError(f"`{override_path}.source_names` repeats configured source(s): {', '.join(sorted(duplicates))}.")
            configured_sources.update(map(str, source_names))
            if str(override.get('action_source', '')).lower() != 'stored':
                raise ConfigValidationError(f'`{override_path}.action_source` currently only supports `stored`.')
            if override.get('disable_gripper_compensation') is not True:
                raise ConfigValidationError(f'`{override_path}.disable_gripper_compensation` must explicitly be true.')
            expected_episodes = override.get('expected_episodes')
            if isinstance(expected_episodes, bool) or not isinstance(expected_episodes, int) or expected_episodes < 1:
                raise ConfigValidationError(f'`{override_path}.expected_episodes` must be an integer >= 1.')
        enabled = True
    overrides = value.get('target_overrides')
    if overrides is not None:
        try:
            from sipai.data.transforms.state_action.absolute_target_overrides import validate_target_override_config
            validate_target_override_config(overrides)
            enabled = True
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(f'Invalid `{path}.target_overrides`: {exc}') from exc
    hold_current_state = value.get('hold_current_state')
    if hold_current_state is not None:
        raise ValueError('Standalone hold_current_state supervision is not part of this training recipe.')
    gripper = value.get('gripper')
    phase_rebind = value.get('statistics_phase_rebind')
    if phase_rebind is not None:
        rebind_path = f'{path}.statistics_phase_rebind'
        if not isinstance(phase_rebind, Mapping):
            raise ConfigValidationError(f'`{rebind_path}` must be a mapping.')
        expected_keys = {'from_phase_fingerprint', 'to_phase_fingerprint'}
        unknown_rebind = sorted(set(phase_rebind) - expected_keys)
        missing_rebind = sorted(expected_keys - set(phase_rebind))
        if unknown_rebind or missing_rebind:
            details = []
            if unknown_rebind:
                details.append(f"unsupported key(s): {', '.join(unknown_rebind)}")
            if missing_rebind:
                details.append(f"missing key(s): {', '.join(missing_rebind)}")
            raise ConfigValidationError(f"`{rebind_path}` has {'; '.join(details)}.")
        fingerprints = []
        for key in sorted(expected_keys):
            fingerprint = phase_rebind.get(key)
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any((char not in '0123456789abcdef' for char in fingerprint)):
                raise ConfigValidationError(f'`{rebind_path}.{key}` must be a lowercase SHA-256 fingerprint.')
            fingerprints.append(fingerprint)
        if fingerprints[0] == fingerprints[1]:
            raise ConfigValidationError(f'`{rebind_path}` must bind two different phase fingerprints.')
        if gripper is None:
            raise ConfigValidationError(f'`{rebind_path}` requires `{path}.gripper`.')
    if gripper is None:
        if not enabled:
            raise ConfigValidationError(f'`{path}` requires gripper, target_overrides, hold_current_state, or source_overrides.')
        return enabled
    if not isinstance(gripper, Mapping):
        raise ConfigValidationError(f'`{path}.gripper` must be a mapping.')
    try:
        from sipai.data.transforms.state_action.gripper_supervision import GripperSupervisionSpec
        return GripperSupervisionSpec.from_config(gripper) is not None or enabled
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f'Invalid `{path}.gripper`: {exc}') from exc

def _validate_episode_filter_config(value: Any) -> None:
    if value is None:
        return
    path = 'data.train.pipeline.episode_filter'
    value = _plain(value)
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f'`{path}` must be a mapping.')
    unknown = sorted(set(value) - {'rules', 'exclusions'})
    if unknown:
        raise ConfigValidationError(f"`{path}` has unsupported key(s): {', '.join(unknown)}.")
    rules = value.get('rules') or ()
    exclusions = value.get('exclusions') or ()
    if not isinstance(rules, (list, tuple)):
        raise ConfigValidationError(f'`{path}.rules` must be a sequence.')
    if not isinstance(exclusions, (list, tuple)):
        raise ConfigValidationError(f'`{path}.exclusions` must be a sequence.')
    if not rules and (not exclusions):
        raise ConfigValidationError(f'`{path}` requires rules or exclusions.')
    for (index, rule) in enumerate(rules):
        rule_path = f'{path}.rules[{index}]'
        if not isinstance(rule, Mapping):
            raise ConfigValidationError(f'`{rule_path}` must be a mapping.')
        unknown_rule_keys = sorted(set(rule) - {'task_instructions', 'task_instruction_contains', 'required_episode_flag', 'episode_audit_path'})
        if unknown_rule_keys:
            raise ConfigValidationError(f"`{rule_path}` has unsupported key(s): {', '.join(unknown_rule_keys)}.")
        instructions = rule.get('task_instructions') or ()
        contains = rule.get('task_instruction_contains') or ()
        for (key, items) in (('task_instructions', instructions), ('task_instruction_contains', contains)):
            if not isinstance(items, (list, tuple)) or any((not isinstance(item, str) or not item.strip() for item in items)) or len(set(items)) != len(items):
                raise ConfigValidationError(f'`{rule_path}.{key}` must be a unique non-empty string sequence.')
        if not instructions and (not contains):
            raise ConfigValidationError(f'`{rule_path}` requires `task_instructions` or `task_instruction_contains`.')
        required_flag = rule.get('required_episode_flag')
        audit_path = rule.get('episode_audit_path')
        if required_flag is not None and (not isinstance(required_flag, str) or not required_flag.strip()):
            raise ConfigValidationError(f'`{rule_path}.required_episode_flag` must be a non-empty string.')
        if audit_path is not None and (not isinstance(audit_path, str) or not audit_path.strip()):
            raise ConfigValidationError(f'`{rule_path}.episode_audit_path` must be a non-empty path string.')
        if required_flag is None and audit_path is None:
            raise ConfigValidationError(f'`{rule_path}` requires `required_episode_flag` or `episode_audit_path`.')
    for (index, exclusion) in enumerate(exclusions):
        exclusion_path = f'{path}.exclusions[{index}]'
        if not isinstance(exclusion, Mapping):
            raise ConfigValidationError(f'`{exclusion_path}` must be a mapping.')
        unknown_exclusion_keys = sorted(set(exclusion) - {'source_names', 'task_instructions', 'expected_episodes'})
        if unknown_exclusion_keys:
            raise ConfigValidationError(f"`{exclusion_path}` has unsupported key(s): {', '.join(unknown_exclusion_keys)}.")
        for key in ('source_names', 'task_instructions'):
            items = exclusion.get(key)
            if not isinstance(items, (list, tuple)) or not items or any((not isinstance(item, str) or not item.strip() for item in items)) or (len(set(items)) != len(items)):
                raise ConfigValidationError(f'`{exclusion_path}.{key}` must be a unique non-empty string sequence.')
        expected_episodes = exclusion.get('expected_episodes')
        if isinstance(expected_episodes, bool) or not isinstance(expected_episodes, int) or expected_episodes < 0:
            raise ConfigValidationError(f'`{exclusion_path}.expected_episodes` must be an integer >= 0.')

def _validate_state_action_representation_config(cfg: Any, *, action_head: Mapping[str, Any]) -> None:
    data_train = get_nested_config(cfg, 'data.train', {})
    if not isinstance(data_train, Mapping):
        return
    pipeline = data_train.get('pipeline', {})
    if not isinstance(pipeline, Mapping):
        return
    state_cfg = pipeline.get('state')
    sampling_cfg = pipeline.get('sampling')
    action_cfg = _select_train_action_config(data_train, pipeline)
    action_supervision_cfg = pipeline.get('action_supervision')
    action_supervision_enabled = _validate_action_supervision_config(action_supervision_cfg)
    _validate_episode_filter_config(pipeline.get('episode_filter'))
    _validate_statistics_generation_config(pipeline.get('state_action_statistics_generation'))
    if not _uses_state_action_representation(state_cfg, sampling_cfg, action_cfg):
        if action_supervision_enabled:
            raise ConfigValidationError('`data.train.pipeline.action_supervision` requires the state/action representation path.')
        return
    if action_cfg is None:
        action_cfg = {}
    if action_supervision_enabled and str(action_cfg.get('source', 'stored')).lower() != 'state_derivation':
        raise ConfigValidationError('`data.train.pipeline.action_supervision` requires `action.source: state_derivation`.')
    statistics_path = pipeline.get('state_action_statistics')
    if statistics_path is not None and (not isinstance(statistics_path, str) or not statistics_path.strip()):
        raise ConfigValidationError('`data.train.pipeline.state_action_statistics` must be a non-empty JSON path string.')
    has_shared_statistics = statistics_path is not None
    if any((key in pipeline for key in ('sample_window_size', 'sample_stride'))):
        raise ConfigValidationError('New state/action representation config cannot also use legacy `sample_window_size` or `sample_stride`.')
    horizon = action_head.get('action_horizon', action_head.get('horizon'))
    if horizon is None:
        raise ConfigValidationError('New state/action representation config requires `model.action_head.action_horizon`.')
    _positive_config_int(horizon, path='model.action_head.action_horizon')
    if sampling_cfg is not None:
        if not isinstance(sampling_cfg, Mapping):
            raise ConfigValidationError('`data.train.pipeline.sampling` must be a mapping.')
        unknown = sorted(set(sampling_cfg) - {'anchor_stride'})
        if unknown:
            raise ConfigValidationError('`data.train.pipeline.sampling` has unsupported key(s): ' + ', '.join(unknown) + '.')
        _positive_config_int(sampling_cfg.get('anchor_stride', 1), path='data.train.pipeline.sampling.anchor_stride')
    if state_cfg is not None:
        if not isinstance(state_cfg, Mapping):
            raise ConfigValidationError('`data.train.pipeline.state` must be a mapping.')
        unknown = sorted(set(state_cfg) - {'representation', 'history_offsets', 'pose_delta_axes', 'rotation', 'fields', 'normalization'})
        if unknown:
            raise ConfigValidationError('`data.train.pipeline.state` has unsupported key(s): ' + ', '.join(unknown) + '.')
        representation = str(state_cfg.get('representation', 'absolute')).lower()
        if representation not in {'absolute', 'relative'}:
            raise ConfigValidationError(f'Unsupported state representation {representation!r}.')
        if representation == 'relative':
            if 'history_offsets' not in state_cfg:
                raise ConfigValidationError('`data.train.pipeline.state.history_offsets` is required for relative state representation.')
            if 'pose_delta_axes' not in state_cfg:
                raise ConfigValidationError('`data.train.pipeline.state.pose_delta_axes` is required for relative state representation.')
        if 'history_offsets' in state_cfg:
            _validate_history_offsets(state_cfg.get('history_offsets'), path='data.train.pipeline.state.history_offsets')
        if 'pose_delta_axes' in state_cfg:
            _validate_pose_delta_axes(state_cfg.get('pose_delta_axes'), path='data.train.pipeline.state.pose_delta_axes')
        _validate_representation_fields(state_cfg.get('fields'), path='data.train.pipeline.state.fields', role='state')
        has_pose = any((field.get('kind') == 'pose' for field in state_cfg['fields']))
        _validate_rotation_config(state_cfg.get('rotation'), path='data.train.pipeline.state.rotation', has_pose=has_pose)
        _validate_normalization_config(state_cfg.get('normalization'), path='data.train.pipeline.state.normalization', has_shared_statistics=has_shared_statistics)
    if not isinstance(action_cfg, Mapping):
        raise ConfigValidationError('`data.train.pipeline.action` must be a mapping for the new representation path.')
    source = str(action_cfg.get('source', 'stored')).lower()
    if source not in {'stored', 'state_derivation', 'none'}:
        raise ConfigValidationError(f'Unsupported action source {source!r}.')
    action_head_type = str(action_head.get('type', ''))
    if action_head_type == 'SimateBetaFlowMatchingActionHead' and source != 'none' and (action_head.get('action_dim') is None):
        raise ConfigValidationError('New SIMATE_BETA state/action representation config requires an explicit `model.action_head.action_dim`; the model capacity cannot be inferred from semantic fields.')
    legacy_keys = {'derivation', 'state_keys', 'gripper_state_keys', 'include_gripper_width', 'preserve_absolute_fields', 'stored_horizon', 'stored_step_interval', 'stored_normalization', 'action_step_interval', 'step_offset'}
    conflicts = sorted(set(action_cfg) & legacy_keys)
    if conflicts:
        raise ConfigValidationError('New state/action representation fields cannot be combined with legacy action key(s): ' + ', '.join(conflicts) + '.')
    if source == 'state_derivation':
        allowed = {'source', 'representation', 'target_start_offset', 'target_step_interval', 'tail_padding', 'rotation', 'fields', 'normalization', 'pose_delta_axes'}
        unknown = sorted(set(action_cfg) - allowed)
        if unknown:
            raise ConfigValidationError('`data.train.pipeline.action` has unsupported key(s): ' + ', '.join(unknown) + '.')
        representation = str(action_cfg.get('representation', 'absolute')).lower()
        if representation not in {'absolute', 'relative', 'delta'}:
            raise ConfigValidationError(f'Unsupported action representation {representation!r}.')
        if representation != 'absolute' and 'pose_delta_axes' not in action_cfg:
            raise ConfigValidationError('`data.train.pipeline.action.pose_delta_axes` is required for relative/delta action representation.')
        if 'pose_delta_axes' in action_cfg:
            _validate_pose_delta_axes(action_cfg.get('pose_delta_axes'), path='data.train.pipeline.action.pose_delta_axes')
        _positive_config_int(action_cfg.get('target_start_offset', 1), path='data.train.pipeline.action.target_start_offset')
        _positive_config_int(action_cfg.get('target_step_interval', 1), path='data.train.pipeline.action.target_step_interval')
        tail_padding = str(action_cfg.get('tail_padding', 'none')).lower()
        if tail_padding not in {'none', 'repeat_last'}:
            raise ConfigValidationError(f'`data.train.pipeline.action.tail_padding` must be `none` or `repeat_last`, got {tail_padding!r}.')
        _validate_representation_fields(action_cfg.get('fields'), path='data.train.pipeline.action.fields', role='action', allow_numeric_action_select=True)
        has_pose = any((field.get('kind') == 'pose' for field in action_cfg['fields']))
        _validate_rotation_config(action_cfg.get('rotation'), path='data.train.pipeline.action.rotation', has_pose=has_pose)
        _validate_normalization_config(action_cfg.get('normalization'), path='data.train.pipeline.action.normalization', has_shared_statistics=has_shared_statistics)
    elif source == 'stored':
        typed = any((key in action_cfg for key in ('representation', 'rotation', 'fields')))
        if typed:
            allowed = {'source', 'representation', 'pose_delta_axes', 'rotation', 'fields', 'normalization'}
            unknown = sorted(set(action_cfg) - allowed)
            if unknown:
                raise ConfigValidationError('Typed stored action config has unsupported key(s): ' + ', '.join(unknown) + '.')
            representation = str(action_cfg.get('representation', 'absolute')).lower()
            if representation not in {'absolute', 'relative', 'delta'}:
                raise ConfigValidationError(f'Unsupported action representation {representation!r}.')
            if representation != 'absolute' and 'pose_delta_axes' not in action_cfg:
                raise ConfigValidationError('`data.train.pipeline.action.pose_delta_axes` is required for relative/delta action representation.')
            if 'pose_delta_axes' in action_cfg:
                _validate_pose_delta_axes(action_cfg.get('pose_delta_axes'), path='data.train.pipeline.action.pose_delta_axes')
            _validate_representation_fields(action_cfg.get('fields'), path='data.train.pipeline.action.fields', role='action', stored_typed=True)
            for (index, field) in enumerate(action_cfg['fields']):
                anchor = field.get('anchor_source')
                if field.get('kind') == 'gripper' and anchor is not None:
                    raise ConfigValidationError(f'`data.train.pipeline.action.fields[{index}].anchor_source` is forbidden for gripper.')
                if representation != 'absolute' and field.get('kind') != 'gripper':
                    if not isinstance(anchor, str) or not anchor.strip():
                        raise ConfigValidationError(f'`data.train.pipeline.action.fields[{index}].anchor_source` is required.')
            has_pose = any((field.get('kind') == 'pose' for field in action_cfg['fields']))
            _validate_rotation_config(action_cfg.get('rotation'), path='data.train.pipeline.action.rotation', has_pose=has_pose)
        else:
            unknown = sorted(set(action_cfg) - {'source', 'normalization'})
            if unknown:
                raise ConfigValidationError('New stored action config only accepts `source` and `normalization`; unsupported: ' + ', '.join(unknown) + '.')
        _validate_normalization_config(action_cfg.get('normalization'), path='data.train.pipeline.action.normalization', has_shared_statistics=has_shared_statistics)
    else:
        unknown = sorted(set(action_cfg) - {'source'})
        if unknown:
            raise ConfigValidationError('`action.source: none` does not accept other action parameters.')
    model_norm = action_head.get('action_normalization')
    model_mode = str(model_norm.get('mode', 'none')).lower() if isinstance(model_norm, Mapping) else 'none'
    if model_mode not in {'', 'none'}:
        raise ConfigValidationError('New state/action representation config cannot use model-local action_normalization; normalization is owned by the shared processor.')

def validate_sipai_config_schema(cfg: Any) -> None:
    if get_config_value(cfg, 'model', None) is None:
        return
    if is_normalized_runtime_config(cfg):
        return
    _reject_legacy_simate_beta_component_names(cfg, backbone_type_path='model.backbone.type', action_head_type_path='model.action_head.type')
    for section in PUBLIC_CONFIG_REQUIRED_SECTIONS:
        _require_section(cfg, section)
    for section in RUNTIME_CONFIG_SECTIONS:
        if get_config_value(cfg, section, None) is not None:
            raise ConfigValidationError(f'Public SiPAI config must not define internal runtime section `{section}`.')
    for field in LEGACY_TOP_LEVEL_FIELDS:
        if get_config_value(cfg, field, None) is not None:
            raise ConfigValidationError(f'Public SiPAI config must not define legacy top-level field `{field}`; use `experiment` or `train_cfg` instead.')
    normalized_engine = _normalize_engine_cfg(cfg)
    _require_nested(cfg, 'model.type')
    if get_nested_config(cfg, 'model.type', None) == 'ComposableVLA':
        raise ConfigValidationError('Public SiPAI config must use `model.type: ComposableVLAFramework`, not alias `ComposableVLA`.')
    _require_nested(cfg, 'model.backbone.type')
    if get_nested_config(cfg, 'model.action_head', None) is None:
        raise ConfigValidationError('Public SiPAI config must use `model.action_head`, not `model.action_model`.')
    action_head_type = _require_nested(cfg, 'model.action_head.type')
    action_head = dict(_plain(get_nested_config(cfg, 'model.action_head', {})) or {})
    allowed_action_head_fields = PUBLIC_ACTION_HEAD_FIELDS_BY_TYPE.get(str(action_head_type), PUBLIC_ACTION_HEAD_FIELDS)
    extra_action_head_fields = sorted(set(action_head) - allowed_action_head_fields)
    if extra_action_head_fields:
        fields = ', '.join(extra_action_head_fields)
        allowed = ', '.join(sorted(allowed_action_head_fields))
        raise ConfigValidationError(f'Public SiPAI config model.action_head exposes internal field(s): {fields}. Expected only: {allowed}.')
    _validate_state_action_representation_config(cfg, action_head=action_head)
    _require_nested(cfg, 'data.train')
    _require_nested(cfg, 'data.train.type')
    _require_nested(cfg, 'data.train.root')
    if get_nested_config(cfg, 'data.train.type', None) in {'canonical_embodied', 'canonical'}:
        raise ConfigValidationError('Public SiPAI config must use `data.train.type: CanonicalEmbodiedDataset`, not canonical dataset aliases.')
    reader = str(get_nested_config(cfg, 'data.train.reader', get_nested_config(cfg, 'data.train.reader_backend', 'simate_unified'))).lower()
    manifest_type = str(get_nested_config(cfg, 'data.train.manifest.type', get_nested_config(cfg, 'data.train.manifest_builder', 'simate_unified'))).lower()
    if reader != 'simate_unified':
        raise ConfigValidationError(f"Public SiPAI training configs must use data.train.reader='simate_unified', got {reader!r}.")
    if manifest_type != 'simate_unified':
        raise ConfigValidationError(f"Public SiPAI training configs must use data.train.manifest.type='simate_unified', got {manifest_type!r}.")
    output_format = get_nested_config(cfg, 'train_dataloader.collate_fn.output_format', 'canonical')
    if output_format == 'legacy_examples':
        raise ConfigValidationError('`train_dataloader.collate_fn.output_format=legacy_examples` has been removed; use `canonical`.')
    if output_format not in SUPPORTED_DATALOADER_OUTPUT_FORMATS:
        supported = ', '.join(SUPPORTED_DATALOADER_OUTPUT_FORMATS)
        raise ConfigValidationError(f'Unsupported train_dataloader.collate_fn.output_format={output_format!r}; expected one of: {supported}.')
    train_cfg = dict(_plain(get_config_value(cfg, 'train_cfg', {})) or {})
    _validate_export_final_weights(train_cfg.get('export_final_weights', 'ema'), path='train_cfg.export_final_weights')
    for field in ('scheduler', 'scheduler_specific_kwargs', 'gradient_clipping'):
        if field in train_cfg:
            raise ConfigValidationError(f'Public SiPAI config must not define `train_cfg.{field}`; use `param_scheduler` for schedules and `train_cfg.max_grad_norm` for gradient clipping.')
    normalized_strategy = (normalized_engine or {}).get('strategy') or {}
    if normalized_strategy.get('train_micro_batch_size_per_gpu') is not None:
        raise ConfigValidationError('`engine.strategy.train_micro_batch_size_per_gpu` is derived from `train_dataloader.batch_size`; remove it from public configs.')
    sampler_type = get_nested_config(cfg, 'train_dataloader.sampler.type', None)
    if sampler_type != 'MemoryMixtureSampler':
        raise ConfigValidationError('This release requires train_dataloader.sampler.type=MemoryMixtureSampler.')
    if normalized_strategy.get('type') != 'FSDPStrategy' or normalized_strategy.get('mode') != 'hybrid_shard':
        raise ConfigValidationError('This release requires engine.strategy.preset=fsdp2_hsdp.')
    scheduler_type = get_nested_config(cfg, 'param_scheduler.type', None)
    if scheduler_type == 'cosine_with_min_lr':
        raise ConfigValidationError('Public SiPAI config must use `param_scheduler.type: CosineWithMinLRScheduler`, not alias `cosine_with_min_lr`.')

def _normalize_model_cfg(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(_plain(model_cfg) or {})
    model_type = str(model.get('type', 'ComposableVLAFramework'))
    framework_name = 'ComposableVLAFramework' if model_type in {'ComposableVLA', 'ComposableVLAFramework'} else model_type
    backbone = dict(model.get('backbone') or {})
    backbone_cfg = {'type': backbone.get('type')}
    input_cameras = _pop_any(backbone, 'input_cameras', 'camera_names', default=None)
    if input_cameras is not None:
        backbone_cfg['input_cameras'] = input_cameras
    include_state = _pop_any(backbone, 'include_state', default=None)
    if include_state is not None:
        backbone_cfg['include_state'] = include_state
    state_keys = _pop_any(backbone, 'state_keys', default=None)
    if state_keys is not None:
        backbone_cfg['state_keys'] = state_keys
    backbone_aliases = {'pretrained': _pop_any(backbone, 'pretrained', default=None), 'base_vlm': _pop_any(backbone, 'base_vlm', default=None), 'base_wm': _pop_any(backbone, 'base_wm', default=None), 'attn_implementation': _pop_any(backbone, 'attn_implementation', default=None), 'hidden_dim': _pop_any(backbone, 'hidden_dim', default=None), 'vl_hidden_dim': _pop_any(backbone, 'vl_hidden_dim', default=None)}
    if backbone_aliases['base_vlm'] is None and backbone_aliases['base_wm'] is None:
        backbone_aliases['base_vlm'] = backbone_aliases['pretrained']
    if backbone_aliases['base_wm'] is None and backbone.get('world_model', False):
        backbone_aliases['base_wm'] = backbone_aliases['pretrained']
    if backbone_aliases['pretrained'] is None:
        backbone_aliases['pretrained'] = backbone_aliases['base_wm'] or backbone_aliases['base_vlm']
    if backbone_aliases['vl_hidden_dim'] is None:
        backbone_aliases['vl_hidden_dim'] = backbone_aliases['hidden_dim']
    if backbone_aliases['hidden_dim'] is None:
        backbone_aliases['hidden_dim'] = backbone_aliases['vl_hidden_dim']
    backbone_cfg.update({key: value for (key, value) in backbone_aliases.items() if value is not None})
    backbone_cfg.update({key: value for (key, value) in backbone.items() if value is not None})
    action_head = dict(model.get('action_head') or model.get('action_model') or {})
    action_model_cfg = dict(action_head)
    if 'horizon' in action_model_cfg and 'action_horizon' not in action_model_cfg:
        action_model_cfg['action_horizon'] = action_model_cfg.pop('horizon')
    if 'future_window_size' in action_model_cfg and 'future_action_window_size' not in action_model_cfg:
        action_model_cfg['future_action_window_size'] = action_model_cfg.pop('future_window_size')
    if 'past_window_size' in action_model_cfg and 'past_action_window_size' not in action_model_cfg:
        action_model_cfg['past_action_window_size'] = action_model_cfg.pop('past_window_size')
    diffusion_cfg = action_model_cfg.pop('diffusion', None)
    if diffusion_cfg is not None and 'diffusion_model_cfg' not in action_model_cfg:
        diffusion = dict(diffusion_cfg)
        if 'num_train_timesteps' in diffusion and 'num_timestep_buckets' not in action_model_cfg:
            action_model_cfg['num_timestep_buckets'] = diffusion.pop('num_train_timesteps')
        if 'num_inference_timesteps' in diffusion and 'num_inference_timesteps' not in action_model_cfg:
            action_model_cfg['num_inference_timesteps'] = diffusion.pop('num_inference_timesteps')
        action_model_cfg['diffusion_model_cfg'] = diffusion
    _apply_action_head_defaults(action_model_cfg)
    return {'name': framework_name, 'backbone': backbone_cfg, 'action_model': action_model_cfg}

def resolve_state_action_statistics_path(cfg: Any, *, config_base_dir: str | Path | None=None) -> Path | None:
    value = get_nested_config(cfg, 'data.train.pipeline.state_action_statistics', None)
    if value is None:
        value = get_nested_config(cfg, 'datasets.vla_data.state_action_statistics', None)
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigValidationError('`data.train.pipeline.state_action_statistics` must be a non-empty JSON path string.')
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        base_dir = Path(config_base_dir).expanduser() if config_base_dir is not None else Path.cwd()
        path = base_dir / path
    return path.resolve(strict=False)

def _normalize_data_cfg(cfg: Any, *, config_base_dir: str | Path | None=None, load_state_action_statistics: bool=True) -> dict[str, Any]:
    data_train = dict(_plain(get_nested_config(cfg, 'data.train', {})) or {})
    dataloader = dict(_plain(get_config_value(cfg, 'train_dataloader', {})) or {})
    collate_fn = dict(dataloader.pop('collate_fn', {}) or {})
    manifest_cfg = dict(data_train.pop('manifest', {}) or {})
    sources = _normalize_data_sources(data_train, manifest_cfg)
    pipeline_cfg = dict(data_train.pop('pipeline', {}) or {})
    action_cfg = _select_train_action_config(data_train, pipeline_cfg)
    data_train.pop('action', None)
    pipeline_cfg.pop('action', None)
    state_cfg = pipeline_cfg.pop('state', None)
    sampling_cfg = pipeline_cfg.pop('sampling', None)
    action_supervision_cfg = pipeline_cfg.pop('action_supervision', None)
    if isinstance(action_supervision_cfg, Mapping):
        action_supervision_cfg = dict(action_supervision_cfg)
        gripper_cfg = action_supervision_cfg.get('gripper')
        if isinstance(gripper_cfg, Mapping):
            gripper_cfg = dict(gripper_cfg)
            phase_index = gripper_cfg.get('phase_index')
            if isinstance(phase_index, str) and phase_index.strip():
                phase_path = Path(phase_index).expanduser()
                if not phase_path.is_absolute():
                    base_dir = Path(config_base_dir).expanduser() if config_base_dir is not None else Path.cwd()
                    phase_path = base_dir / phase_path
                gripper_cfg['phase_index'] = str(phase_path.resolve(strict=False))
            action_supervision_cfg['gripper'] = gripper_cfg
        override_cfg = action_supervision_cfg.get('target_overrides')
        if isinstance(override_cfg, Mapping):
            override_cfg = dict(override_cfg)
            override_path = Path(override_cfg['path']).expanduser()
            if not override_path.is_absolute():
                base_dir = Path(config_base_dir).expanduser() if config_base_dir is not None else Path.cwd()
                override_path = base_dir / override_path
            override_cfg['path'] = str(override_path.resolve(strict=False))
            action_supervision_cfg['target_overrides'] = override_cfg
        hold_cfg = action_supervision_cfg.get('hold_current_state')
        if isinstance(hold_cfg, Mapping):
            hold_cfg = dict(hold_cfg)
            hold_path = Path(hold_cfg['path']).expanduser()
            if not hold_path.is_absolute():
                base_dir = Path(config_base_dir).expanduser() if config_base_dir is not None else Path.cwd()
                hold_path = base_dir / hold_path
            hold_cfg['path'] = str(hold_path.resolve(strict=False))
            action_supervision_cfg['hold_current_state'] = hold_cfg
    pipeline_cfg.pop('state_action_statistics', None)
    pipeline_cfg.pop('state_action_statistics_generation', None)
    statistics_path = resolve_state_action_statistics_path(cfg, config_base_dir=config_base_dir)
    representation_cfg = None
    if _uses_state_action_representation(state_cfg, sampling_cfg, action_cfg):
        if load_state_action_statistics and statistics_path is not None and statistics_path.is_file():
            statistics_artifact = _load_state_action_statistics_artifact(statistics_path)
            _validate_action_supervision_statistics(action_supervision_cfg, payload=statistics_artifact, artifact_path=statistics_path)
            state_cfg = _resolve_statistics_normalization(state_cfg, role='state', payload=statistics_artifact, artifact_path=statistics_path)
            action_cfg = _resolve_statistics_normalization(action_cfg, role='action', payload=statistics_artifact, artifact_path=statistics_path)
        public_action_head = dict(_plain(get_nested_config(cfg, 'model.action_head', {})) or {})
        horizon = public_action_head.get('action_horizon', public_action_head.get('horizon'))
        if horizon is None:
            raise ConfigValidationError('State/action representation config requires `model.action_head.action_horizon`.')
        action_head_type = str(public_action_head.get('type', ''))
        representation_cfg = {'schema_version': 1, 'horizon': int(horizon), 'sampling': dict(sampling_cfg or {}), 'action': dict(action_cfg or {}), 'dim_policy': 'pad_to_capacity' if action_head_type == 'SimateBetaFlowMatchingActionHead' else 'exact'}
        if state_cfg is not None:
            representation_cfg['state'] = dict(state_cfg)
        action_dim = public_action_head.get('action_dim')
        if action_dim is not None:
            representation_cfg['model_dim'] = int(action_dim)
    output_format = collate_fn.get('output_format', data_train.pop('output_format', 'canonical'))
    if output_format == 'legacy_examples':
        raise ConfigValidationError('`legacy_examples` output format has been removed; use `canonical`.')
    if output_format != 'canonical':
        raise ConfigValidationError('Only `canonical` dataloader output is supported in the SiPAI training path.')
    data_cfg: dict[str, Any] = {'dataset_py': data_train.pop('dataset_py', data_train.pop('type', 'CanonicalEmbodiedDataset')), 'data_root': data_train.pop('root', data_train.pop('data_root', '.')), 'reader_backend': data_train.pop('reader', data_train.pop('reader_backend', 'simate_unified')), 'manifest_builder': manifest_cfg.pop('type', data_train.pop('manifest_builder', 'simate_unified')), 'manifest_max_episodes': manifest_cfg.pop('max_episodes', data_train.pop('manifest_max_episodes', None)), 'manifest_episode_glob': manifest_cfg.pop('episode_glob', data_train.pop('manifest_episode_glob', None)), 'manifest_cache_path': manifest_cfg.pop('cache_path', data_train.pop('manifest_cache_path', None)), 'manifest_cache_policy': manifest_cfg.pop('cache_policy', data_train.pop('manifest_cache_policy', None)), 'manifest_sample_episodes': manifest_cfg.pop('sample_episodes', data_train.pop('manifest_sample_episodes', None)), 'manifest_sample_seed': manifest_cfg.pop('sample_seed', data_train.pop('manifest_sample_seed', None)), 'output_format': output_format, 'language_source': pipeline_cfg.pop('language_source', data_train.pop('language_source', 'sub_task')), 'sample_window_size': pipeline_cfg.pop('sample_window_size', data_train.pop('sample_window_size', 1)), 'sample_stride': pipeline_cfg.pop('sample_stride', data_train.pop('sample_stride', 1)), 'strict': pipeline_cfg.pop('strict', data_train.pop('strict', True)), 'action_config': action_cfg, 'action_supervision': action_supervision_cfg, 'state_action_representation': representation_cfg, 'state_action_statistics': str(statistics_path) if statistics_path is not None else None, 'per_device_batch_size': dataloader.pop('batch_size', data_train.pop('per_device_batch_size', 1)), 'num_workers': dataloader.pop('num_workers', data_train.pop('num_workers', 0)), 'pin_memory': dataloader.pop('pin_memory', data_train.pop('pin_memory', False)), 'shuffle': dataloader.pop('shuffle', data_train.pop('shuffle', False))}
    if sources is not None:
        data_cfg['sources'] = sources
    data_cfg.update(data_train)
    data_cfg.update(pipeline_cfg)
    data_cfg.update(manifest_cfg)
    data_cfg.update(dataloader)
    return {'vla_data': {key: value for (key, value) in data_cfg.items() if value is not None}}

def _load_state_action_statistics_artifact(artifact_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(artifact_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ConfigValidationError(f'State/action statistics do not exist: {artifact_path}') from exc
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f'State/action statistics are not valid JSON: {artifact_path}') from exc
    if not isinstance(payload, Mapping):
        raise ConfigValidationError(f'State/action statistics must contain an object: {artifact_path}')
    return payload

def _validate_action_supervision_statistics(config: Any, *, payload: Mapping[str, Any], artifact_path: Path) -> None:
    requested = None
    if isinstance(config, Mapping):
        gripper = config.get('gripper')
        if isinstance(gripper, Mapping):
            try:
                from sipai.data.transforms.state_action.gripper_supervision import GripperSupervisionSpec, phase_index_fingerprint
                spec = GripperSupervisionSpec.from_config(gripper)
                if spec is not None:
                    requested = spec.to_dict(phase_fingerprint=phase_index_fingerprint(spec.phase_index))
            except (OSError, TypeError, ValueError) as exc:
                raise ConfigValidationError(f'Cannot resolve action supervision for statistics validation: {exc}') from exc
    recorded = payload.get('action_supervision_spec')
    if requested == recorded:
        return
    phase_rebind = config.get('statistics_phase_rebind') if isinstance(config, Mapping) else None
    if isinstance(phase_rebind, Mapping) and isinstance(requested, Mapping) and isinstance(recorded, Mapping):
        requested_contract = dict(requested)
        recorded_contract = dict(recorded)
        requested_phase = requested_contract.pop('phase_fingerprint', None)
        recorded_phase = recorded_contract.pop('phase_fingerprint', None)
        if recorded_contract == requested_contract and recorded_phase == phase_rebind.get('from_phase_fingerprint') and (requested_phase == phase_rebind.get('to_phase_fingerprint')):
            return
    raise ConfigValidationError(f'State/action statistics action supervision does not match the configured gripper compensation: {artifact_path}')

def _resolve_statistics_normalization(block: Any, *, role: str, payload: Mapping[str, Any], artifact_path: Path) -> Any:
    if not isinstance(block, Mapping):
        return block
    resolved_block = dict(block)
    normalization = resolved_block.get('normalization')
    if not isinstance(normalization, Mapping):
        normalization = {'mode': 'none'}
    try:
        from sipai.data.transforms.state_action.processor import CompiledStateActionTransform
        artifact_transform = CompiledStateActionTransform.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigValidationError(f'State/action statistics contain an invalid compiled contract: {artifact_path}: {exc}') from exc
    spec = artifact_transform.state_spec if role == 'state' else artifact_transform.action_spec
    if role == 'action':
        requested_source = str(resolved_block.get('source', 'stored')).lower()
        if artifact_transform.action_source != requested_source:
            raise ConfigValidationError(f'State/action statistics action source {artifact_transform.action_source!r} does not match requested source {requested_source!r}.')
    if spec is None:
        raise ConfigValidationError(f'State/action statistics have no {role}_spec: {artifact_path}')
    artifact_normalization = {'mode': spec.normalization_mode, **({'statistics': spec.normalization_statistics.to_dict()} if spec.normalization_statistics is not None else {})}
    requested_mode = str(normalization.get('mode', 'none')).lower()
    artifact_mode = str(artifact_normalization.get('mode', 'none')).lower()
    if artifact_mode != requested_mode:
        raise ConfigValidationError(f'State/action statistics {role} normalization mode {artifact_mode!r} does not match requested mode {requested_mode!r}.')
    statistics = artifact_normalization.get('statistics')
    if requested_mode != 'none' and (not isinstance(statistics, Mapping)):
        raise ConfigValidationError(f'State/action statistics lack {role} values for mode {requested_mode!r}.')
    resolved_block['normalization'] = {'mode': requested_mode, **({'statistics': dict(statistics)} if isinstance(statistics, Mapping) else {}), 'statistics_spec': spec.to_dict()}
    return resolved_block

def _normalize_trainer_cfg(cfg: Any) -> dict[str, Any]:
    train_cfg = dict(_plain(get_config_value(cfg, 'train_cfg', {})) or {})
    optim_wrapper = dict(_plain(get_config_value(cfg, 'optim_wrapper', {})) or {})
    scheduler = get_config_value(cfg, 'param_scheduler', None)
    scheduler_cfg: dict[str, Any] = {}
    default_hooks = dict(_plain(get_config_value(cfg, 'default_hooks', {})) or {})
    experiment = dict(_plain(get_config_value(cfg, 'experiment', {})) or {})
    optimizer_cfg = dict(optim_wrapper.get('optimizer') or {})
    if 'type' in optimizer_cfg and 'name' not in optimizer_cfg:
        optimizer_cfg['name'] = optimizer_cfg['type']
    freeze = train_cfg.pop('freeze', '')
    freeze_modules = ' '.join((str(item) for item in _as_list(freeze)))
    wandb_cfg = train_cfg.pop('wandb', experiment.get('wandb', None))
    checkpoint_hook = dict(default_hooks.get('checkpoint') or {})
    logger_hook = dict(default_hooks.get('logger') or {})
    save_last = train_cfg.pop('save_last', checkpoint_hook.get('save_last', True))
    save_final_checkpoint = train_cfg.pop('save_final_checkpoint', save_last)
    export_final_weights = _validate_export_final_weights(train_cfg.pop('export_final_weights', 'ema'), path='train_cfg.export_final_weights')
    save_interval = train_cfg.pop('save_interval', checkpoint_hook.get('interval', 1))
    logging_interval = train_cfg.pop('logging_interval', train_cfg.pop('logging_frequency', logger_hook.get('interval', 25)))
    loss_scale = dict(train_cfg.pop('loss_scale', {}) or {})
    loss_scale.setdefault('vla', 1.0)
    trainer = {'max_train_steps': train_cfg.pop('max_steps', train_cfg.pop('max_train_steps', 1)), 'num_warmup_steps': train_cfg.pop('num_warmup_steps', 0), 'save_interval': save_interval, 'eval_interval': train_cfg.pop('eval_interval', 1), 'learning_rate': {'base': optimizer_cfg.get('lr', get_nested_config(cfg, 'experiment.lr', 0.0001))}, 'lr_scheduler_type': 'constant', 'freeze_modules': freeze_modules, 'loss_scale': loss_scale, 'max_grad_norm': train_cfg.pop('max_grad_norm', 1.0), 'ema_decay': train_cfg.pop('ema_decay', None), 'weight_decay': optimizer_cfg.get('weight_decay', train_cfg.pop('weight_decay', 0.0)), 'logging_frequency': logging_interval, 'gradient_accumulation_steps': train_cfg.pop('gradient_accumulation_steps', 1), 'optimizer': optimizer_cfg, 'wandb': wandb_cfg, 'save_last': save_last, 'save_final_checkpoint': save_final_checkpoint, 'export_final_weights': export_final_weights}
    if isinstance(scheduler, str):
        trainer['lr_scheduler_type'] = scheduler
    elif isinstance(scheduler, list) and scheduler:
        scheduler_cfg = dict(_plain(scheduler[0]))
        trainer['lr_scheduler_type'] = scheduler_cfg.pop('type', 'constant')
    elif scheduler is not None:
        scheduler_cfg = dict(_plain(scheduler))
        trainer['lr_scheduler_type'] = scheduler_cfg.pop('type', 'constant')
    if scheduler_cfg:
        trainer['scheduler'] = scheduler_cfg
    paramwise_cfg = optim_wrapper.get('paramwise_cfg', {})
    custom_keys = dict(paramwise_cfg.get('custom_keys', {}) or {})
    for (module_name, module_cfg) in custom_keys.items():
        module_cfg = dict(module_cfg or {})
        if 'lr' in module_cfg:
            trainer['learning_rate'][module_name] = module_cfg['lr']
        elif 'lr_mult' in module_cfg:
            trainer['learning_rate'][module_name] = trainer['learning_rate']['base'] * float(module_cfg['lr_mult'])
    trainer.update(train_cfg)
    return {key: value for (key, value) in trainer.items() if value is not None}

def normalize_sipai_config(cfg: Any, *, config_base_dir: str | Path | None=None, load_state_action_statistics: bool=True) -> Any:
    if is_normalized_runtime_config(cfg):
        statistics_path = resolve_state_action_statistics_path(cfg, config_base_dir=config_base_dir)
        representation = get_nested_config(cfg, 'datasets.vla_data.state_action_representation', None)
        if load_state_action_statistics and statistics_path is not None and statistics_path.is_file() and isinstance(representation, Mapping):
            payload = _load_state_action_statistics_artifact(statistics_path)
            _validate_action_supervision_statistics(get_nested_config(cfg, 'datasets.vla_data.action_supervision', None), payload=payload, artifact_path=statistics_path)
            resolved_representation = dict(_plain(representation) or {})
            for role in ('state', 'action'):
                if role in resolved_representation:
                    resolved_representation[role] = _resolve_statistics_normalization(resolved_representation[role], role=role, payload=payload, artifact_path=statistics_path)
            if OmegaConf.is_config(cfg):
                cfg.datasets.vla_data.state_action_representation = OmegaConf.create(resolved_representation)
                cfg.datasets.vla_data.action_config = OmegaConf.create(resolved_representation.get('action'))
            elif isinstance(cfg, dict):
                data_cfg = cfg['datasets']['vla_data']
                data_cfg['state_action_representation'] = resolved_representation
                data_cfg['action_config'] = resolved_representation.get('action')
        _reject_legacy_simate_beta_component_names(cfg, backbone_type_path='framework.backbone.type', action_head_type_path='framework.action_model.type')
        _configure_simate_beta_shared_state_tokenization(get_nested_config(cfg, 'framework.backbone', {}), get_nested_config(cfg, 'datasets.vla_data.state_action_representation', None))
        normalized_engine = _normalize_engine_cfg(cfg)
        if normalized_engine is not None:
            if OmegaConf.is_config(cfg):
                cfg.engine = OmegaConf.create(normalized_engine)
            elif isinstance(cfg, dict):
                cfg['engine'] = normalized_engine
            else:
                cfg.engine = normalized_engine
        return cfg
    if get_config_value(cfg, 'model', None) is None:
        return cfg
    validate_sipai_config_schema(cfg)
    normalized = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else cfg)
    experiment = get_config_value(normalized, 'experiment', {})
    if get_config_value(normalized, 'run_id', None) is None:
        normalized.run_id = get_config_value(experiment, 'name', 'sipai_train')
    if get_config_value(normalized, 'run_root_dir', None) is None:
        normalized.run_root_dir = get_config_value(experiment, 'work_dir', 'results/Checkpoints')
    if get_config_value(normalized, 'seed', None) is None:
        normalized.seed = get_config_value(experiment, 'seed', 42)
    if get_config_value(normalized, 'is_debug', None) is None:
        normalized.is_debug = get_config_value(experiment, 'debug', False)
    datasets_cfg = _normalize_data_cfg(normalized, config_base_dir=config_base_dir, load_state_action_statistics=load_state_action_statistics)
    framework_cfg = _normalize_model_cfg(normalized.model)
    representation_cfg = datasets_cfg.get('vla_data', {}).get('state_action_representation')
    backbone_cfg = framework_cfg.get('backbone', {})
    _configure_simate_beta_shared_state_tokenization(backbone_cfg, representation_cfg)
    normalized.framework = OmegaConf.create(framework_cfg)
    normalized.datasets = OmegaConf.create(datasets_cfg)
    normalized.trainer = OmegaConf.create(_normalize_trainer_cfg(normalized))
    normalized_engine = _normalize_engine_cfg(normalized)
    if normalized_engine is not None:
        normalized.engine = OmegaConf.create(normalized_engine)
    return normalized

def _configure_simate_beta_shared_state_tokenization(backbone_cfg: Any, representation_cfg: Any) -> None:
    if not isinstance(backbone_cfg, Mapping) or not isinstance(representation_cfg, Mapping):
        return
    state_cfg = representation_cfg.get('state')
    if not isinstance(state_cfg, Mapping) or not state_cfg or backbone_cfg.get('type') != 'SimateBetaPrefixBackbone' or (not bool(backbone_cfg.get('discrete_state_input', True))):
        return
    state_normalization = state_cfg.get('normalization') or {}
    state_normalization_mode = str(state_normalization.get('mode', 'none')).lower()
    if state_normalization_mode not in {'min_max', 'q01_q99'}:
        raise ConfigValidationError(f'SimateBeta discrete state tokenization requires shared state normalization in the nominal [-1, 1] domain; use min_max or q01_q99, not {state_normalization_mode!r}.')
    backbone_cfg['state_is_normalized'] = True
    backbone_cfg['state_tokenization_domain'] = 'shared_unit_interval'
