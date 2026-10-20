from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
import numpy as np
from .fields import flatten_field, read_field
from .normalization import normalize_values, unnormalize_values
from .representation import canonicalize_semantic_values, decode_action_values, encode_action_values, encode_state_values
from .spec import SCHEMA_VERSION, RepresentationSpec, StoredActionSpec, compile_representation_spec, compile_stored_action_spec, compile_typed_stored_action_spec

@dataclass(frozen=True)
class CompiledStateActionTransform:
    state_spec: RepresentationSpec | None
    action_spec: RepresentationSpec | StoredActionSpec | None
    action_source: str
    action_horizon: int
    anchor_stride: int
    target_start_offset: int
    target_step_interval: int
    required_state_sources: tuple[str, ...]
    tail_padding: str = 'none'
    compute_dtype: str = 'float64'
    output_dtype: str = 'float32'
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, 'compute_dtype', _floating_dtype_name(self.compute_dtype, 'compute_dtype'))
        object.__setattr__(self, 'output_dtype', _floating_dtype_name(self.output_dtype, 'output_dtype'))

    def encode_state(self, raw_state: Mapping[str, Any], history: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None=None) -> dict[str, np.ndarray]:
        if self.state_spec is None:
            raise ValueError('This transform has no compiled state representation')
        values = self.encode_state_semantic(raw_state, history)
        values = normalize_values(values, mode=self.state_spec.normalization_mode, statistics=self.state_spec.normalization_statistics, semantic_dim=self.state_spec.semantic_dim)
        return {'values': values.astype(self.output_dtype, copy=False)}

    def encode_state_semantic(self, raw_state: Mapping[str, Any], history: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None=None) -> np.ndarray:
        if self.state_spec is None:
            raise ValueError('This transform has no compiled state representation')
        selected_history = None
        if self.state_spec.representation == 'relative':
            selected_history = _select_history(history, self.state_spec.history_offsets[0])
        return encode_state_values(raw_state, selected_history, self.state_spec, dtype=self.compute_dtype)

    def encode_action(self, raw_targets: Sequence[Mapping[str, Any]], raw_anchor: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        values = self.encode_action_semantic(raw_targets, raw_anchor)
        return self.encode_action_semantic_values(values)

    def encode_action_semantic_values(self, semantic_values: object) -> dict[str, np.ndarray]:
        spec = self._transformable_action_spec()
        values = np.asarray(semantic_values, dtype=self.compute_dtype)
        expected = (self.action_horizon, spec.semantic_dim)
        if values.shape != expected:
            raise ValueError(f'Semantic action must have shape {expected}, got {values.shape}')
        if not np.all(np.isfinite(values)):
            raise ValueError('Semantic action contains non-finite values')
        values = normalize_values(values, mode=spec.normalization_mode, statistics=spec.normalization_statistics, semantic_dim=spec.semantic_dim)
        if spec.model_dim > spec.semantic_dim:
            padding = np.zeros((self.action_horizon, spec.model_dim - spec.semantic_dim), dtype=values.dtype)
            values = np.concatenate((values, padding), axis=1)
        return {'values': values.astype(self.output_dtype, copy=False)}

    def encode_action_semantic(self, raw_targets: Sequence[Mapping[str, Any]], raw_anchor: Mapping[str, Any] | None) -> np.ndarray:
        spec = self._transformable_action_spec()
        if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes, Mapping)):
            raise ValueError('raw_targets must be an ordered sequence of canonical mappings')
        if len(raw_targets) != self.action_horizon:
            raise ValueError(f'Action target horizon must be {self.action_horizon}, got {len(raw_targets)}')
        values = encode_action_values(raw_targets, raw_anchor, spec, dtype=self.compute_dtype)
        if values.shape != (self.action_horizon, spec.semantic_dim):
            raise ValueError(f'Encoded action must have shape ({self.action_horizon}, {spec.semantic_dim}), got {values.shape}')
        return values

    def encode_stored_action(self, payload: object, raw_anchor: Mapping[str, Any] | None=None) -> dict[str, np.ndarray]:
        if self.action_source != 'stored':
            raise ValueError('This transform has no compiled stored action contract')
        if isinstance(self.action_spec, RepresentationSpec):
            values = payload.get('values') if isinstance(payload, Mapping) else payload
            array = np.asarray(values, dtype=self.compute_dtype)
            raw_dim = self.action_spec.fields[0].source_shape[0]
            if array.shape != (self.action_horizon, raw_dim):
                raise ValueError(f'Typed stored action must have shape ({self.action_horizon}, {raw_dim}), got {array.shape}')
            return self.encode_action(tuple(({'values': row} for row in array)), raw_anchor)
        if not isinstance(self.action_spec, StoredActionSpec):
            raise ValueError('This transform has no compiled stored action contract')
        values = payload.get('values') if isinstance(payload, Mapping) else payload
        array = np.asarray(values, dtype=self.compute_dtype)
        expected = (self.action_horizon, self.action_spec.semantic_dim)
        if array.shape != expected:
            raise ValueError(f'Stored action must have shape {expected}, got {array.shape}')
        array = normalize_values(array, mode=self.action_spec.normalization_mode, statistics=self.action_spec.normalization_statistics, semantic_dim=self.action_spec.semantic_dim)
        if self.action_spec.model_dim > self.action_spec.semantic_dim:
            array = np.pad(array, ((0, 0), (0, self.action_spec.model_dim - self.action_spec.semantic_dim)))
        return {'values': array.astype(self.output_dtype, copy=False)}

    def decode_action(self, model_values: object, raw_anchor: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        spec = self._transformable_action_spec()
        semantic_values = self.decode_action_semantic(model_values)
        return decode_action_values(semantic_values, raw_anchor, spec, dtype=self.compute_dtype)

    def decode_action_semantic(self, model_values: object) -> np.ndarray:
        spec = self._numeric_action_spec()
        if isinstance(model_values, Mapping):
            if 'values' not in model_values:
                raise ValueError("Action payload mapping must contain 'values'")
            model_values = model_values['values']
        values = np.asarray(model_values, dtype=self.compute_dtype)
        expected_shape = (self.action_horizon, spec.model_dim)
        if values.shape != expected_shape:
            raise ValueError(f'Model action must have shape {expected_shape}, got {values.shape}')
        if not np.all(np.isfinite(values)):
            raise ValueError('Model action contains non-finite values')
        semantic_values = values[:, :spec.semantic_dim]
        semantic_values = unnormalize_values(semantic_values, mode=spec.normalization_mode, statistics=spec.normalization_statistics, semantic_dim=spec.semantic_dim)
        if isinstance(spec, RepresentationSpec):
            semantic_values = canonicalize_semantic_values(semantic_values, spec, dtype=self.compute_dtype)
        return semantic_values.astype(self.output_dtype, copy=False)

    def decode_action_to_absolute_values(self, model_values: object, raw_anchor: Mapping[str, Any] | None, *, output_rotation_encoding: str | None=None) -> np.ndarray:
        spec = self._transformable_action_spec()
        decoded = self.decode_action(model_values, raw_anchor)
        if output_rotation_encoding is not None:
            rows = []
            for raw_row in decoded:
                rows.append(np.concatenate(tuple((flatten_field(read_field(raw_row, field), field, rotation_encoding=output_rotation_encoding if field.kind == 'pose' else None) for field in spec.fields))))
            return np.stack(rows, axis=0).astype(self.output_dtype, copy=False)
        absolute_spec = replace(spec, representation='absolute', normalization_mode='none', normalization_statistics=None, model_dim=spec.semantic_dim, dim_policy='exact')
        return encode_action_values(decoded, None, absolute_spec, dtype=self.compute_dtype).astype(self.output_dtype, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': self.schema_version, 'state_spec': self.state_spec.to_dict() if self.state_spec is not None else None, 'action_spec': self.action_spec.to_dict() if self.action_spec is not None else None, 'action_source': self.action_source, 'action_horizon': self.action_horizon, 'anchor_stride': self.anchor_stride, 'target_start_offset': self.target_start_offset, 'target_step_interval': self.target_step_interval, 'tail_padding': self.tail_padding, 'required_state_sources': list(self.required_state_sources)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompiledStateActionTransform:
        if not isinstance(value, Mapping):
            raise ValueError('Compiled transform artifact must be a mapping')
        schema_version = int(value.get('schema_version', SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f'Unsupported state/action schema_version: {schema_version}')
        state_value = value.get('state_spec')
        action_value = value.get('action_spec')
        if state_value is not None and (not isinstance(state_value, Mapping)):
            raise ValueError('Compiled transform state_spec must be a mapping or null')
        if action_value is not None and (not isinstance(action_value, Mapping)):
            raise ValueError('Compiled transform action_spec must be a mapping or null')
        state_spec = RepresentationSpec.from_dict(state_value) if isinstance(state_value, Mapping) else None
        action_spec = None
        if isinstance(action_value, Mapping):
            action_spec = StoredActionSpec.from_dict(action_value) if action_value.get('role') == 'stored_action' else RepresentationSpec.from_dict(action_value)
        transform = cls(state_spec=state_spec, action_spec=action_spec, action_source=str(value.get('action_source', 'none')), action_horizon=_non_negative_int(value.get('action_horizon', 0), 'action_horizon'), anchor_stride=_positive_int(value.get('anchor_stride', 1), 'anchor_stride'), target_start_offset=_positive_int(value.get('target_start_offset', 1), 'target_start_offset'), target_step_interval=_positive_int(value.get('target_step_interval', 1), 'target_step_interval'), required_state_sources=tuple((str(item) for item in value.get('required_state_sources', []))), tail_padding=str(value.get('tail_padding', 'none')).lower(), schema_version=schema_version)
        transform._validate()
        return transform

    def _transformable_action_spec(self) -> RepresentationSpec:
        if self.action_source not in {'state_derivation', 'stored'} or not isinstance(self.action_spec, RepresentationSpec):
            raise ValueError(f'Action source {self.action_source!r} has no state-derived representation transform')
        return self.action_spec

    def _numeric_action_spec(self) -> RepresentationSpec | StoredActionSpec:
        if self.action_spec is None or self.action_source == 'none':
            raise ValueError(f'Action source {self.action_source!r} has no numeric action contract')
        return self.action_spec

    def _validate(self) -> None:
        if self.state_spec is not None and self.state_spec.role != 'state':
            raise ValueError("Compiled transform state_spec must have role='state'")
        if self.action_source not in {'state_derivation', 'stored', 'none'}:
            raise ValueError(f'Unsupported action source: {self.action_source!r}')
        if self.tail_padding not in {'none', 'repeat_last'}:
            raise ValueError(f'Unsupported action tail padding: {self.tail_padding!r}')
        if self.tail_padding != 'none' and self.action_source != 'state_derivation':
            raise ValueError('Action tail padding is only supported for state-derived actions')
        if self.action_source == 'state_derivation':
            if not isinstance(self.action_spec, RepresentationSpec) or self.action_horizon < 1:
                raise ValueError('state_derivation requires an action spec and a positive action horizon')
            if self.action_spec.role != 'action':
                raise ValueError("Compiled state-derived action_spec must have role='action'")
        elif self.action_source == 'stored':
            if not isinstance(self.action_spec, (StoredActionSpec, RepresentationSpec)) or self.action_horizon < 1:
                raise ValueError('stored requires a stored action spec and a positive action horizon')
            if isinstance(self.action_spec, RepresentationSpec) and self.action_spec.role != 'action':
                raise ValueError("Typed stored action_spec must have role='action'")
        elif self.action_spec is not None:
            raise ValueError("action.source='none' cannot carry an action spec")
        expected_sources = _required_sources(self.state_spec, self.action_spec, action_source=self.action_source)
        if tuple(self.required_state_sources) != expected_sources:
            raise ValueError('required_state_sources does not match the compiled ordered fields')

def compile_state_action_transform(config: Mapping[str, Any], *, model_dim: int | None=None, dim_policy: str='exact', compute_dtype: str | np.dtype='float64', output_dtype: str | np.dtype='float32') -> CompiledStateActionTransform:
    if not isinstance(config, Mapping):
        raise ValueError('State/action transform config must be a mapping')
    pipeline = _pipeline_config(config)
    schema = _optional_mapping(pipeline.get('schema', config.get('schema')), 'schema')
    state_config = pipeline.get('state')
    state_spec: RepresentationSpec | None = None
    if state_config is not None:
        state_mapping = _required_mapping(state_config, 'state')
        state_spec = compile_representation_spec(state_mapping, role='state', schema=schema, model_dim=None, dim_policy='exact')
    action_config = _optional_mapping(pipeline.get('action'), 'action')
    action_source = str(action_config.get('source', 'stored'))
    if action_source not in {'stored', 'state_derivation', 'none'}:
        raise ValueError(f'Unsupported action source: {action_source!r}')
    action_spec: RepresentationSpec | StoredActionSpec | None = None
    action_horizon = 0
    target_start_offset = 1
    target_step_interval = 1
    tail_padding = 'none'
    if action_source == 'state_derivation':
        action_horizon = _resolve_horizon(config, pipeline)
        target_start_offset = _positive_int(action_config.get('target_start_offset', 1), 'action.target_start_offset')
        target_step_interval = _positive_int(action_config.get('target_step_interval', 1), 'action.target_step_interval')
        tail_padding = str(action_config.get('tail_padding', 'none')).lower()
        if tail_padding not in {'none', 'repeat_last'}:
            raise ValueError(f'Unsupported action tail padding: {tail_padding!r}')
        action_spec = compile_representation_spec(action_config, role='action', schema=schema, model_dim=model_dim, dim_policy=dim_policy)
    elif action_source == 'none':
        disallowed = set(action_config) - {'source'}
        if disallowed:
            raise ValueError(f"action.source='none' does not allow other action keys: {sorted(disallowed)}")
    else:
        action_horizon = _resolve_horizon(config, pipeline)
        typed = any((key in action_config for key in ('representation', 'fields', 'rotation')))
        if typed:
            disallowed = set(action_config) - {'source', 'representation', 'pose_delta_axes', 'rotation', 'fields', 'normalization'}
            if disallowed:
                raise ValueError(f'Typed stored action config cannot declare: {sorted(disallowed)}')
            action_spec = compile_typed_stored_action_spec(action_config, schema=schema, model_dim=model_dim, dim_policy=dim_policy)
        else:
            disallowed = set(action_config) - {'source', 'normalization'}
            if disallowed:
                raise ValueError(f'Stored action public config cannot declare: {sorted(disallowed)}')
            action_spec = compile_stored_action_spec(action_config, schema=schema, model_dim=model_dim, dim_policy=dim_policy)
    sampling = _optional_mapping(pipeline.get('sampling'), 'sampling')
    anchor_stride = _positive_int(sampling.get('anchor_stride', 1), 'sampling.anchor_stride')
    transform = CompiledStateActionTransform(state_spec=state_spec, action_spec=action_spec, action_source=action_source, action_horizon=action_horizon, anchor_stride=anchor_stride, target_start_offset=target_start_offset, target_step_interval=target_step_interval, required_state_sources=_required_sources(state_spec, action_spec, action_source=action_source), tail_padding=tail_padding, compute_dtype=np.dtype(compute_dtype).name, output_dtype=np.dtype(output_dtype).name)
    transform._validate()
    return transform

def _pipeline_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config.get('pipeline'), Mapping):
        return config['pipeline']
    data = config.get('data')
    if isinstance(data, Mapping):
        train = data.get('train')
        if isinstance(train, Mapping) and isinstance(train.get('pipeline'), Mapping):
            return train['pipeline']
    return config

def _resolve_horizon(config: Mapping[str, Any], pipeline: Mapping[str, Any]) -> int:
    for value in (config.get('action_horizon'), config.get('horizon'), pipeline.get('action_horizon'), pipeline.get('horizon')):
        if value is not None:
            return _positive_int(value, 'action_horizon')
    model = config.get('model')
    if isinstance(model, Mapping):
        action_head = model.get('action_head')
        if isinstance(action_head, Mapping):
            if action_head.get('action_horizon') is not None:
                return _positive_int(action_head['action_horizon'], 'model.action_head.action_horizon')
            if action_head.get('horizon') is not None:
                return _positive_int(action_head['horizon'], 'model.action_head.horizon')
    raise ValueError('state_derivation requires model.action_head.action_horizon (or resolved action_horizon)')

def _required_sources(state_spec: RepresentationSpec | None, action_spec: RepresentationSpec | StoredActionSpec | None, *, action_source: str) -> tuple[str, ...]:
    result: list[str] = []
    if state_spec is not None:
        for field in state_spec.fields:
            if field.source not in result:
                result.append(field.source)
    if action_spec is not None:
        for field in getattr(action_spec, 'fields', ()):
            source = field.source if action_source == 'state_derivation' else field.anchor_source
            if source is not None and source not in result:
                result.append(source)
    return tuple(result)

def _select_history(history: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None, offset: int) -> Mapping[str, Any] | None:
    if history is None or isinstance(history, Mapping):
        return history
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError('State history must be a canonical mapping, an ordered sequence, or None')
    if len(history) < abs(offset):
        return None
    selected = history[offset]
    if not isinstance(selected, Mapping):
        raise ValueError('Selected state history row must be a mapping')
    return selected

def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping')
    return value

def _optional_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _required_mapping(value, label)

def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f'{label} must be an integer >= 1')
    return int(value)

def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f'{label} must be a non-negative integer')
    return int(value)

def _floating_dtype_name(value: str | np.dtype, label: str) -> str:
    dtype = np.dtype(value)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f'{label} must be a floating-point dtype, got {dtype.name!r}')
    return dtype.name
