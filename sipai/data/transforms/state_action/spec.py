from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from .normalization import NORMALIZATION_MODES, NormalizationStatistics
from .rotation import ROTATION_DIMS, rotation_dim
SCHEMA_VERSION = 1
_FIELD_KINDS = frozenset({'pose', 'joint', 'gripper'})
_DIM_POLICIES = frozenset({'exact', 'pad_to_capacity'})
_STATE_REPRESENTATIONS = frozenset({'absolute', 'relative'})
_ACTION_REPRESENTATIONS = frozenset({'absolute', 'relative', 'delta'})
_POSE_DELTA_AXES = frozenset({'parent', 'anchor', 'parent_position_anchor_rotation'})

@dataclass(frozen=True)
class ComponentLayout:
    name: str
    start: int
    end: int
    unit: str | None = None
    encoding: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {'name': self.name, 'start': self.start, 'end': self.end}
        if self.unit is not None:
            result['unit'] = self.unit
        if self.encoding is not None:
            result['encoding'] = self.encoding
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComponentLayout:
        return cls(name=str(value['name']), start=int(value['start']), end=int(value['end']), unit=_optional_string(value.get('unit')), encoding=_optional_string(value.get('encoding')))

@dataclass(frozen=True)
class FieldSpec:
    name: str
    source: str
    kind: str
    size: int
    source_shape: tuple[int, ...]
    start: int
    end: int
    components: tuple[ComponentLayout, ...]
    selector: tuple[int, int] | None = None
    layout_fields: tuple[str, ...] = ()
    source_rotation_encoding: str | None = None
    units: tuple[str | None, ...] = ()
    anchor_source: str | None = None
    anchor_source_shape: tuple[int, ...] = ()
    anchor_source_rotation_encoding: str | None = None
    anchor_units: tuple[str | None, ...] = ()
    joint_type: tuple[str, ...] = ()
    difference: tuple[str, ...] = ()
    period: tuple[float | None, ...] = ()
    canonical_min: tuple[float | None, ...] = ()
    canonical_max: tuple[float | None, ...] = ()

    @property
    def semantic_dim(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {'name': self.name, 'source': self.source, 'kind': self.kind, 'size': self.size, 'source_shape': list(self.source_shape), 'start': self.start, 'end': self.end, 'components': [component.to_dict() for component in self.components]}
        if self.kind == 'gripper':
            result['effective_representation'] = 'absolute'
        if self.selector is not None:
            result['select'] = {'start': self.selector[0], 'stop': self.selector[1], **({'layout_fields': list(self.layout_fields)} if self.layout_fields else {})}
        if self.source_rotation_encoding is not None:
            result['source_rotation_encoding'] = self.source_rotation_encoding
        if self.units:
            result['unit'] = _collapse_tuple(self.units)
        if self.anchor_source is not None:
            result['anchor'] = {'source': self.anchor_source, 'source_shape': list(self.anchor_source_shape), **({'source_rotation_encoding': self.anchor_source_rotation_encoding} if self.anchor_source_rotation_encoding is not None else {}), **({'unit': _collapse_tuple(self.anchor_units)} if self.anchor_units else {})}
        if self.joint_type:
            result['joint_type'] = _collapse_tuple(self.joint_type)
        if self.difference:
            result['difference'] = _collapse_tuple(self.difference)
        if self.period:
            result['period'] = _collapse_tuple(self.period)
        if self.canonical_min:
            result['canonical_min'] = _collapse_tuple(self.canonical_min)
        if self.canonical_max:
            result['canonical_max'] = _collapse_tuple(self.canonical_max)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FieldSpec:
        size = _positive_int(value.get('size'), 'field.size')
        kind = str(value.get('kind', ''))
        if kind not in _FIELD_KINDS:
            raise ValueError(f'Unsupported resolved field kind: {kind!r}')
        effective_representation = value.get('effective_representation')
        if kind == 'gripper' and effective_representation not in {None, 'absolute'}:
            raise ValueError("Resolved gripper fields must use effective_representation='absolute'")
        select_value = value.get('select')
        selector = _parse_resolved_selector(select_value, label='field.select') if 'select' in value else None
        layout_fields = tuple((str(item) for item in _sequence(select_value.get('layout_fields'), 'field.select.layout_fields'))) if isinstance(select_value, Mapping) and select_value.get('layout_fields') is not None else ()
        anchor = _mapping(value.get('anchor'), 'field.anchor', allow_none=True)
        return cls(name=str(value['name']), source=str(value['source']), kind=kind, size=size, source_shape=tuple((int(item) for item in value.get('source_shape', [size]))), start=int(value['start']), end=int(value['end']), components=tuple((ComponentLayout.from_dict(item) for item in value.get('components', []))), selector=selector, layout_fields=layout_fields, source_rotation_encoding=_optional_string(value.get('source_rotation_encoding')), units=_expand_optional_strings(value.get('unit'), size), anchor_source=_optional_string(anchor.get('source')), anchor_source_shape=tuple((int(item) for item in anchor.get('source_shape', []))), anchor_source_rotation_encoding=_optional_string(anchor.get('source_rotation_encoding')), anchor_units=_expand_optional_strings(anchor.get('unit'), 1 if kind == 'pose' else size) if anchor else (), joint_type=_expand_strings(value.get('joint_type'), size) if kind == 'joint' else (), difference=_expand_strings(value.get('difference'), size) if kind == 'joint' else (), period=_expand_optional_floats(value.get('period'), size) if kind == 'joint' else (), canonical_min=_expand_optional_floats(value.get('canonical_min'), size) if kind == 'joint' else (), canonical_max=_expand_optional_floats(value.get('canonical_max'), size) if kind == 'joint' else ())

@dataclass(frozen=True)
class RepresentationSpec:
    role: str
    representation: str
    rotation_encoding: str | None
    fields: tuple[FieldSpec, ...]
    semantic_dim: int
    model_dim: int
    dim_policy: str
    normalization_mode: str = 'none'
    normalization_statistics: NormalizationStatistics | None = None
    pose_delta_axes: str = 'parent'
    history_offsets: tuple[int, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        normalization: dict[str, Any] = {'mode': self.normalization_mode}
        if self.normalization_statistics is not None:
            normalization['statistics'] = self.normalization_statistics.to_dict()
        return {'schema_version': self.schema_version, 'role': self.role, 'representation': self.representation, 'rotation': {'encoding': self.rotation_encoding} if self.rotation_encoding else None, 'pose_delta_axes': self.pose_delta_axes, **({'history_offsets': list(self.history_offsets)} if self.role == 'state' else {}), 'semantic_dim': self.semantic_dim, 'model_dim': self.model_dim, 'dim_policy': self.dim_policy, 'normalization': normalization, 'fields': [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RepresentationSpec:
        schema_version = int(value.get('schema_version', SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f'Unsupported state/action schema_version: {schema_version}')
        fields = tuple((FieldSpec.from_dict(item) for item in _sequence(value.get('fields'), 'fields')))
        _validate_resolved_layout(fields)
        semantic_dim = int(value['semantic_dim'])
        if not fields or fields[-1].end != semantic_dim:
            raise ValueError('Resolved field layout does not match semantic_dim')
        rotation = _mapping(value.get('rotation'), 'rotation', allow_none=True)
        normalization = _mapping(value.get('normalization'), 'normalization', allow_none=True)
        normalization_mode = str(normalization.get('mode', 'none'))
        statistics_value = normalization.get('statistics')
        statistics = NormalizationStatistics.from_dict(statistics_value) if isinstance(statistics_value, Mapping) else None
        spec = cls(role=str(value['role']), representation=str(value['representation']), rotation_encoding=_optional_string(rotation.get('encoding')), fields=fields, semantic_dim=semantic_dim, model_dim=int(value['model_dim']), dim_policy=str(value['dim_policy']), normalization_mode=normalization_mode, normalization_statistics=statistics, pose_delta_axes=str(value.get('pose_delta_axes', 'parent')), history_offsets=tuple((int(item) for item in value.get('history_offsets', ()))), schema_version=schema_version)
        validate_representation_spec(spec)
        return spec

@dataclass(frozen=True)
class StoredActionSpec:
    semantic_dim: int
    model_dim: int
    dim_policy: str
    normalization_mode: str = 'none'
    normalization_statistics: NormalizationStatistics | None = None
    stored_representation: str | None = None
    stored_layout: Mapping[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        normalization: dict[str, Any] = {'mode': self.normalization_mode}
        if self.normalization_statistics is not None:
            normalization['statistics'] = self.normalization_statistics.to_dict()
        return {'schema_version': self.schema_version, 'role': 'stored_action', 'semantic_dim': self.semantic_dim, 'model_dim': self.model_dim, 'dim_policy': self.dim_policy, 'normalization': normalization, 'stored_representation': self.stored_representation, 'stored_layout': _without_coordinate_frame_metadata(self.stored_layout or {})}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StoredActionSpec:
        normalization = _mapping(value.get('normalization'), 'normalization', allow_none=True)
        statistics_value = normalization.get('statistics')
        spec = cls(semantic_dim=int(value['semantic_dim']), model_dim=int(value['model_dim']), dim_policy=str(value['dim_policy']), normalization_mode=str(normalization.get('mode', 'none')), normalization_statistics=NormalizationStatistics.from_dict(statistics_value) if isinstance(statistics_value, Mapping) else None, stored_representation=_optional_string(value.get('stored_representation')), stored_layout=_without_coordinate_frame_metadata(value['stored_layout']) if isinstance(value.get('stored_layout'), Mapping) else None, schema_version=int(value.get('schema_version', SCHEMA_VERSION)))
        validate_stored_action_spec(spec)
        return spec

def compile_stored_action_spec(config: Mapping[str, Any], *, schema: Mapping[str, Any], model_dim: int | None, dim_policy: str) -> StoredActionSpec:
    action_schema = schema.get('action') if isinstance(schema, Mapping) else None
    if not isinstance(action_schema, Mapping):
        raise ValueError('New stored action path requires canonical schema.action metadata')
    semantic_dim = _stored_action_dim(action_schema)
    resolved_model_dim = semantic_dim if model_dim is None else _positive_int(model_dim, 'action.model_dim')
    normalization = _mapping(config.get('normalization'), 'action.normalization', allow_none=True)
    mode = str(normalization.get('mode', 'none')).lower()
    statistics_value = normalization.get('statistics')
    spec = StoredActionSpec(semantic_dim=semantic_dim, model_dim=resolved_model_dim, dim_policy=dim_policy, normalization_mode=mode, normalization_statistics=NormalizationStatistics.from_dict(statistics_value) if isinstance(statistics_value, Mapping) else None, stored_representation=_optional_string(action_schema.get('representation', action_schema.get('encoding'))), stored_layout=_without_coordinate_frame_metadata(action_schema['layout']) if isinstance(action_schema.get('layout'), Mapping) else None)
    validate_stored_action_spec(spec)
    statistics_spec_value = normalization.get('statistics_spec')
    if statistics_spec_value is not None:
        if not isinstance(statistics_spec_value, Mapping):
            raise ValueError('action.normalization.statistics_spec must be a resolved mapping')
        statistics_spec = StoredActionSpec.from_dict(statistics_spec_value)
        validate_stored_statistics_compatibility(spec, statistics_spec)
    return spec

def validate_stored_action_spec(spec: StoredActionSpec) -> None:
    if spec.schema_version != SCHEMA_VERSION:
        raise ValueError(f'Unsupported state/action schema_version: {spec.schema_version}')
    if spec.dim_policy not in _DIM_POLICIES:
        raise ValueError(f'Unsupported dim_policy: {spec.dim_policy!r}')
    if spec.semantic_dim <= 0 or spec.model_dim <= 0:
        raise ValueError('semantic_dim and model_dim must be positive')
    if spec.dim_policy == 'exact' and spec.semantic_dim != spec.model_dim:
        raise ValueError("dim_policy='exact' requires semantic_dim == model_dim")
    if spec.dim_policy == 'pad_to_capacity' and spec.semantic_dim > spec.model_dim:
        raise ValueError('pad_to_capacity requires semantic_dim <= model_dim')
    if spec.normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f'Unsupported normalization mode: {spec.normalization_mode!r}')
    if spec.normalization_mode != 'none':
        if spec.normalization_statistics is None:
            raise ValueError(f'Normalization mode {spec.normalization_mode!r} requires statistics')
        spec.normalization_statistics.validate(mode=spec.normalization_mode, semantic_dim=spec.semantic_dim)

def validate_stored_statistics_compatibility(current: StoredActionSpec, statistics_spec: StoredActionSpec) -> None:
    current_signature = (current.semantic_dim, current.stored_representation, _stored_layout_semantic_signature(current.stored_layout), current.normalization_mode)
    statistics_signature = (statistics_spec.semantic_dim, statistics_spec.stored_representation, _stored_layout_semantic_signature(statistics_spec.stored_layout), statistics_spec.normalization_mode)
    if current_signature != statistics_signature:
        raise ValueError('Stored action statistics semantics do not match the resolved canonical representation, ordered layout, semantic_dim, or normalization mode')

def _stored_action_dim(action_schema: Mapping[str, Any]) -> int:
    for value in (action_schema.get('dim'), ((action_schema.get('layout') or {}).get('summary') or {}).get('dim') if isinstance(action_schema.get('layout'), Mapping) else None):
        if value is not None:
            return _positive_int(value, 'schema.action.dim')
    shape = action_schema.get('shape')
    if isinstance(shape, Sequence) and (not isinstance(shape, (str, bytes))) and (len(shape) == 1):
        return _positive_int(shape[0], 'schema.action.shape[0]')
    raise ValueError('Canonical schema.action must declare a one-dimensional shape or dim')

def compile_typed_stored_action_spec(config: Mapping[str, Any], *, schema: Mapping[str, Any], model_dim: int | None, dim_policy: str) -> RepresentationSpec:
    action_schema = schema.get('action') if isinstance(schema, Mapping) else None
    if not isinstance(action_schema, Mapping):
        raise ValueError('Typed stored actions require canonical schema.action metadata')
    action_dim = _stored_action_dim(action_schema)
    layout = action_schema.get('layout')
    layout_entries = layout.get('fields') if isinstance(layout, Mapping) else None
    if not isinstance(layout_entries, Sequence) or isinstance(layout_entries, (str, bytes)):
        raise ValueError('Typed stored actions require schema.action.layout.fields')
    by_name: dict[str, Mapping[str, Any]] = {}
    spans_by_name: dict[str, tuple[int, int]] = {}
    declared_order: list[str] = []
    expected_start = 0
    for (index, entry) in enumerate(layout_entries):
        if not isinstance(entry, Mapping) or not str(entry.get('name', '')).strip():
            raise ValueError('schema.action.layout.fields entries require names')
        name = str(entry['name'])
        if name in by_name:
            raise ValueError(f'Duplicate schema.action layout field {name!r}')
        start = entry.get('start')
        stop = entry.get('end')
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or isinstance(stop, bool) or (not isinstance(stop, (int, np.integer))) or (int(start) < 0) or (int(stop) <= int(start)):
            raise ValueError(f'schema.action.layout.fields[{index}] requires integer bounds satisfying 0 <= start < end')
        (start, stop) = (int(start), int(stop))
        if start != expected_start:
            relation = 'overlap' if start < expected_start else 'gap'
            raise ValueError(f'schema.action.layout.fields spans must continuously cover [0, {action_dim}) without gaps or overlaps; {relation} before {name!r}')
        by_name[name] = entry
        spans_by_name[name] = (start, stop)
        declared_order.append(name)
        expected_start = stop
    if expected_start != action_dim:
        raise ValueError(f'schema.action.layout.fields spans must continuously cover [0, {action_dim}); coverage ends at {expected_start}')
    resolved_fields: list[dict[str, Any]] = []
    selected_names: list[str] = []
    numeric_spans: list[tuple[int, int]] = []
    selection_mode: str | None = None
    for (index, raw_field) in enumerate(_sequence(config.get('fields'), 'action.fields')):
        field = dict(_mapping(raw_field, f'action.fields[{index}]'))
        if str(field.get('source', '')) != 'values':
            raise ValueError("Typed stored action fields require source='values'")
        select = _mapping(field.get('select'), f'action.fields[{index}].select')
        select_keys = set(select)
        if select_keys == {'layout_fields'}:
            if selection_mode == 'numeric':
                raise ValueError('Typed stored action fields cannot mix numeric and layout selectors')
            selection_mode = 'layout'
            names = tuple((str(item) for item in _sequence(select.get('layout_fields'), f'action.fields[{index}].select.layout_fields')))
            if not names or any((not name for name in names)):
                raise ValueError('Typed stored action layout_fields must be non-empty strings')
            try:
                entries = [by_name[name] for name in names]
            except KeyError as exc:
                raise ValueError(f'Unknown schema.action layout field {exc.args[0]!r}') from exc
            starts = [spans_by_name[name][0] for name in names]
            stops = [spans_by_name[name][1] for name in names]
            if any((left != right for (left, right) in zip(stops[:-1], starts[1:], strict=True))):
                raise ValueError('Typed stored action layout_fields must form one contiguous ordered slice')
            if any((name in selected_names for name in names)):
                raise ValueError('Typed stored action layout_fields cannot overlap')
            selected_names.extend(names)
        elif select_keys == {'start', 'stop'}:
            if selection_mode == 'layout':
                raise ValueError('Typed stored action fields cannot mix numeric and layout selectors')
            selection_mode = 'numeric'
            (start, stop) = (select.get('start'), select.get('stop'))
            if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or isinstance(stop, bool) or (not isinstance(stop, (int, np.integer))) or (int(start) < 0) or (int(stop) <= int(start)) or (int(stop) > action_dim):
                raise ValueError('Typed stored numeric selectors require 0 <= start < stop <= action_dim')
            (starts, stops, names, entries) = ([int(start)], [int(stop)], (), [])
            numeric_spans.append((int(start), int(stop)))
        else:
            raise ValueError('Typed stored action select requires layout_fields or numeric start/stop')
        kind = str(field.get('kind', ''))
        metadata = dict(_mapping(field.get('metadata'), f'action.fields[{index}].metadata', allow_none=True))
        declared_shape = metadata.get('shape')
        if declared_shape is not None:
            (_, target_shape) = _field_size_and_shape(metadata, kind=kind, name=str(field.get('name')))
            if target_shape != (action_dim,):
                raise ValueError(f"Typed stored action source='values' shape must be ({action_dim},)")
        metadata['shape'] = [action_dim]
        if metadata.get('unit') is None and names:
            metadata['unit'] = _typed_stored_target_unit(action_schema.get('units'), kind=kind, layout_fields=names)
        if kind == 'gripper' and metadata.get('unit') is None:
            raise ValueError('Typed stored gripper fields require target unit metadata from field metadata or schema.action.units')
        if kind == 'pose' and metadata.get('source_rotation_encoding') is None:
            encoding = str(action_schema.get('encoding', action_schema.get('representation', '')))
            if 'xyzw' in encoding:
                metadata['source_rotation_encoding'] = 'quat_xyzw'
        anchor_source = field.get('anchor_source')
        representation = str(config.get('representation', 'absolute'))
        if kind == 'gripper':
            if anchor_source is not None:
                raise ValueError('Typed stored gripper fields cannot declare anchor_source')
        elif representation != 'absolute':
            if not isinstance(anchor_source, str) or not anchor_source.strip():
                raise ValueError(f'Typed stored {kind} fields require anchor_source')
            anchor_metadata = dict(_lookup_field_metadata(schema, source=anchor_source, name=str(field.get('name', '')), role='state'))
            if not anchor_metadata:
                raise ValueError(f'Typed stored anchor_source {anchor_source!r} is missing from schema.state')
            if selection_mode == 'numeric':
                (_, anchor_shape) = _field_size_and_shape(anchor_metadata, kind=kind, name=str(field.get('name', '')))
                anchor_encoding = _source_rotation_encoding(metadata) if kind == 'pose' else None
                anchor_unit = metadata.get('unit')
            elif kind == 'pose':
                anchor_encoding = _source_rotation_encoding(anchor_metadata)
                anchor_shape = (3 + rotation_dim(anchor_encoding),)
                anchor_unit = _single_unit(anchor_metadata.get('unit'))
            else:
                (anchor_size, anchor_shape) = _field_size_and_shape(anchor_metadata, kind=kind, name=str(field.get('name', '')))
                if anchor_size != stops[-1] - starts[0]:
                    raise ValueError('Typed stored joint target and anchor dimensions do not match')
                anchor_encoding = None
                anchor_unit = _collapse_tuple(_expand_optional_strings(anchor_metadata.get('unit'), anchor_size))
            target_unit = metadata.get('unit')
            if target_unit is None or anchor_unit is None:
                raise ValueError('Typed stored relative pose/joint fields require target and anchor unit metadata')
            metadata.update({'anchor_source': anchor_source, 'anchor_source_shape': list(anchor_shape), 'anchor_source_rotation_encoding': anchor_encoding, 'anchor_unit': anchor_unit})
        field['metadata'] = metadata
        field['select'] = {'start': starts[0], 'stop': stops[-1], **({'layout_fields': list(names)} if names else {})}
        resolved_fields.append(field)
    if selection_mode == 'numeric':
        expected_start = 0
        for (start, stop) in numeric_spans:
            if start != expected_start:
                raise ValueError('Typed stored numeric selectors must continuously cover the action vector')
            expected_start = stop
        if expected_start != action_dim:
            raise ValueError('Typed stored numeric selectors must continuously cover the action vector')
    elif selected_names != declared_order:
        raise ValueError('Typed stored action fields must cover schema.action.layout.fields exactly in declared order')
    resolved = dict(config)
    resolved['fields'] = resolved_fields
    spec = compile_representation_spec(resolved, role='action', schema={}, model_dim=model_dim, dim_policy=dim_policy)
    return spec

def _typed_stored_target_unit(action_units: Any, *, kind: str, layout_fields: Sequence[str]) -> Any:
    if action_units is None or isinstance(action_units, str):
        return action_units
    if not isinstance(action_units, Mapping):
        raise ValueError('schema.action.units must be a string or mapping')
    keys_by_kind = {'pose': ('xyz', 'position', 'translation', 'pose', 'value'), 'gripper': ('gripper', 'width', 'value'), 'joint': ('joint', 'joints', 'value')}
    for key in keys_by_kind.get(kind, ('value',)):
        if action_units.get(key) is not None:
            return action_units[key]
    selected = [action_units.get(name) for name in layout_fields]
    if selected and all((unit is not None for unit in selected)):
        first = selected[0]
        if all((unit == first for unit in selected[1:])):
            return first
        raise ValueError('schema.action.units for one typed stored field must resolve to one unit')
    return None

def compile_representation_spec(config: Mapping[str, Any], *, role: str, schema: Mapping[str, Any] | None=None, model_dim: int | None=None, dim_policy: str='exact') -> RepresentationSpec:
    if role not in {'state', 'action'}:
        raise ValueError(f'Unsupported representation role: {role!r}')
    allowed_representations = _STATE_REPRESENTATIONS if role == 'state' else _ACTION_REPRESENTATIONS
    representation = str(config.get('representation', 'absolute'))
    if representation not in allowed_representations:
        raise ValueError(f'Unsupported {role} representation: {representation!r}')
    pose_delta_axes = str(config.get('pose_delta_axes', 'parent'))
    if pose_delta_axes not in _POSE_DELTA_AXES:
        raise ValueError(f"Unsupported {role} pose_delta_axes: {pose_delta_axes!r}; expected one of 'parent', 'anchor', or 'parent_position_anchor_rotation'")
    history_offsets: tuple[int, ...] = ()
    if role == 'state' and representation == 'relative':
        raw_history_offsets = config.get('history_offsets')
        if not isinstance(raw_history_offsets, Sequence) or isinstance(raw_history_offsets, (str, bytes)) or len(raw_history_offsets) != 1 or isinstance(raw_history_offsets[0], (bool, np.bool_)) or (not isinstance(raw_history_offsets[0], (int, np.integer))) or (int(raw_history_offsets[0]) >= 0):
            raise ValueError('state.history_offsets must contain exactly one negative integer')
        history_offsets = (int(raw_history_offsets[0]),)
    normalization = _mapping(config.get('normalization'), f'{role}.normalization', allow_none=True)
    normalization_mode = str(normalization.get('mode', 'none')).lower()
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f'Unsupported {role} normalization mode: {normalization_mode!r}')
    statistics_value = normalization.get('statistics')
    normalization_statistics = NormalizationStatistics.from_dict(statistics_value) if isinstance(statistics_value, Mapping) else None
    if normalization_mode != 'none' and normalization_statistics is None:
        raise ValueError(f'{role} normalization mode {normalization_mode!r} requires statistics')
    field_configs = _sequence(config.get('fields'), f'{role}.fields')
    if not field_configs:
        raise ValueError(f'{role}.fields must be a non-empty ordered list')
    names: set[str] = set()
    resolved_fields: list[FieldSpec] = []
    has_pose = any((isinstance(item, Mapping) and item.get('kind') == 'pose' for item in field_configs))
    rotation_config = _mapping(config.get('rotation'), f'{role}.rotation', allow_none=True)
    if rotation_config and (not has_pose):
        raise ValueError(f'{role}.rotation is only valid when {role}.fields contains a pose')
    rotation_encoding = str(rotation_config.get('encoding', 'quat_xyzw')) if has_pose else None
    if rotation_encoding is not None and rotation_encoding not in ROTATION_DIMS:
        raise ValueError(f'Unsupported rotation encoding: {rotation_encoding!r}')
    offset = 0
    allow_numeric_action_select = role == 'action' and str(config.get('source', '')) in {'state_derivation', 'stored'}
    for (index, raw_field) in enumerate(field_configs):
        field_config = _mapping(raw_field, f'{role}.fields[{index}]')
        name = str(field_config.get('name', '')).strip()
        source = str(field_config.get('source', '')).strip()
        kind = str(field_config.get('kind', '')).strip()
        if not name or not source:
            raise ValueError(f'{role}.fields[{index}] requires non-empty name and source')
        if name in names:
            raise ValueError(f'Duplicate {role} field name: {name!r}')
        if kind not in _FIELD_KINDS:
            raise ValueError(f'Unsupported field kind for {name!r}: {kind!r}')
        names.add(name)
        metadata = dict(_lookup_field_metadata(schema or {}, source=source, name=name, role=role))
        inline_metadata = _mapping(field_config.get('metadata'), f'{role}.fields[{index}].metadata', allow_none=True)
        metadata.update(inline_metadata)
        metadata.update({key: value for (key, value) in field_config.items() if key != 'metadata'})
        field = _compile_field(name=name, source=source, kind=kind, role=role, metadata=metadata, target_rotation_encoding=rotation_encoding, offset=offset, allow_numeric_action_select=allow_numeric_action_select)
        if kind == 'joint' and representation != 'absolute':
            _validate_joint_delta_metadata(field)
        resolved_fields.append(field)
        offset = field.end
    resolved_model_dim = offset if model_dim is None else _positive_int(model_dim, f'{role}.model_dim')
    if dim_policy not in _DIM_POLICIES:
        raise ValueError(f'Unsupported dim_policy: {dim_policy!r}')
    if dim_policy == 'exact' and offset != resolved_model_dim:
        raise ValueError(f"{role} semantic_dim ({offset}) must equal model_dim ({resolved_model_dim}) for dim_policy='exact'")
    if dim_policy == 'pad_to_capacity' and offset > resolved_model_dim:
        raise ValueError(f'{role} semantic_dim ({offset}) exceeds model capacity ({resolved_model_dim})')
    spec = RepresentationSpec(role=role, representation=representation, rotation_encoding=rotation_encoding, fields=tuple(resolved_fields), semantic_dim=offset, model_dim=resolved_model_dim, dim_policy=dim_policy, normalization_mode=normalization_mode, normalization_statistics=normalization_statistics, pose_delta_axes=pose_delta_axes, history_offsets=history_offsets)
    validate_representation_spec(spec)
    statistics_spec_value = normalization.get('statistics_spec')
    if statistics_spec_value is not None:
        if not isinstance(statistics_spec_value, Mapping):
            raise ValueError(f'{role}.normalization.statistics_spec must be a resolved mapping')
        statistics_spec = RepresentationSpec.from_dict(statistics_spec_value)
        validate_representation_statistics_compatibility(spec, statistics_spec)
    return spec

def validate_representation_spec(spec: RepresentationSpec) -> None:
    if spec.schema_version != SCHEMA_VERSION:
        raise ValueError(f'Unsupported state/action schema_version: {spec.schema_version}')
    if spec.role not in {'state', 'action'}:
        raise ValueError(f'Unsupported resolved role: {spec.role!r}')
    allowed = _STATE_REPRESENTATIONS if spec.role == 'state' else _ACTION_REPRESENTATIONS
    if spec.representation not in allowed:
        raise ValueError(f'Unsupported {spec.role} representation: {spec.representation!r}')
    if spec.pose_delta_axes not in _POSE_DELTA_AXES:
        raise ValueError("pose_delta_axes must be 'parent', 'anchor', or 'parent_position_anchor_rotation'")
    if spec.role == 'state' and spec.representation == 'relative':
        if len(spec.history_offsets) != 1 or isinstance(spec.history_offsets[0], (bool, np.bool_)) or (not isinstance(spec.history_offsets[0], (int, np.integer))) or (int(spec.history_offsets[0]) >= 0):
            raise ValueError('Relative state history_offsets must contain exactly one negative integer')
    elif spec.history_offsets:
        raise ValueError('history_offsets are only valid for relative state representations')
    if spec.normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f'Unsupported normalization mode: {spec.normalization_mode!r}')
    if spec.dim_policy not in _DIM_POLICIES:
        raise ValueError(f'Unsupported dim_policy: {spec.dim_policy!r}')
    if spec.semantic_dim <= 0 or spec.model_dim <= 0:
        raise ValueError('semantic_dim and model_dim must be positive')
    if spec.dim_policy == 'exact' and spec.semantic_dim != spec.model_dim:
        raise ValueError("dim_policy='exact' requires semantic_dim == model_dim")
    if spec.dim_policy == 'pad_to_capacity' and spec.semantic_dim > spec.model_dim:
        raise ValueError('pad_to_capacity requires semantic_dim <= model_dim')
    if spec.normalization_mode != 'none':
        if spec.normalization_statistics is None:
            raise ValueError(f'Normalization mode {spec.normalization_mode!r} requires statistics')
        spec.normalization_statistics.validate(mode=spec.normalization_mode, semantic_dim=spec.semantic_dim)
    elif spec.normalization_statistics is not None:
        spec.normalization_statistics.validate_available_shapes(spec.semantic_dim)
    _validate_resolved_layout(spec.fields)
    _validate_resolved_field_contracts(spec)

def validate_representation_statistics_compatibility(current: RepresentationSpec, statistics_spec: RepresentationSpec) -> None:
    if _representation_semantic_signature(current) != _representation_semantic_signature(statistics_spec):
        raise ValueError(f'{current.role} statistics semantics do not match the resolved canonical ordered fields/layout, representation, rotation encoding, units, semantic_dim, pose_delta_axes, history_offsets, or normalization mode')

def _representation_semantic_signature(spec: RepresentationSpec) -> tuple[Any, ...]:
    fields = tuple(((field.name, field.kind, field.start, field.end, field.selector, tuple(field.layout_fields), tuple(((component.name, component.start, component.end, component.unit, component.encoding) for component in field.components)), tuple(field.units), field.source_rotation_encoding, field.anchor_source, field.anchor_source_shape, field.anchor_source_rotation_encoding, tuple(field.anchor_units), tuple(field.joint_type), tuple(field.difference), tuple(field.period), tuple(field.canonical_min), tuple(field.canonical_max)) for field in spec.fields))
    return (spec.role, spec.representation, spec.rotation_encoding, spec.semantic_dim, spec.pose_delta_axes, spec.history_offsets, spec.normalization_mode, fields)

def _stored_layout_semantic_signature(layout: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    if not isinstance(layout, Mapping):
        return None
    summary = layout.get('summary')
    fields = layout.get('fields')
    field_signature = tuple((_stored_layout_field_signature(field) for field in fields if isinstance(field, Mapping))) if isinstance(fields, Sequence) and (not isinstance(fields, (str, bytes))) else ()
    return (_stored_layout_selected_values(layout), _stored_layout_selected_values(summary) if isinstance(summary, Mapping) else (), field_signature)

def _without_coordinate_frame_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_coordinate_frame_metadata(item) for (key, item) in value.items() if key not in {'coordinate_frame', 'anchor_coordinate_frame', 'frame'}}
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return [_without_coordinate_frame_metadata(item) for item in value]
    return value

def _stored_layout_field_signature(field: Mapping[str, Any]) -> tuple[Any, ...]:
    components = field.get('components')
    component_signature = tuple((_stored_layout_field_signature(component) for component in components if isinstance(component, Mapping))) if isinstance(components, Sequence) and (not isinstance(components, (str, bytes))) else ()
    return (_stored_layout_selected_values(field), component_signature)

def _stored_layout_selected_values(value: Mapping[str, Any]) -> tuple[Any, ...]:
    semantic_keys = ('name', 'kind', 'start', 'end', 'unit', 'units', 'encoding', 'rotation_encoding', 'source_rotation_encoding', 'representation', 'effective_representation', 'pose_delta_axes')
    return tuple(((key, _freeze_semantic_value(value[key])) for key in semantic_keys if key in value))

def _freeze_semantic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted(((str(key), _freeze_semantic_value(item)) for (key, item) in value.items())))
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return tuple((_freeze_semantic_value(item) for item in value))
    return value

def _compile_field(*, name: str, source: str, kind: str, role: str, metadata: Mapping[str, Any], target_rotation_encoding: str | None, offset: int, allow_numeric_action_select: bool) -> FieldSpec:
    unit_value = metadata.get('unit')
    select_value = metadata.get('select')
    layout_fields = tuple((str(item) for item in select_value.get('layout_fields', ()))) if isinstance(select_value, Mapping) else ()
    anchor_source = _optional_string(metadata.get('anchor_source'))
    anchor_source_shape = tuple((int(item) for item in metadata.get('anchor_source_shape', ())))
    anchor_source_rotation_encoding = _optional_string(metadata.get('anchor_source_rotation_encoding'))
    anchor_units_value = metadata.get('anchor_unit')
    if kind == 'pose':
        if 'select' in metadata and (not (role == 'state' or allow_numeric_action_select or layout_fields)):
            raise ValueError('select is not supported for pose fields')
        source_encoding = _source_rotation_encoding(metadata)
        source_dim = 3 + rotation_dim(source_encoding)
        if 'select' in metadata:
            (raw_size, source_shape) = _field_size_and_shape(metadata, kind=kind, name=name)
            selector = _compile_selector(select_value, role=role, kind=kind, source_size=raw_size, label=f'field {name!r}.select', allow_numeric_action_select=allow_numeric_action_select)
            if selector[1] - selector[0] != source_dim:
                raise ValueError(f'Pose field {name!r} selected layout must contain {source_dim} source values')
        else:
            _validate_declared_shape(metadata.get('shape'), expected_dim=source_dim, label=f'pose field {name!r}')
            source_shape = (source_dim,)
            selector = None
        target_encoding = target_rotation_encoding or 'quat_xyzw'
        target_dim = 3 + rotation_dim(target_encoding)
        position_unit = _single_unit(unit_value)
        components = (ComponentLayout('position', 0, 3, unit=position_unit), ComponentLayout('rotation', 3, target_dim, encoding=target_encoding))
        return FieldSpec(name=name, source=source, kind=kind, size=1, source_shape=source_shape, start=offset, end=offset + target_dim, components=components, selector=selector, layout_fields=layout_fields, source_rotation_encoding=source_encoding, units=(position_unit,), anchor_source=anchor_source, anchor_source_shape=anchor_source_shape, anchor_source_rotation_encoding=anchor_source_rotation_encoding, anchor_units=_expand_optional_strings(anchor_units_value, 1) if anchor_source is not None else ())
    (raw_size, source_shape) = _field_size_and_shape(metadata, kind=kind, name=name)
    selector = _compile_selector(metadata.get('select'), role=role, kind=kind, source_size=raw_size, label=f'field {name!r}.select', allow_numeric_action_select=allow_numeric_action_select) if 'select' in metadata else None
    selection = slice(*selector) if selector is not None else slice(None)
    size = selector[1] - selector[0] if selector is not None else raw_size
    units = _expand_optional_strings(unit_value, raw_size)[selection]
    component_name = 'joint' if kind == 'joint' else 'value'
    components = (ComponentLayout(component_name, 0, size, unit=_uniform_or_none(units)),)
    if kind == 'gripper':
        return FieldSpec(name=name, source=source, kind=kind, size=size, source_shape=source_shape, start=offset, end=offset + size, components=components, selector=selector, layout_fields=layout_fields, units=units, anchor_source=anchor_source, anchor_source_shape=anchor_source_shape, anchor_units=_expand_optional_strings(anchor_units_value, size) if anchor_source is not None else ())
    joint_type = _expand_strings(metadata.get('joint_type'), raw_size)[selection]
    difference = _expand_strings(metadata.get('difference'), raw_size)[selection]
    period = _expand_optional_floats(metadata.get('period'), raw_size)[selection]
    (raw_canonical_min, raw_canonical_max) = _canonical_ranges(metadata, raw_size)
    canonical_min = raw_canonical_min[selection]
    canonical_max = raw_canonical_max[selection]
    return FieldSpec(name=name, source=source, kind=kind, size=size, source_shape=source_shape, start=offset, end=offset + size, components=components, selector=selector, layout_fields=layout_fields, units=units, anchor_source=anchor_source, anchor_source_shape=anchor_source_shape, anchor_source_rotation_encoding=anchor_source_rotation_encoding, anchor_units=_expand_optional_strings(anchor_units_value, size) if anchor_source is not None else (), joint_type=joint_type, difference=difference, period=period, canonical_min=canonical_min, canonical_max=canonical_max)

def _validate_joint_delta_metadata(field: FieldSpec) -> None:
    size = field.size
    if len(field.units) != size or any((unit is None or not unit.strip() for unit in field.units)):
        raise ValueError(f'Joint field {field.name!r} requires unit metadata for relative/delta arithmetic')
    if len(field.joint_type) != size or len(field.difference) != size:
        raise ValueError(f'Joint field {field.name!r} requires joint_type and difference metadata for relative/delta arithmetic')
    _validate_joint_difference_metadata(field, require_joint_type=True)

def _validate_joint_difference_metadata(field: FieldSpec, *, require_joint_type: bool) -> None:
    size = field.size
    if len(field.difference) != size:
        raise ValueError(f'Joint field {field.name!r} difference metadata must contain {size} values')
    if len(field.period) != size or len(field.canonical_min) != size or len(field.canonical_max) != size:
        raise ValueError(f'Joint field {field.name!r} periodic metadata must contain {size} values')
    if require_joint_type and len(field.joint_type) != size:
        raise ValueError(f'Joint field {field.name!r} joint_type metadata must contain {size} values')
    if field.joint_type and len(field.joint_type) != size:
        raise ValueError(f'Joint field {field.name!r} joint_type metadata must contain {size} values')
    joint_types = field.joint_type if field.joint_type else tuple(('' for _ in range(size)))
    for (index, (joint_type, difference)) in enumerate(zip(joint_types, field.difference, strict=True)):
        if require_joint_type and joint_type not in {'revolute', 'prismatic'}:
            raise ValueError(f'Joint field {field.name!r} channel {index} has invalid joint_type {joint_type!r}')
        if joint_type and joint_type not in {'revolute', 'prismatic'}:
            raise ValueError(f'Joint field {field.name!r} channel {index} has invalid joint_type {joint_type!r}')
        if difference not in {'linear', 'shortest_periodic'}:
            raise ValueError(f'Joint field {field.name!r} channel {index} has invalid difference {difference!r}')
        if joint_type == 'prismatic' and difference == 'shortest_periodic':
            raise ValueError(f'Prismatic joint field {field.name!r} cannot use shortest_periodic difference')
        if difference == 'shortest_periodic':
            channel_period = field.period[index] if len(field.period) == size else None
            lower = field.canonical_min[index] if len(field.canonical_min) == size else None
            upper = field.canonical_max[index] if len(field.canonical_max) == size else None
            if channel_period is None or not np.isfinite(channel_period) or channel_period <= 0.0:
                raise ValueError(f'Periodic joint field {field.name!r} channel {index} requires positive period')
            if lower is None or upper is None or (not np.isfinite([lower, upper]).all()) or (upper <= lower):
                raise ValueError(f'Periodic joint field {field.name!r} channel {index} requires a finite canonical range')
            if not np.isclose(upper - lower, channel_period, rtol=1e-09, atol=1e-12):
                raise ValueError(f'Periodic joint field {field.name!r} channel {index} canonical range must span one period')

def _validate_resolved_layout(fields: tuple[FieldSpec, ...]) -> None:
    if not fields:
        raise ValueError('Resolved fields must be non-empty')
    expected_start = 0
    names: set[str] = set()
    for field in fields:
        if not field.name.strip() or not field.source.strip():
            raise ValueError('Resolved fields require non-empty name and source')
        if field.kind not in _FIELD_KINDS:
            raise ValueError(f'Unsupported resolved field kind: {field.kind!r}')
        if field.name in names:
            raise ValueError(f'Duplicate resolved field name: {field.name!r}')
        names.add(field.name)
        if field.start != expected_start or field.end <= field.start:
            raise ValueError(f'Resolved layout for field {field.name!r} is not contiguous')
        expected_start = field.end

def _validate_resolved_field_contracts(spec: RepresentationSpec) -> None:
    pose_fields = tuple((field for field in spec.fields if field.kind == 'pose'))
    if pose_fields:
        if spec.rotation_encoding not in ROTATION_DIMS:
            raise ValueError('Resolved pose fields require a supported target rotation encoding')
    elif spec.rotation_encoding is not None:
        raise ValueError('Resolved rotation encoding is only valid when fields contain a pose')
    for field in spec.fields:
        typed_stored_field = spec.role == 'action' and bool(field.layout_fields)
        if field.selector is not None and spec.role not in {'state', 'action'}:
            raise ValueError('select is only supported for state fields')
        if field.layout_fields and (not typed_stored_field):
            raise ValueError('layout_fields selection is only supported for typed stored action fields')
        if typed_stored_field and field.selector is None:
            raise ValueError('Typed stored action fields require a resolved layout selector')
        if field.kind == 'gripper' and field.anchor_source is not None:
            raise ValueError('Gripper fields cannot declare anchor_source')
        if typed_stored_field and spec.representation != 'absolute' and (field.kind != 'gripper'):
            if field.anchor_source is None:
                raise ValueError(f'Typed stored {field.kind} field {field.name!r} requires anchor_source')
        if field.anchor_source is not None:
            _validate_anchor_contract(field)
        if field.kind == 'pose':
            _validate_resolved_pose_field(field, target_rotation_encoding=spec.rotation_encoding)
            continue
        if field.source_rotation_encoding is not None:
            raise ValueError(f'Non-pose field {field.name!r} cannot declare source_rotation_encoding')
        if field.kind == 'joint':
            if spec.representation != 'absolute':
                _validate_joint_delta_metadata(field)
            elif field.difference:
                _validate_joint_difference_metadata(field, require_joint_type=False)
        _validate_resolved_flat_field(field)

def _validate_resolved_pose_field(field: FieldSpec, *, target_rotation_encoding: str | None) -> None:
    source_encoding = field.source_rotation_encoding
    if source_encoding not in ROTATION_DIMS:
        raise ValueError(f'Pose field {field.name!r} requires a supported source_rotation_encoding')
    if target_rotation_encoding not in ROTATION_DIMS:
        raise ValueError(f'Pose field {field.name!r} requires a supported target rotation encoding')
    source_dim = 3 + rotation_dim(source_encoding)
    expected_source_shape = (source_dim,)
    source_layout_valid = field.source_shape == expected_source_shape
    if field.selector is not None:
        source_size = field.source_shape[0] if len(field.source_shape) == 1 else 0
        (start, stop) = field.selector
        source_layout_valid = 0 <= start < stop <= source_size and stop - start == source_dim
    if field.size != 1 or not source_layout_valid:
        raise ValueError(f'Pose field {field.name!r} source_shape must be {expected_source_shape} for {source_encoding}')
    target_dim = 3 + rotation_dim(target_rotation_encoding)
    if field.semantic_dim != target_dim:
        raise ValueError(f'Pose field {field.name!r} semantic layout must contain {target_dim} values for {target_rotation_encoding}')
    expected_unit = field.units[0] if len(field.units) == 1 else None
    expected_components = (('position', 0, 3, expected_unit, None), ('rotation', 3, target_dim, None, target_rotation_encoding))
    _validate_component_layout(field, expected_components)

def _validate_anchor_contract(field: FieldSpec) -> None:
    if not field.anchor_source_shape:
        raise ValueError(f'Field {field.name!r} anchor_source requires source_shape metadata')
    if field.kind == 'pose':
        encoding = field.anchor_source_rotation_encoding
        if encoding not in ROTATION_DIMS:
            raise ValueError(f'Pose field {field.name!r} anchor requires a supported rotation encoding')
        expected = (3 + rotation_dim(encoding),)
        selected_anchor_valid = False
        if field.selector is not None and (not field.layout_fields):
            source_size = field.anchor_source_shape[0] if len(field.anchor_source_shape) == 1 else 0
            (start, stop) = field.selector
            selected_anchor_valid = 0 <= start < stop <= source_size and stop - start == expected[0]
        if field.anchor_source_shape != expected and (not selected_anchor_valid):
            raise ValueError(f'Pose field {field.name!r} anchor source_shape must be {expected}')
    elif field.anchor_source_shape != (field.size,):
        selected_anchor_valid = False
        if field.selector is not None and (not field.layout_fields):
            source_size = field.anchor_source_shape[0] if len(field.anchor_source_shape) == 1 else 0
            (start, stop) = field.selector
            selected_anchor_valid = 0 <= start < stop <= source_size and stop - start == field.size
        if not selected_anchor_valid:
            raise ValueError(f'Field {field.name!r} anchor source_shape must be ({field.size},)')
    if field.units and field.anchor_units and (tuple(field.units) != tuple(field.anchor_units)):
        raise ValueError(f'Field {field.name!r} target and anchor units do not match')

def _validate_resolved_flat_field(field: FieldSpec) -> None:
    if len(field.source_shape) > 1 or any((item <= 0 for item in field.source_shape)):
        raise ValueError(f'Resolved field {field.name!r} source_shape must be scalar or a flat vector')
    source_size = field.source_shape[0] if field.source_shape else 1
    expected_size = source_size
    if field.selector is not None:
        (start, stop) = field.selector
        if start < 0 or start >= stop or stop > source_size:
            raise ValueError(f'Resolved field {field.name!r} select must satisfy 0 <= start < stop <= {source_size}')
        expected_size = stop - start
    if expected_size != field.size or field.semantic_dim != field.size:
        raise ValueError(f'Resolved field {field.name!r} size, source_shape, and semantic layout disagree')
    component_name = 'joint' if field.kind == 'joint' else 'value'
    _validate_component_layout(field, ((component_name, 0, field.size, _uniform_or_none(field.units), None),))

def _validate_component_layout(field: FieldSpec, expected: tuple[tuple[str, int, int, str | None, str | None], ...]) -> None:
    if len(field.components) != len(expected):
        raise ValueError(f'Resolved component layout for field {field.name!r} is inconsistent')
    for (component, expected_values) in zip(field.components, expected, strict=True):
        actual = (component.name, component.start, component.end, component.unit, component.encoding)
        if actual != expected_values:
            raise ValueError(f'Resolved component layout for field {field.name!r} is inconsistent')

def _lookup_field_metadata(schema: Mapping[str, Any], *, source: str, name: str, role: str) -> Mapping[str, Any]:
    if not schema:
        return {}
    candidates: list[Any] = []
    schema_sections: list[Any] = [schema.get('state')]
    if role != 'state':
        schema_sections.append(schema.get(role))
    for section in schema_sections:
        if isinstance(section, Mapping):
            exact_prefixes = [key for key in section if isinstance(key, str) and (source == key or source.startswith(f'{key}.'))]
            if exact_prefixes:
                longest = max(exact_prefixes, key=len)
                candidate = section[longest]
                if isinstance(candidate, Mapping):
                    candidate = _canonical_reader_field_metadata(candidate, source_suffix=source[len(longest):].lstrip('.'))
                candidates.append(candidate)
    for collection_key in ('fields', 'sources', 'state_fields'):
        collection = schema.get(collection_key)
        if isinstance(collection, Mapping):
            candidates.extend((collection.get(source), collection.get(name)))
        elif isinstance(collection, Sequence) and (not isinstance(collection, (str, bytes))):
            for entry in collection:
                if isinstance(entry, Mapping) and (entry.get('source') == source or entry.get('name') == name):
                    candidates.append(entry)
    candidates.extend((schema.get(source), schema.get(name)))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}

def _canonical_reader_field_metadata(metadata: Mapping[str, Any], *, source_suffix: str) -> Mapping[str, Any]:
    if not source_suffix:
        return metadata
    resolved = dict(metadata)
    if source_suffix == 'pose':
        resolved['shape'] = [7]
        units = metadata.get('units')
        if isinstance(units, Mapping) and units.get('xyz') is not None:
            resolved.setdefault('unit', units['xyz'])
    elif source_suffix == 'position':
        resolved['shape'] = [3]
        units = metadata.get('units')
        if isinstance(units, Mapping) and units.get('xyz') is not None:
            resolved.setdefault('unit', units['xyz'])
    elif source_suffix == 'rotation_xyzw':
        resolved['shape'] = [4]
    elif source_suffix == 'width':
        resolved['shape'] = []
        units = metadata.get('units')
        if isinstance(units, Mapping) and units.get('width') is not None:
            resolved.setdefault('unit', units['width'])
    return resolved

def _source_rotation_encoding(metadata: Mapping[str, Any]) -> str:
    direct = metadata.get('source_rotation_encoding')
    if direct is None:
        source_rotation = metadata.get('source_rotation')
        if isinstance(source_rotation, Mapping):
            direct = source_rotation.get('encoding')
    if direct is None:
        direct = metadata.get('rotation_repr')
    if direct is None:
        order = metadata.get('order')
        if order == 'xyz_xyzw':
            direct = 'quat_xyzw'
        elif order == 'xyz_wxyz':
            direct = 'quat_wxyz'
    if direct is None:
        raise ValueError('Pose fields require source_rotation_encoding metadata or a supported SiMate rotation_repr/order declaration')
    aliases = {'quaternion_xyzw': 'quat_xyzw', 'quaternion_wxyz': 'quat_wxyz'}
    encoding = aliases.get(str(direct), str(direct))
    if encoding not in ROTATION_DIMS:
        raise ValueError(f'Unsupported source rotation encoding: {encoding!r}')
    return encoding

def _field_size_and_shape(metadata: Mapping[str, Any], *, kind: str, name: str) -> tuple[int, tuple[int, ...]]:
    shape_value = metadata.get('shape')
    if shape_value is not None:
        if isinstance(shape_value, int):
            shape = (shape_value,)
        else:
            shape = tuple((int(item) for item in _sequence(shape_value, f'field {name!r}.shape')))
        if any((item <= 0 for item in shape)) or len(shape) > 1:
            raise ValueError(f'Field {name!r} must be scalar or a flat vector, got shape {shape}')
        return (int(np.prod(shape)) if shape else 1, shape)
    dimension = metadata.get('size', metadata.get('dim'))
    if dimension is None and kind == 'joint':
        joint_names = metadata.get('joint_names', metadata.get('order'))
        if isinstance(joint_names, Sequence) and (not isinstance(joint_names, (str, bytes))):
            dimension = len(joint_names)
    if dimension is None:
        if kind == 'gripper':
            return (1, ())
        raise ValueError(f'Joint field {name!r} requires an explicit flat shape/size or joint order')
    size = _positive_int(dimension, f'field {name!r}.size')
    return (size, (size,))

def _compile_selector(value: Any, *, role: str, kind: str, source_size: int, label: str, allow_numeric_action_select: bool) -> tuple[int, int]:
    layout_fields = value.get('layout_fields') if isinstance(value, Mapping) else None
    typed_stored_action_select = role == 'action' and isinstance(layout_fields, Sequence) and (not isinstance(layout_fields, (str, bytes))) and bool(layout_fields)
    if role != 'state' and (not typed_stored_action_select) and (not allow_numeric_action_select):
        raise ValueError(f'{label} is only supported for state fields')
    if kind == 'pose' and role != 'state' and (not typed_stored_action_select) and (not allow_numeric_action_select):
        raise ValueError(f'{label} is not supported for pose fields')
    selector = _parse_resolved_selector(value, label=label) if layout_fields is not None else _parse_selector(value, label=label)
    if selector[1] > source_size:
        raise ValueError(f'{label}.stop={selector[1]} exceeds raw source size {source_size}')
    return selector

def _parse_selector(value: Any, *, label: str) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping')
    unknown = sorted(set(value) - {'start', 'stop'})
    if unknown:
        raise ValueError(f"{label} has unsupported key(s): {', '.join(unknown)}")
    if set(value) != {'start', 'stop'}:
        raise ValueError(f'{label} requires exactly start and stop')
    start = value.get('start')
    stop = value.get('stop')
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or isinstance(stop, bool) or (not isinstance(stop, (int, np.integer))) or (int(start) < 0) or (int(stop) < 0) or (int(start) >= int(stop)):
        raise ValueError(f'{label} requires integer bounds satisfying 0 <= start < stop')
    return (int(start), int(stop))

def _parse_resolved_selector(value: Any, *, label: str) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping')
    unknown = sorted(set(value) - {'start', 'stop', 'layout_fields'})
    if unknown:
        raise ValueError(f"{label} has unsupported key(s): {', '.join(unknown)}")
    return _parse_selector({'start': value.get('start'), 'stop': value.get('stop')}, label=label)

def _validate_declared_shape(shape_value: Any, *, expected_dim: int, label: str) -> None:
    if shape_value is None:
        return
    if isinstance(shape_value, int):
        shape = (shape_value,)
    else:
        shape = tuple((int(item) for item in _sequence(shape_value, f'{label}.shape')))
    if shape != (expected_dim,):
        raise ValueError(f'{label} source shape must be ({expected_dim},), got {shape}')

def _canonical_ranges(metadata: Mapping[str, Any], size: int) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    range_value = metadata.get('canonical_range')
    if range_value is not None:
        sequence = _sequence(range_value, 'canonical_range')
        if len(sequence) == 2 and all((_is_number(item) for item in sequence)):
            lower = tuple((float(sequence[0]) for _ in range(size)))
            upper = tuple((float(sequence[1]) for _ in range(size)))
            return (lower, upper)
        if len(sequence) != size:
            raise ValueError(f'canonical_range must contain one pair or {size} pairs')
        pairs = [_sequence(item, 'canonical_range item') for item in sequence]
        if any((len(pair) != 2 for pair in pairs)):
            raise ValueError('Each canonical_range item must contain [min, max]')
        return (tuple((float(pair[0]) for pair in pairs)), tuple((float(pair[1]) for pair in pairs)))
    return (_expand_optional_floats(metadata.get('canonical_min'), size), _expand_optional_floats(metadata.get('canonical_max'), size))

def _mapping(value: Any, label: str, *, allow_none: bool=False) -> Mapping[str, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping')
    return value

def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f'{label} must be a sequence')
    return value

def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f'{label} must be an integer >= 1')
    return int(value)

def _expand_strings(value: Any, size: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple((value for _ in range(size)))
    sequence = _sequence(value, 'metadata')
    if len(sequence) != size:
        raise ValueError(f'Metadata must contain {size} values')
    return tuple((str(item) for item in sequence))

def _expand_optional_strings(value: Any, size: int) -> tuple[str | None, ...]:
    if value is None:
        return tuple((None for _ in range(size)))
    if isinstance(value, str):
        return tuple((value for _ in range(size)))
    sequence = _sequence(value, 'unit metadata')
    if len(sequence) != size:
        raise ValueError(f'Unit metadata must contain {size} values')
    return tuple((None if item is None else str(item) for item in sequence))

def _expand_optional_floats(value: Any, size: int) -> tuple[float | None, ...]:
    if value is None:
        return tuple((None for _ in range(size)))
    if _is_number(value):
        return tuple((float(value) for _ in range(size)))
    sequence = _sequence(value, 'numeric metadata')
    if len(sequence) != size:
        raise ValueError(f'Numeric metadata must contain {size} values')
    return tuple((None if item is None else float(item) for item in sequence))

def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)

def _single_unit(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    sequence = _sequence(value, 'pose unit')
    if len(sequence) != 1:
        raise ValueError('Pose position unit must be a single value')
    return str(sequence[0])

def _uniform_or_none(values: tuple[str | None, ...]) -> str | None:
    return values[0] if values and all((value == values[0] for value in values)) else None

def _collapse_tuple(value: tuple[Any, ...]) -> Any:
    if not value:
        return []
    if all((item == value[0] for item in value)):
        return value[0]
    return list(value)

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and (not isinstance(value, bool))
