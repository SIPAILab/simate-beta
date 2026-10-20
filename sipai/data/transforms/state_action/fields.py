from __future__ import annotations
from collections.abc import Mapping, MutableMapping
from typing import Any
import numpy as np
from .pose import Pose, make_pose
from .rotation import decode_rotation, encode_rotation, rotation_dim
from .spec import FieldSpec
FieldValue = Pose | np.ndarray

def read_dotted_source(payload: Mapping[str, Any], source: str) -> Any:
    (value, _) = _read_dotted_source_with_trace(payload, source)
    return value

def read_field(payload: Mapping[str, Any], field: FieldSpec) -> FieldValue:
    raw_value = read_dotted_source(payload, field.source)
    if field.kind == 'pose':
        value = _finite_array(raw_value, field.source_shape, field.name)
        if field.selector is not None:
            (start, stop) = field.selector
            value = value.reshape(-1)[start:stop]
        encoding = field.source_rotation_encoding
        if encoding is None:
            raise ValueError(f'Pose field {field.name!r} has no source rotation encoding')
        return make_pose(value[:3], decode_rotation(value[3:], encoding))
    value = np.asarray(raw_value, dtype=np.float64)
    if value.shape != field.source_shape:
        raise ValueError(f'Field {field.name!r} at {field.source!r} must have shape {field.source_shape}, got {value.shape}')
    if not np.all(np.isfinite(value)):
        raise ValueError(f'Field {field.name!r} contains non-finite values')
    if field.selector is not None:
        (start, stop) = field.selector
        value = value.reshape(-1)[start:stop]
    return value.reshape(field.size)

def read_anchor_field(payload: Mapping[str, Any], field: FieldSpec) -> FieldValue:
    if field.anchor_source is None:
        raise ValueError(f'Field {field.name!r} has no anchor_source')
    raw_value = read_dotted_source(payload, field.anchor_source)
    if field.kind == 'pose':
        value = _finite_array(raw_value, field.anchor_source_shape, f'{field.name}.anchor')
        if field.selector is not None and (not field.layout_fields):
            (start, stop) = field.selector
            value = value.reshape(-1)[start:stop]
        encoding = field.anchor_source_rotation_encoding
        if encoding is None:
            raise ValueError(f'Pose field {field.name!r} anchor has no rotation encoding')
        return make_pose(value[:3], decode_rotation(value[3:], encoding))
    value = _finite_array(raw_value, field.anchor_source_shape, f'{field.name}.anchor')
    if field.selector is not None and (not field.layout_fields):
        (start, stop) = field.selector
        value = value.reshape(-1)[start:stop]
    return value.reshape(field.size)

def flatten_field(value: FieldValue, field: FieldSpec, *, rotation_encoding: str | None) -> np.ndarray:
    if field.kind == 'pose':
        if not isinstance(value, Pose):
            raise TypeError(f'Pose field {field.name!r} requires a Pose value')
        if rotation_encoding is None:
            raise ValueError(f'Pose field {field.name!r} requires a target rotation encoding')
        return np.concatenate((value.position, encode_rotation(value.rotation, rotation_encoding)))
    if isinstance(value, Pose):
        raise TypeError(f'Non-pose field {field.name!r} cannot contain a Pose value')
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (field.size,):
        raise ValueError(f'Field {field.name!r} must flatten to ({field.size},), got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'Field {field.name!r} contains non-finite values')
    return array

def unflatten_field(value: object, field: FieldSpec, *, rotation_encoding: str | None) -> FieldValue:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (field.semantic_dim,):
        raise ValueError(f'Semantic field {field.name!r} must have shape ({field.semantic_dim},), got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'Semantic field {field.name!r} contains non-finite values')
    if field.kind == 'pose':
        if rotation_encoding is None:
            raise ValueError(f'Pose field {field.name!r} requires a target rotation encoding')
        expected = 3 + rotation_dim(rotation_encoding)
        if array.shape != (expected,):
            raise ValueError(f'Pose field {field.name!r} must have {expected} semantic channels')
        return make_pose(array[:3], decode_rotation(array[3:], rotation_encoding))
    return array

def raw_field_value(value: FieldValue, field: FieldSpec) -> Any:
    if field.kind == 'pose':
        if not isinstance(value, Pose):
            raise TypeError(f'Pose field {field.name!r} requires a Pose value')
        encoding = field.source_rotation_encoding
        if encoding is None:
            raise ValueError(f'Pose field {field.name!r} has no source rotation encoding')
        return np.concatenate((value.position, encode_rotation(value.rotation, encoding))).astype(np.float32)
    if isinstance(value, Pose):
        raise TypeError(f'Non-pose field {field.name!r} cannot contain a Pose value')
    reshaped = np.asarray(value, dtype=np.float32).reshape(field.source_shape)
    if field.source_shape == ():
        return float(reshaped)
    return reshaped

def pack_raw_fields(values: Mapping[str, FieldValue], fields: tuple[FieldSpec, ...], *, template: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        if field.name not in values:
            raise ValueError(f'Decoded field {field.name!r} is missing')
        trace = _source_trace_or_default(template, field.source)
        if field.selector is None:
            _assign_trace(result, trace, raw_field_value(values[field.name], field))
            continue
        selected = _selected_raw_field_value(values[field.name], field)
        try:
            raw = np.asarray(read_dotted_source(result, field.source), dtype=np.float32)
        except KeyError:
            raw = np.zeros(field.source_shape, dtype=np.float32)
            _assign_trace(result, trace, raw)
        (start, stop) = field.selector
        raw.reshape(-1)[start:stop] = selected
    return result

def _selected_raw_field_value(value: FieldValue, field: FieldSpec) -> np.ndarray:
    if field.kind == 'pose':
        if not isinstance(value, Pose):
            raise TypeError(f'Pose field {field.name!r} requires a Pose value')
        encoding = field.source_rotation_encoding
        if encoding is None:
            raise ValueError(f'Pose field {field.name!r} has no source rotation encoding')
        return np.concatenate((value.position, encode_rotation(value.rotation, encoding))).astype(np.float32)
    if isinstance(value, Pose):
        raise TypeError(f'Non-pose field {field.name!r} cannot contain Pose values')
    return np.asarray(value, dtype=np.float32).reshape(field.size)

def _read_dotted_source_with_trace(payload: Mapping[str, Any], source: str) -> tuple[Any, tuple[str, ...]]:
    if not isinstance(payload, Mapping):
        raise ValueError('Canonical field payload must be a mapping')
    if not source:
        raise ValueError('Field source must be non-empty')
    current: Any = payload
    remaining = source
    trace: list[str] = []
    while True:
        if not isinstance(current, Mapping):
            raise KeyError(f'Source {source!r} traverses through a non-mapping value')
        candidates = [key for key in current if isinstance(key, str) and (remaining == key or remaining.startswith(f'{key}.'))]
        if not candidates:
            raise KeyError(f'Required field source {source!r} was not found')
        longest_length = max((len(key) for key in candidates))
        longest = [key for key in candidates if len(key) == longest_length]
        if len(longest) != 1:
            raise KeyError(f'Required field source {source!r} has an ambiguous exact-prefix resolution')
        key = longest[0]
        trace.append(key)
        current = current[key]
        if remaining == key:
            return (current, tuple(trace))
        remaining = remaining[len(key) + 1:]

def _source_trace_or_default(template: Mapping[str, Any] | None, source: str) -> tuple[str, ...]:
    if template is not None:
        try:
            (_, trace) = _read_dotted_source_with_trace(template, source)
            return trace
        except KeyError:
            pass
    return tuple(source.split('.'))

def _assign_trace(target: MutableMapping[str, Any], trace: tuple[str, ...], value: Any) -> None:
    current = target
    for key in trace[:-1]:
        existing = current.get(key)
        if existing is None:
            child: dict[str, Any] = {}
            current[key] = child
            current = child
        elif isinstance(existing, MutableMapping):
            current = existing
        else:
            raise ValueError(f'Decoded source layout conflicts at key {key!r}')
    final_key = trace[-1]
    if final_key in current:
        raise ValueError(f"Multiple fields write decoded source {'.'.join(trace)!r}")
    current[final_key] = value

def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f'Field {label!r} must have shape {shape}, got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'Field {label!r} contains non-finite values')
    return array
