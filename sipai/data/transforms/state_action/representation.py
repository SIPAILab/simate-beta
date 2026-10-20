from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from .fields import FieldValue, flatten_field, pack_raw_fields, read_anchor_field, read_field, unflatten_field
from .joint import canonicalize_joint, compose_joint, joint_difference
from .pose import Pose, compose_pose, identity_pose, pose_difference
from .spec import FieldSpec, RepresentationSpec

def encode_state_values(raw_state: Mapping[str, Any], history: Mapping[str, Any] | None, spec: RepresentationSpec, *, dtype: str | np.dtype='float64') -> np.ndarray:
    if spec.role != 'state':
        raise ValueError('encode_state_values requires a state spec')
    current = _read_fields(raw_state, spec.fields)
    if spec.representation == 'absolute':
        represented = {field.name: _absolute_value(current[field.name], field) for field in spec.fields}
    elif spec.representation == 'relative':
        historical = _read_fields(history, spec.fields) if history is not None else None
        represented = {}
        for field in spec.fields:
            current_value = current[field.name]
            if field.kind == 'gripper':
                represented[field.name] = current_value
            elif historical is None:
                represented[field.name] = _identity_value(field, dtype=np.dtype(dtype))
            else:
                represented[field.name] = _difference(historical[field.name], current_value, field, axes=spec.pose_delta_axes)
    else:
        raise ValueError(f'Unsupported state representation: {spec.representation!r}')
    return _flatten_fields(represented, spec, dtype=np.dtype(dtype))

def encode_action_values(raw_targets: Sequence[Mapping[str, Any]], raw_anchor: Mapping[str, Any] | None, spec: RepresentationSpec, *, dtype: str | np.dtype='float64') -> np.ndarray:
    if spec.role != 'action':
        raise ValueError('encode_action_values requires an action spec')
    needs_anchor = spec.representation != 'absolute' and any((field.kind != 'gripper' for field in spec.fields))
    if needs_anchor and raw_anchor is None:
        raise ValueError(f'Action representation {spec.representation!r} requires a raw absolute anchor')
    anchor = _read_action_anchor(raw_anchor, spec) if raw_anchor is not None else None
    target_rows = [_read_fields(target, spec.fields) for target in raw_targets]
    rows: list[np.ndarray] = []
    previous = anchor
    for target in target_rows:
        represented: dict[str, FieldValue] = {}
        for field in spec.fields:
            target_value = target[field.name]
            if field.kind == 'gripper' or spec.representation == 'absolute':
                represented[field.name] = _absolute_value(target_value, field)
            elif spec.representation == 'relative':
                if anchor is None:
                    raise ValueError('A raw action anchor is required for relative representation')
                represented[field.name] = _difference(target_value, anchor[field.name], field, axes=spec.pose_delta_axes)
            elif spec.representation == 'delta':
                if previous is None:
                    raise ValueError('A raw action anchor is required for delta representation')
                represented[field.name] = _difference(target_value, previous[field.name], field, axes=spec.pose_delta_axes)
            else:
                raise NotImplementedError(f'Unsupported action representation: {spec.representation!r}')
        rows.append(_flatten_fields(represented, spec, dtype=np.dtype(dtype)))
        previous = target
    resolved_dtype = np.dtype(dtype)
    return np.stack(rows, axis=0) if rows else np.empty((0, spec.semantic_dim), dtype=resolved_dtype)

def decode_action_values(semantic_values: object, raw_anchor: Mapping[str, Any] | None, spec: RepresentationSpec, *, dtype: str | np.dtype='float64') -> list[dict[str, Any]]:
    if spec.role != 'action':
        raise ValueError('decode_action_values requires an action spec')
    resolved_dtype = np.dtype(dtype)
    values = np.asarray(semantic_values, dtype=resolved_dtype)
    if values.ndim != 2 or values.shape[1] != spec.semantic_dim:
        raise ValueError(f'Semantic action must have shape [H, {spec.semantic_dim}], got {values.shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('Semantic action contains non-finite values')
    needs_anchor = spec.representation != 'absolute' and any((field.kind != 'gripper' for field in spec.fields))
    if needs_anchor and raw_anchor is None:
        raise ValueError(f'Action representation {spec.representation!r} requires a raw absolute anchor')
    anchor = _read_action_anchor(raw_anchor, spec) if raw_anchor is not None else None
    previous = anchor
    decoded: list[dict[str, Any]] = []
    for row in values:
        represented = _unflatten_fields(row, spec, dtype=resolved_dtype)
        absolute: dict[str, FieldValue] = {}
        for field in spec.fields:
            value = represented[field.name]
            if field.kind == 'gripper' or spec.representation == 'absolute':
                absolute[field.name] = _absolute_value(value, field)
            elif spec.representation == 'relative':
                if anchor is None:
                    raise ValueError('A raw action anchor is required for relative inverse')
                absolute[field.name] = _compose(value, anchor[field.name], field, axes=spec.pose_delta_axes)
            elif spec.representation == 'delta':
                if previous is None:
                    raise ValueError('A raw action anchor is required for delta inverse')
                absolute[field.name] = _compose(value, previous[field.name], field, axes=spec.pose_delta_axes)
            else:
                raise NotImplementedError(f'Unsupported action representation: {spec.representation!r}')
        decoded.append(pack_raw_fields(absolute, spec.fields, template=raw_anchor))
        previous = absolute
    return decoded

def canonicalize_semantic_values(values: object, spec: RepresentationSpec, *, dtype: str | np.dtype='float64') -> np.ndarray:
    resolved_dtype = np.dtype(dtype)
    array = np.asarray(values, dtype=resolved_dtype)
    if array.ndim != 2 or array.shape[1] != spec.semantic_dim:
        raise ValueError(f'Semantic values must have shape [H, {spec.semantic_dim}], got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError('Semantic values contain non-finite values')
    return np.stack([_flatten_fields(_unflatten_fields(row, spec, dtype=resolved_dtype), spec, dtype=resolved_dtype) for row in array], axis=0)

def _read_fields(payload: Mapping[str, Any] | None, fields: tuple[FieldSpec, ...]) -> dict[str, FieldValue]:
    if payload is None or not isinstance(payload, Mapping):
        raise ValueError('Raw state/action rows must be mappings')
    return {field.name: read_field(payload, field) for field in fields}

def _read_anchor_fields(payload: Mapping[str, Any] | None, fields: tuple[FieldSpec, ...]) -> dict[str, FieldValue]:
    if payload is None or not isinstance(payload, Mapping):
        raise ValueError('Raw action anchor must be a mapping')
    return {field.name: read_anchor_field(payload, field) for field in fields if field.kind != 'gripper'}

def _read_action_anchor(payload: Mapping[str, Any], spec: RepresentationSpec) -> dict[str, FieldValue]:
    if any((field.anchor_source is not None for field in spec.fields)):
        return _read_anchor_fields(payload, spec.fields)
    return _read_fields(payload, spec.fields)

def _flatten_fields(values: Mapping[str, FieldValue], spec: RepresentationSpec, *, dtype: np.dtype) -> np.ndarray:
    flattened = [flatten_field(values[field.name], field, rotation_encoding=spec.rotation_encoding) for field in spec.fields]
    result = np.asarray(np.concatenate(flattened), dtype=dtype)
    if result.shape != (spec.semantic_dim,):
        raise ValueError(f'Flattened representation must have shape ({spec.semantic_dim},), got {result.shape}')
    return result

def _unflatten_fields(row: np.ndarray, spec: RepresentationSpec, *, dtype: np.dtype) -> dict[str, FieldValue]:
    return {field.name: unflatten_field(np.asarray(row[field.start:field.end], dtype=dtype), field, rotation_encoding=spec.rotation_encoding) for field in spec.fields}

def _absolute_value(value: FieldValue, field: FieldSpec) -> FieldValue:
    if field.kind == 'joint' and field.difference:
        return canonicalize_joint(value, difference=field.difference, period=field.period, canonical_min=field.canonical_min)
    return value

def _difference(target: FieldValue, reference: FieldValue, field: FieldSpec, *, axes: str) -> FieldValue:
    if field.kind == 'pose':
        if not isinstance(target, Pose) or not isinstance(reference, Pose):
            raise TypeError(f'Pose field {field.name!r} requires Pose values')
        return pose_difference(target, reference, axes=axes)
    if field.kind == 'joint':
        if isinstance(target, Pose) or isinstance(reference, Pose):
            raise TypeError(f'Joint field {field.name!r} cannot contain Pose values')
        return joint_difference(target, reference, difference=field.difference, period=field.period)
    return target

def _compose(delta: FieldValue, reference: FieldValue, field: FieldSpec, *, axes: str) -> FieldValue:
    if field.kind == 'pose':
        if not isinstance(delta, Pose) or not isinstance(reference, Pose):
            raise TypeError(f'Pose field {field.name!r} requires Pose values')
        return compose_pose(delta, reference, axes=axes)
    if field.kind == 'joint':
        if isinstance(delta, Pose) or isinstance(reference, Pose):
            raise TypeError(f'Joint field {field.name!r} cannot contain Pose values')
        return compose_joint(delta, reference, difference=field.difference, period=field.period, canonical_min=field.canonical_min)
    return delta

def _identity_value(field: FieldSpec, *, dtype: np.dtype) -> FieldValue:
    if field.kind == 'pose':
        return identity_pose()
    return np.zeros(field.size, dtype=dtype)
