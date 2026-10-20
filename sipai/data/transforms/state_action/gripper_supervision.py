from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .spec import RepresentationSpec
PHASE_OPEN = 0
PHASE_CLOSING = 1
PHASE_HOLD = 2
PHASE_OPENING = 3
PHASE_NAMES = ('open', 'closing', 'hold', 'opening')
PHASE_INDEX_SCHEMA_VERSION = 1

def manifest_episode_fingerprint(dataset_name: str, episodes: Sequence[Mapping[str, Any]]) -> str:
    payload = {'dataset_name': str(dataset_name), 'episodes': [{'episode_id': str(episode.get('episode_id', '')), 'num_steps': _episode_num_steps(episode)} for episode in episodes]}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def phase_index_fingerprint(path: str | Path) -> str:
    metadata_path = Path(path).expanduser().resolve(strict=False) / 'metadata.json'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Gripper phase metadata does not exist: {metadata_path}') from exc
    if not isinstance(metadata, Mapping):
        raise ValueError(f'Gripper phase metadata must contain an object: {metadata_path}')
    fingerprint_payload = {key: metadata.get(key) for key in ('schema_version', 'manifest_fingerprint', 'sides', 'phase_parameters')}
    fingerprint_payload['phase_sha256'] = metadata.get('phase_sha256')
    return hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def build_gripper_phase_arrays(gripper_values: object, *, smoothing_window: int=3, motion_threshold: float=0.01, settle_threshold: float=0.003, settle_steps: int=3, release_confirm_steps: int=2, minimum_closure_amount: float=0.1, ramp_steps: int=4) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(gripper_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f'gripper_values must have shape [T,S], got {values.shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('gripper_values contain non-finite values')
    smoothing_window = _positive_odd_int(smoothing_window, 'smoothing_window')
    settle_steps = _positive_int(settle_steps, 'settle_steps')
    release_confirm_steps = _positive_int(release_confirm_steps, 'release_confirm_steps')
    ramp_steps = _positive_int(ramp_steps, 'ramp_steps')
    motion_threshold = _positive_float(motion_threshold, 'motion_threshold')
    settle_threshold = _non_negative_float(settle_threshold, 'settle_threshold')
    minimum_closure_amount = _positive_float(minimum_closure_amount, 'minimum_closure_amount')
    if settle_threshold >= motion_threshold:
        raise ValueError('settle_threshold must be smaller than motion_threshold')
    smoothed = _median_smooth(values, smoothing_window)
    phases = np.full(values.shape, PHASE_OPEN, dtype=np.uint8)
    strengths = np.zeros(values.shape, dtype=np.float32)
    for side in range(values.shape[1]):
        _classify_one_gripper(smoothed[:, side], phases[:, side], strengths[:, side], motion_threshold=motion_threshold, settle_threshold=settle_threshold, settle_steps=settle_steps, release_confirm_steps=release_confirm_steps, minimum_closure_amount=minimum_closure_amount, ramp_steps=ramp_steps)
    return (phases, strengths)

@dataclass(frozen=True)
class GripperSupervisionSpec:
    phase_index: str
    fields: tuple[str, ...]
    hold_offset: float
    supervision_type: str = 'phase_aware_hold_offset'
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f'Unsupported gripper supervision schema_version: {self.schema_version}')
        if self.supervision_type != 'phase_aware_hold_offset':
            raise ValueError(f'Unsupported gripper supervision type: {self.supervision_type!r}')
        if not self.phase_index:
            raise ValueError('gripper supervision requires phase_index')
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError('gripper supervision fields must be a non-empty unique sequence')
        if _positive_float(self.hold_offset, 'hold_offset') > 1.0:
            raise ValueError('hold_offset must be in (0, 1]')

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> GripperSupervisionSpec | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError('gripper supervision must be a mapping')
        enabled = value.get('enabled', True)
        if not isinstance(enabled, bool):
            raise TypeError('gripper supervision enabled must be boolean')
        if not value or not enabled:
            return None
        unknown = set(value) - {'enabled', 'type', 'phase_index', 'fields', 'hold_offset'}
        if unknown:
            raise ValueError(f'Unsupported gripper supervision key(s): {sorted(unknown)}')
        raw_fields = value.get('fields', ())
        if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
            raise TypeError('gripper supervision fields must be a sequence')
        return cls(phase_index=str(value.get('phase_index', '')).strip(), fields=tuple((str(item) for item in raw_fields)), hold_offset=float(value.get('hold_offset', 0.0)), supervision_type=str(value.get('type', 'phase_aware_hold_offset')))

    def to_dict(self, *, phase_fingerprint: str | None=None) -> dict[str, Any]:
        result: dict[str, Any] = {'schema_version': self.schema_version, 'type': self.supervision_type, 'fields': list(self.fields), 'hold_offset': self.hold_offset}
        if phase_fingerprint is not None:
            result['phase_fingerprint'] = phase_fingerprint
        return result

class GripperPhaseIndex:

    def __init__(self, path: str | Path, *, dataset_name: str, episodes: Sequence[Mapping[str, Any]]) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        metadata_path = self.path / 'metadata.json'
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Gripper phase metadata does not exist: {metadata_path}') from exc
        if not isinstance(metadata, Mapping):
            raise ValueError(f'Gripper phase metadata must contain an object: {metadata_path}')
        if int(metadata.get('schema_version', -1)) != PHASE_INDEX_SCHEMA_VERSION:
            raise ValueError(f'Unsupported gripper phase index: {metadata_path}')
        expected_manifest = manifest_episode_fingerprint(dataset_name, episodes)
        if metadata.get('manifest_fingerprint') != expected_manifest:
            raise ValueError('Gripper phase index does not match the ordered dataset episodes')
        episode_ids = tuple((str(item) for item in metadata.get('episode_ids', ())))
        offsets = np.asarray(metadata.get('offsets', ()), dtype=np.int64)
        sides = tuple((str(item) for item in metadata.get('sides', ())))
        expected_episode_ids = tuple((str(episode.get('episode_id', '')) for episode in episodes))
        if episode_ids != expected_episode_ids:
            raise ValueError('Gripper phase episode IDs do not match the dataset')
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError('Gripper phase episode IDs must be unique')
        expected_offsets = np.concatenate(([0], np.cumsum([_episode_num_steps(episode) for episode in episodes], dtype=np.int64)))
        if offsets.shape != expected_offsets.shape or not np.array_equal(offsets, expected_offsets):
            raise ValueError('Gripper phase offsets do not match dataset episode lengths')
        if not sides or len(set(sides)) != len(sides):
            raise ValueError('Gripper phase sides are invalid')
        self.metadata = metadata
        self.episode_ids = episode_ids
        self.offsets = offsets
        self.sides = sides
        self._episode_lookup = {episode_id: index for (index, episode_id) in enumerate(episode_ids)}
        self.phases = np.load(self.path / 'phases.npy', mmap_mode='r')
        self.strengths = np.load(self.path / 'strengths.npy', mmap_mode='r')
        expected_shape = (int(offsets[-1]), len(sides))
        if self.phases.shape != expected_shape or self.phases.dtype != np.uint8:
            raise ValueError(f'Gripper phase array must be uint8 {expected_shape}, got {self.phases.shape}')
        if self.strengths.shape != expected_shape or not np.issubdtype(self.strengths.dtype, np.floating):
            raise ValueError(f'Gripper strength array must be floating {expected_shape}, got {self.strengths.shape}')
        if np.any(self.phases > PHASE_OPENING):
            raise ValueError('Gripper phase array contains unsupported phase codes')
        if not np.all(np.isfinite(self.strengths)) or np.any(self.strengths < 0) or np.any(self.strengths > 1):
            raise ValueError('Gripper strength array must contain finite values in [0, 1]')
        if np.any(self.strengths[self.phases != PHASE_HOLD] != 0):
            raise ValueError('Gripper strength must be zero outside HOLD')
        expected_sha256 = str(metadata.get('phase_sha256', ''))
        actual_sha256 = _phase_arrays_sha256(self.phases, self.strengths)
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise ValueError('Gripper phase array checksum does not match metadata')
        self.fingerprint = phase_index_fingerprint(self.path)

    def rows(self, episode_id: str, step_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        try:
            episode_index = self._episode_lookup[str(episode_id)]
        except KeyError as exc:
            raise KeyError(f'Episode {episode_id!r} is missing from the gripper phase index') from exc
        start = int(self.offsets[episode_index])
        length = int(self.offsets[episode_index + 1] - start)
        indices = np.asarray(step_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= length):
            raise IndexError(f'Gripper phase indices are outside episode length {length}: {indices.tolist()}')
        absolute = start + indices
        return (np.asarray(self.phases[absolute]), np.asarray(self.strengths[absolute], dtype=np.float32))

class GripperActionSupervision:

    def __init__(self, spec: GripperSupervisionSpec, *, action_spec: RepresentationSpec, dataset_name: str, episodes: Sequence[Mapping[str, Any]]) -> None:
        self.spec = spec
        self.action_spec = action_spec
        self.phase_index = GripperPhaseIndex(spec.phase_index, dataset_name=dataset_name, episodes=episodes)
        side_lookup = {side: index for (index, side) in enumerate(self.phase_index.sides)}
        fields = {field.name: field for field in action_spec.fields}
        resolved: list[tuple[Any, int, float]] = []
        for name in spec.fields:
            try:
                field = fields[name]
            except KeyError as exc:
                raise ValueError(f'Gripper supervision field {name!r} is absent from the action spec') from exc
            if field.kind != 'gripper' or field.semantic_dim != 1:
                raise ValueError(f'Gripper supervision field {name!r} must be a scalar gripper')
            units = tuple(field.units)
            if units == ('open_fraction',):
                direction = -1.0
            elif units == ('closed_fraction',):
                direction = 1.0
            else:
                raise ValueError(f'Gripper supervision field {name!r} requires open_fraction or closed_fraction, got {units}')
            side = name.split('.', 1)[0]
            if side not in side_lookup:
                raise ValueError(f'Gripper side {side!r} is absent from phase index sides {self.phase_index.sides}')
            resolved.append((field, side_lookup[side], direction))
        self._resolved_fields = tuple(resolved)

    def apply(self, semantic_values: object, *, episode_id: str, target_step_indices: Sequence[int]) -> np.ndarray:
        values = np.asarray(semantic_values).copy()
        expected = (len(target_step_indices), self.action_spec.semantic_dim)
        if values.shape != expected:
            raise ValueError(f'Semantic action must have shape {expected}, got {values.shape}')
        (_, strengths) = self.phase_index.rows(episode_id, target_step_indices)
        for (field, side_index, direction) in self._resolved_fields:
            values[:, field.start:field.end] += direction * self.spec.hold_offset * strengths[:, side_index, None]
            np.clip(values[:, field.start:field.end], 0.0, 1.0, out=values[:, field.start:field.end])
        return values

    def contract_dict(self) -> dict[str, Any]:
        return self.spec.to_dict(phase_fingerprint=self.phase_index.fingerprint)

def _classify_one_gripper(values: np.ndarray, phases: np.ndarray, strengths: np.ndarray, *, motion_threshold: float, settle_threshold: float, settle_steps: int, release_confirm_steps: int, minimum_closure_amount: float, ramp_steps: int) -> None:
    if values.size < 2:
        return
    delta = np.diff(values, prepend=values[0])
    mode = PHASE_OPEN
    close_start_value = float(values[0])
    stable_count = 0
    opening_count = 0
    hold_age = 0
    for index in range(1, values.size):
        movement = float(delta[index])
        if mode in {PHASE_OPEN, PHASE_OPENING}:
            if movement <= -motion_threshold:
                mode = PHASE_CLOSING
                close_start_value = float(values[index - 1])
                stable_count = 0
                opening_count = 0
            elif mode == PHASE_OPENING and abs(movement) <= settle_threshold:
                mode = PHASE_OPEN
        elif mode in {PHASE_CLOSING, PHASE_HOLD}:
            opening_count = opening_count + 1 if movement >= motion_threshold else 0
            if opening_count >= release_confirm_steps:
                begin = index - release_confirm_steps + 1
                phases[begin:index + 1] = PHASE_OPENING
                strengths[begin:index + 1] = 0.0
                mode = PHASE_OPENING
                hold_age = 0
                continue
            if mode == PHASE_CLOSING:
                stable_count = stable_count + 1 if abs(movement) <= settle_threshold else 0
                closure = close_start_value - float(values[index])
                if stable_count >= settle_steps and closure >= minimum_closure_amount:
                    begin = index - settle_steps + 1
                    phases[begin:index + 1] = PHASE_HOLD
                    for row in range(begin, index + 1):
                        strengths[row] = min((row - begin + 1) / ramp_steps, 1.0)
                    mode = PHASE_HOLD
                    hold_age = settle_steps
            else:
                hold_age += 1
                strengths[index] = min(hold_age / ramp_steps, 1.0)
        phases[index] = mode

def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1 or values.shape[0] < 2:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode='edge')
    return np.stack([np.median(padded[index:index + window], axis=0) for index in range(values.shape[0])], axis=0)

def _phase_arrays_sha256(phases: np.ndarray, strengths: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (phases, strengths):
        raw = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        for start in range(0, raw.size, 8 * 1024 * 1024):
            digest.update(memoryview(raw[start:start + 8 * 1024 * 1024]))
    return digest.hexdigest()

def _episode_num_steps(episode: Mapping[str, Any]) -> int:
    for value in ((episode.get('step_index') or {}).get('num_steps'), ((episode.get('meta_data') or {}).get('summary') or {}).get('num_steps'), ((episode.get('extra') or {}).get('summary') or {}).get('num_steps')):
        if value is not None:
            return int(value)
    raise ValueError(f"Episode {episode.get('episode_id')!r} has no compact num_steps")

def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f'{name} must be an integer >= 1')
    return int(value)

def _positive_odd_int(value: object, name: str) -> int:
    result = _positive_int(value, name)
    if result % 2 != 1:
        raise ValueError(f'{name} must be odd')
    return result

def _positive_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return result

def _non_negative_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f'{name} must be finite and non-negative')
    return result
__all__ = ['PHASE_CLOSING', 'PHASE_HOLD', 'PHASE_NAMES', 'PHASE_OPEN', 'PHASE_OPENING', 'GripperActionSupervision', 'GripperPhaseIndex', 'GripperSupervisionSpec', 'build_gripper_phase_arrays', 'manifest_episode_fingerprint', 'phase_index_fingerprint']
