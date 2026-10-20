from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from .fields import read_dotted_source
from .gripper_supervision import _episode_num_steps

def validate_target_override_config(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError('target_overrides must be a mapping')
    unknown = set(value) - {'path', 'normalization'}
    if unknown:
        raise ValueError(f'Unsupported target_overrides keys: {sorted(unknown)}')
    path = value.get('path')
    if not isinstance(path, str) or not path.strip():
        raise ValueError('target_overrides.path is required')
    if value.get('normalization') != 'frozen_base':
        raise ValueError('target_overrides.normalization must explicitly be frozen_base')
    return path

class AbsoluteTargetOverrides:

    def __init__(self, config: Mapping[str, Any], *, episodes: Sequence[Mapping[str, Any]]) -> None:
        self.path = Path(validate_target_override_config(config)).expanduser().resolve()
        raw = self.path.read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if data.get('schema_version') != 1 or data.get('source') != 'state.values':
            raise ValueError('Unsupported absolute target override schema/source')
        if data.get('source_shape') != [14]:
            raise ValueError('Target overrides require the RoboDojo 14-channel joint state')
        self.source = 'state.values'
        episode_lookup = {str(e['episode_id']): e for e in episodes}
        if len(episode_lookup) != len(episodes):
            raise ValueError('Target override matching requires unique episode IDs')
        self.rows: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self.disabled_gripper_episodes: set[str] = set()
        for record in data['episodes']:
            episode_id = str(record['episode_id'])
            if episode_id in self.rows:
                raise ValueError(f'Duplicate override episode: {episode_id}')
            if episode_id not in episode_lookup:
                raise ValueError(f'Override episode is absent from this dataset: {episode_id}')
            episode = episode_lookup[episode_id]
            length = _episode_num_steps(episode)
            instruction = str((episode.get('task') or {}).get('instruction', ''))
            if record['num_steps'] != length or record['task_instruction'] != instruction:
                raise ValueError(f'Override identity/length mismatch: {episode_id}')
            disabled = record.get('disable_gripper_compensation', False)
            if not isinstance(disabled, bool):
                raise TypeError('disable_gripper_compensation must be boolean')
            if disabled:
                self.disabled_gripper_episodes.add(episode_id)
            rows = {}
            for row in record['overrides']:
                step = row['step']
                if isinstance(step, bool) or not isinstance(step, int) or (not 0 <= step < length):
                    raise ValueError(f'Invalid target override frame: {episode_id}:{step}')
                if step in rows:
                    raise ValueError(f'Duplicate target override frame: {episode_id}:{step}')
                indices = np.asarray(row['indices'])
                original = np.asarray(row['original_values'], dtype=np.float64)
                values = np.asarray(row['values'], dtype=np.float64)
                if indices.ndim != 1 or not indices.size or (not np.issubdtype(indices.dtype, np.integer)):
                    raise ValueError('Override indices must be a nonempty integer vector')
                if len(set(indices.tolist())) != indices.size or np.any(indices < 0) or np.any(indices >= 14):
                    raise ValueError('Override indices must be unique channels in [0, 14)')
                if original.shape != indices.shape or values.shape != indices.shape:
                    raise ValueError('Override indices/values/original_values shape mismatch')
                if not np.all(np.isfinite(original)) or not np.all(np.isfinite(values)):
                    raise ValueError('Non-finite target override')
                gripper = np.isin(indices, [6, 13])
                if np.any(values[gripper] < 0) or np.any(values[gripper] > 1):
                    raise ValueError('Overridden gripper values must be open fractions in [0, 1]')
                rows[step] = (indices.astype(np.int64), original, values)
            self.rows[episode_id] = rows

    def disables_gripper(self, episode_id: str) -> bool:
        return str(episode_id) in self.disabled_gripper_episodes

    def apply(self, targets: Sequence[Mapping[str, Any]], *, episode_id: str, target_step_indices: Sequence[int]) -> list[Mapping[str, Any]]:
        if len(targets) != len(target_step_indices):
            raise ValueError('Target rows and absolute frame indices differ in length')
        overrides = self.rows.get(str(episode_id), {})
        result = []
        for (target, step) in zip(targets, target_step_indices, strict=True):
            override = overrides.get(int(step))
            if override is None:
                result.append(target)
                continue
            (indices, original, values) = override
            raw = np.asarray(read_dotted_source(target, self.source))
            if raw.shape != (14,) or not np.allclose(raw[indices], original, atol=1e-06, rtol=0):
                raise ValueError(f'Original GT does not match override: {episode_id}:{step}')
            patched = raw.copy()
            patched[indices] = values
            changed = copy.deepcopy(target)
            if self.source in changed:
                changed[self.source] = patched
            else:
                changed['state']['values'] = patched
            result.append(changed)
        return result
