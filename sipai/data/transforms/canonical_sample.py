from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

@dataclass
class CanonicalSampleTransform:
    language_source: str = 'sub_task'
    include_raw: bool = True
    action_config: Optional[Mapping[str, Any]] = None
    action_already_processed: bool = False

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(sample)
        prompt = self._build_language_prompt(sample)
        if prompt is not None:
            out['language_prompt'] = prompt
        out['action'] = self._build_action(sample)
        if not self.include_raw:
            out.pop('raw_episode', None)
            out.pop('raw_step', None)
        return out

    def _build_language_prompt(self, sample: Mapping[str, Any]) -> Optional[str]:
        source = (self.language_source or 'none').lower()
        task = sample.get('task') or {}
        sub_task = sample.get('sub_task') or {}
        if source == 'none':
            return None
        if source == 'task':
            return _non_empty(task.get('instruction'))
        if source == 'sub_task':
            return _non_empty(sub_task.get('instruction')) or _non_empty(task.get('instruction'))
        if source == 'cot':
            cot = sub_task.get('cot') or {}
            return _non_empty(cot.get('text')) or _non_empty(sub_task.get('instruction')) or _non_empty(task.get('instruction'))
        if source == 'task_and_sub_task':
            task_text = _non_empty(task.get('instruction'))
            sub_text = _non_empty(sub_task.get('instruction'))
            if task_text and sub_text and (task_text != sub_text):
                return f'{task_text}\n{sub_text}'
            return sub_text or task_text
        raise ValueError(f'Unsupported language_source: {self.language_source}')

    def _build_action(self, sample: Mapping[str, Any]) -> Any:
        cfg = _mapping(self.action_config)
        if self.action_already_processed:
            return sample.get('action')
        source = str(cfg.get('source', 'stored')).lower()
        if source == 'stored':
            return sample.get('action')
        if source == 'none':
            return None
        if source == 'state_derivation':
            derivation_cfg = _mapping(cfg.get('derivation', cfg))
            strategy = str(derivation_cfg.get('strategy', 'eef_pose_delta')).lower()
            if strategy == 'eef_pose_delta':
                return _derive_eef_pose_delta_action(sample, derivation_cfg)
            raise ValueError(f'Unsupported action derivation strategy: {strategy}')
        raise ValueError(f'Unsupported action_config.source: {source}')

def _derive_eef_pose_delta_action(sample: Mapping[str, Any], cfg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    state_seq = (sample.get('observation') or {}).get('state')
    if not isinstance(state_seq, Sequence) or isinstance(state_seq, (str, bytes, Mapping)):
        return None
    step_offset = max(1, int(cfg.get('step_offset', 1)))
    action_step_interval_raw = cfg.get('action_step_interval', None)
    action_step_interval = max(1, int(action_step_interval_raw)) if action_step_interval_raw is not None else None
    state_keys = list(cfg.get('state_keys') or ['eef_pose.left', 'eef_pose.right'])
    pose_key = str(cfg.get('pose_key', 'pose'))
    include_position = bool(cfg.get('include_position', True))
    include_rotation = bool(cfg.get('include_rotation', True))
    include_gripper_width = bool(cfg.get('include_gripper_width', False))
    rotation_mode = str(cfg.get('rotation_mode', 'quat_delta_xyzw')).lower()
    scale_cfg = _mapping(cfg.get('scale', {}))
    xyz_scale = float(scale_cfg.get('xyz', 1.0))
    quat_vec_scale = float(scale_cfg.get('quat_vec', 1.0))
    gripper_width_scale = float(scale_cfg.get('gripper_width', 1.0))
    gripper_state_keys = list(cfg.get('gripper_state_keys') or _default_gripper_state_keys(state_keys))
    gripper_width_key = str(cfg.get('gripper_width_key', 'width'))
    flatten = bool(cfg.get('flatten', False))
    if action_step_interval is None:
        action_pairs = [(index, index + step_offset) for index in range(0, max(0, len(state_seq) - step_offset))]
        action_sampling = 'sliding'
    else:
        action_pairs = [(index, index + action_step_interval) for index in range(0, max(0, len(state_seq) - action_step_interval), action_step_interval)]
        action_sampling = 'sequential'
    values = []
    masks = []
    for (cur_index, next_index) in action_pairs:
        cur_state = state_seq[cur_index]
        next_state = state_seq[next_index]
        row = []
        mask = []
        for (state_key, gripper_state_key) in zip(state_keys, gripper_state_keys):
            cur_pose = _pose_from_state(cur_state, state_key, pose_key)
            next_pose = _pose_from_state(next_state, state_key, pose_key)
            if cur_pose is None or next_pose is None:
                dim = (3 if include_position else 0) + (_rotation_dim(rotation_mode) if include_rotation else 0)
                row.extend([0.0] * dim)
                mask.extend([0] * dim)
            else:
                if include_position:
                    row.extend([(float(next_pose[i]) - float(cur_pose[i])) * xyz_scale for i in range(3)])
                    mask.extend([1, 1, 1])
                if include_rotation:
                    cur_quat = [float(v) for v in cur_pose[3:7]]
                    next_quat = [float(v) for v in next_pose[3:7]]
                    if rotation_mode == 'quat_delta_xyzw':
                        row.extend(_quat_delta_xyzw(cur_quat, next_quat))
                    elif rotation_mode == 'quat_delta_xyz':
                        delta_quat = _canonicalize_quat_w_positive(_quat_delta_xyzw(cur_quat, next_quat))
                        row.extend([value * quat_vec_scale for value in delta_quat[:3]])
                    elif rotation_mode == 'quat_xyzw_diff':
                        row.extend([next_quat[i] - cur_quat[i] for i in range(4)])
                    elif rotation_mode == 'absolute_quat_xyzw':
                        row.extend(next_quat)
                    else:
                        raise ValueError(f'Unsupported rotation_mode: {rotation_mode}')
                    mask.extend([1] * _rotation_dim(rotation_mode))
            if include_gripper_width:
                gripper_width = _gripper_width_from_state(next_state, gripper_state_key, gripper_width_key)
                if gripper_width is None:
                    row.append(0.0)
                    mask.append(0)
                else:
                    row.append(float(gripper_width) * gripper_width_scale)
                    mask.append(1)
        values.append(row)
        masks.append(mask)
    if flatten:
        flat_values = [value for row in values for value in row]
        flat_mask = [value for row in masks for value in row]
        values_out: Any = flat_values
        mask_out: Any = flat_mask
    else:
        values_out = values
        mask_out = masks
    layout = _eef_pose_delta_layout(state_keys=state_keys, include_position=include_position, include_rotation=include_rotation, include_gripper_width=include_gripper_width, rotation_mode=rotation_mode, scale={'xyz': xyz_scale, 'quat_vec': quat_vec_scale, 'gripper_width': gripper_width_scale}, gripper_state_keys=gripper_state_keys, gripper_width_key=gripper_width_key)
    return {'values': values_out, 'mask': mask_out, 'source': 'state_derivation', 'derivation': {'strategy': 'eef_pose_delta', 'state_keys': state_keys, 'pose_key': pose_key, 'step_offset': step_offset, 'action_step_interval': action_step_interval, 'action_sampling': action_sampling, 'rotation_mode': rotation_mode, 'include_gripper_width': include_gripper_width, 'gripper_state_keys': gripper_state_keys, 'gripper_width_key': gripper_width_key, 'scale': {'xyz': xyz_scale, 'quat_vec': quat_vec_scale, 'gripper_width': gripper_width_scale}}, 'layout': layout}

def _pose_from_state(state: Any, state_key: str, pose_key: str) -> Optional[Sequence[float]]:
    if not isinstance(state, Mapping):
        return None
    entry = state.get(state_key)
    if not isinstance(entry, Mapping):
        return None
    pose = entry.get(pose_key)
    if pose is None and pose_key != 'pose':
        pose = entry.get('pose')
    if pose is None or len(pose) < 7:
        return None
    return pose

def _default_gripper_state_keys(state_keys: Sequence[str]) -> list[str]:
    keys = []
    for state_key in state_keys:
        if state_key.startswith('eef_pose.'):
            keys.append('end_effectors.' + state_key.split('.', 1)[1])
        else:
            keys.append(state_key)
    return keys

def _gripper_width_from_state(state: Any, state_key: str, width_key: str) -> Optional[float]:
    if not isinstance(state, Mapping):
        return None
    entry = state.get(state_key)
    if not isinstance(entry, Mapping):
        return None
    return _find_numeric_value(entry, preferred_keys=[width_key, 'width', 'position', 'gripper_width', 'opening_width', 'open_width', 'value'])

def _find_numeric_value(value: Any, *, preferred_keys: Sequence[str]) -> Optional[float]:
    if isinstance(value, Mapping):
        for key in preferred_keys:
            if key in value:
                result = _as_float(value[key])
                if result is not None:
                    return result
        message = value.get('message')
        if isinstance(message, Mapping):
            result = _find_numeric_value(message, preferred_keys=preferred_keys)
            if result is not None:
                return result
    return None

def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes, Mapping))) and (len(value) == 1):
        return _as_float(value[0])
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None

def _quat_delta_xyzw(cur_quat: Sequence[float], next_quat: Sequence[float]) -> list[float]:
    return _quat_multiply_xyzw(next_quat, _quat_inverse_xyzw(cur_quat))

def _quat_inverse_xyzw(quat: Sequence[float]) -> list[float]:
    (x, y, z, w) = quat
    norm_sq = x * x + y * y + z * z + w * w
    if norm_sq <= 0:
        return [0.0, 0.0, 0.0, 1.0]
    return [-x / norm_sq, -y / norm_sq, -z / norm_sq, w / norm_sq]

def _quat_multiply_xyzw(lhs: Sequence[float], rhs: Sequence[float]) -> list[float]:
    (lx, ly, lz, lw) = lhs
    (rx, ry, rz, rw) = rhs
    return [lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz]

def _canonicalize_quat_w_positive(quat: Sequence[float]) -> list[float]:
    if len(quat) < 4 or quat[3] >= 0:
        return [float(value) for value in quat]
    return [-float(value) for value in quat]

def _rotation_dim(rotation_mode: str) -> int:
    if rotation_mode == 'quat_delta_xyz':
        return 3
    if rotation_mode in {'quat_delta_xyzw', 'quat_xyzw_diff', 'absolute_quat_xyzw'}:
        return 4
    raise ValueError(f'Unsupported rotation_mode: {rotation_mode}')

def _eef_pose_delta_layout(*, state_keys: Sequence[str], include_position: bool, include_rotation: bool, include_gripper_width: bool, rotation_mode: str, scale: Mapping[str, float] | None=None, gripper_state_keys: Sequence[str] | None=None, gripper_width_key: str='width') -> Dict[str, Any]:
    fields = []
    cursor = 0
    scale = scale or {}
    xyz_scale = float(scale.get('xyz', 1.0))
    quat_vec_scale = float(scale.get('quat_vec', 1.0))
    gripper_width_scale = float(scale.get('gripper_width', 1.0))
    gripper_state_keys = list(gripper_state_keys or _default_gripper_state_keys(state_keys))
    for (state_key, gripper_state_key) in zip(state_keys, gripper_state_keys):
        prefix = state_key.replace('eef_pose.', '')
        if include_position:
            fields.append({'name': f'{prefix}.eef_delta_pos', 'start': cursor, 'end': cursor + 3, 'unit': 'meter', 'mode': 'delta', 'rotation_repr': None, 'source_state': state_key, 'scale': xyz_scale})
            cursor += 3
        if include_rotation:
            rotation_dim = _rotation_dim(rotation_mode)
            rotation_unit = 'quaternion_vector' if rotation_mode == 'quat_delta_xyz' else 'quaternion'
            rotation_scale = quat_vec_scale if rotation_mode == 'quat_delta_xyz' else 1.0
            fields.append({'name': f'{prefix}.eef_delta_rot', 'start': cursor, 'end': cursor + rotation_dim, 'unit': rotation_unit, 'mode': 'delta', 'rotation_repr': rotation_mode, 'source_state': state_key, 'scale': rotation_scale})
            cursor += rotation_dim
        if include_gripper_width:
            fields.append({'name': f'{prefix}.gripper_width', 'start': cursor, 'end': cursor + 1, 'unit': 'meter', 'mode': 'absolute', 'rotation_repr': None, 'source_state': gripper_state_key, 'source_field': gripper_width_key, 'scale': gripper_width_scale})
            cursor += 1
    scale_summary = {'xyz': xyz_scale, 'quat_vec': quat_vec_scale}
    if include_gripper_width:
        scale_summary['gripper_width'] = gripper_width_scale
    return {'summary': {'dim': cursor, 'normalized': any((value != 1.0 for value in scale_summary.values())), 'source': 'state_derivation', 'scale': scale_summary}, 'fields': fields}

def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, 'get'):
        return value
    return {}

def _non_empty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
