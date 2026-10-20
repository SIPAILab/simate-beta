from __future__ import annotations
import os
from typing import Any, Dict, Mapping, Sequence
from sipai.data.schema import CanonicalBatch

class CanonicalCollator:

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> CanonicalBatch:
        return canonical_collate_fn(samples)

def canonical_collate_fn(samples: Sequence[Mapping[str, Any]]) -> CanonicalBatch:
    if not samples:
        return _empty_canonical_batch()
    history_images = _collate_history_images(samples)
    return {'inputs': {'images': _collate_images(samples), **({'history_images': history_images} if history_images else {}), 'language': [_language_from_sample(sample) for sample in samples], 'state': [_get_nested(sample, 'observation', 'state') for sample in samples], 'tactile': [_get_nested(sample, 'observation', 'tactile', default={}) for sample in samples]}, 'targets': {'actions': [sample.get('action') for sample in samples], 'sub_goal': [sample.get('sub_goal') for sample in samples], 'perception': [sample.get('perception_gt') for sample in samples]}, 'masks': {'images': _collate_image_masks(samples), **({'history_images': _history_image_masks(history_images)} if history_images else {}), 'actions': [sample.get('action') is not None for sample in samples], 'state': [_get_nested(sample, 'observation', 'state') is not None for sample in samples], 'tactile': [bool(_get_nested(sample, 'observation', 'tactile', default={})) for sample in samples], 'sub_goal': [sample.get('sub_goal') is not None for sample in samples], 'perception': [sample.get('perception_gt') is not None for sample in samples]}, 'meta': {'dataset_name': [sample.get('dataset_name') for sample in samples], 'episode_id': [sample.get('episode_id') for sample in samples], 'step_index': [sample.get('step_index') for sample in samples], 'timestamp': [sample.get('timestamp') for sample in samples], 'window': [sample.get('window') for sample in samples], 'task': [sample.get('task') for sample in samples], 'sub_task': [sample.get('sub_task') for sample in samples], 'quality': [_get_nested(sample, 'meta_data', 'quality') for sample in samples], 'mistake': [_get_nested(sample, 'meta_data', 'mistake') for sample in samples], 'value_weight': [sample.get('value_weight', _get_nested(sample, 'meta_data', 'value_weight', default=1.0)) for sample in samples], 'meta_data': [sample.get('meta_data') for sample in samples], 'source_info': [sample.get('source_info') for sample in samples], 'raw_state': [sample.get('raw_state') for sample in samples]}}

def canonical_flat_collate_fn(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    keys = set().union(*(sample.keys() for sample in samples))
    return {key: [sample.get(key) for sample in samples] for key in sorted(keys)}

def statistics_collate_fn(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    trace = getattr(samples, 'trace', None)
    if not isinstance(trace, Mapping):
        trace = {'pid': os.getpid(), 'worker_id': None, 'indices': tuple((int(sample['sample_index']) for sample in samples if sample.get('sample_index') is not None)), 'block_size': len(samples), 'materialize_seconds': None, 'used_getitems': False}
    return {'samples': list(samples), 'materialization_trace': dict(trace)}

def _empty_canonical_batch() -> CanonicalBatch:
    return {'inputs': {'images': {}, 'language': [], 'state': [], 'tactile': []}, 'targets': {'actions': [], 'sub_goal': [], 'perception': []}, 'masks': {'images': {}, 'actions': [], 'state': [], 'tactile': [], 'sub_goal': [], 'perception': []}, 'meta': {'dataset_name': [], 'episode_id': [], 'step_index': [], 'timestamp': [], 'window': [], 'task': [], 'sub_task': [], 'quality': [], 'mistake': [], 'value_weight': [], 'meta_data': [], 'source_info': [], 'raw_state': []}}

def _get_nested(mapping: Mapping[str, Any], *keys: str, default: Any=None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(key, default)
        if cur is default:
            return default
    return cur

def _language_from_sample(sample: Mapping[str, Any]) -> str | None:
    prompt = sample.get('language_prompt')
    if prompt is not None:
        return str(prompt)
    sub_task = sample.get('sub_task') or {}
    task = sample.get('task') or {}
    text = sub_task.get('instruction') or task.get('instruction')
    return str(text) if text is not None else None

def _collate_images(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    camera_names = sorted({camera_name for sample in samples for camera_name in _get_nested(sample, 'observation', 'cameras', default={}) or {}})
    return {camera_name: [(_get_nested(sample, 'observation', 'cameras', default={}) or {}).get(camera_name) for sample in samples] for camera_name in camera_names}

def _collate_image_masks(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[bool]]:
    images = _collate_images(samples)
    return {camera_name: [payload is not None for payload in payloads] for (camera_name, payloads) in images.items()}

def _collate_history_images(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    camera_names = sorted({camera_name for sample in samples for camera_name in _get_nested(sample, 'observation', 'camera_history', default={}) or {}})
    return {camera_name: [(_get_nested(sample, 'observation', 'camera_history', default={}) or {}).get(camera_name) for sample in samples] for camera_name in camera_names}

def _history_image_masks(histories: Mapping[str, Sequence[Any]]) -> dict[str, list[list[bool] | None]]:
    return {camera_name: [None if history is None else [frame is not None for frame in history] for history in batch_histories] for (camera_name, batch_histories) in histories.items()}

def _image_mask(sample: Mapping[str, Any]) -> dict[str, bool]:
    cameras = _get_nested(sample, 'observation', 'cameras', default={}) or {}
    return {camera_name: payload is not None for (camera_name, payload) in cameras.items()}
