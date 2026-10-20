from __future__ import annotations
import inspect
from functools import partial
from typing import Any, Mapping
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sipai.config import get_config_value, get_nested_config, normalize_sipai_config
from sipai.data.collators import canonical_collate_fn
from sipai.data.datasets import CanonicalEmbodiedDataset
from sipai.data.samplers import MemoryMixtureSampler
from sipai.engine.distributed import DistributedContext
from sipai.registry import DATALOADERS

def _to_plain_config(value: Any) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    wrapped = getattr(value, '_cfg', None)
    if wrapped is not None and OmegaConf.is_config(wrapped):
        return OmegaConf.to_container(wrapped, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_plain_config(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_config(item) for item in value]
    return value

def canonical_worker_init_fn(worker_id: int, *, torch_num_threads: int | None=1, opencv_num_threads: int | None=1, disable_opencv_opencl: bool=True) -> None:
    del worker_id
    if torch_num_threads is not None and int(torch_num_threads) > 0:
        try:
            import torch
            torch.set_num_threads(int(torch_num_threads))
        except Exception:
            pass
    try:
        import cv2
        if opencv_num_threads is not None and int(opencv_num_threads) >= 0:
            cv2.setNumThreads(int(opencv_num_threads))
        if disable_opencv_opencl and hasattr(cv2, 'ocl'):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

@DATALOADERS.register_module(name='CanonicalDataloader', force=True)
def build_canonical_dataloader(cfg: Any, *, dataset_manifest: Any, distributed_context: DistributedContext | None=None) -> DataLoader:
    cfg = normalize_sipai_config(cfg)
    data_cfg = get_nested_config(cfg, 'datasets.vla_data', default=cfg)
    data_root = get_config_value(data_cfg, 'data_root', '.')
    reader_backend = get_config_value(data_cfg, 'reader_backend', 'simate_unified')
    reader_options = _to_plain_config(get_config_value(data_cfg, 'reader_options', {})) or {}
    language_source = get_config_value(data_cfg, 'language_source', 'sub_task')
    action_config = _to_plain_config(get_config_value(data_cfg, 'action_config', None))
    action_supervision = _to_plain_config(get_config_value(data_cfg, 'action_supervision', None))
    counterfactual_action_supervision = _to_plain_config(get_config_value(data_cfg, 'counterfactual_action_supervision', None))
    state_recovery_supervision = _to_plain_config(get_config_value(data_cfg, 'state_recovery_supervision', None))
    episode_filter = _to_plain_config(get_config_value(data_cfg, 'episode_filter', None))
    representation_config = _to_plain_config(get_config_value(data_cfg, 'state_action_representation', None))
    representation_compute_dtype = str(get_config_value(data_cfg, 'representation_compute_dtype', 'float64'))
    representation_output_dtype = str(get_config_value(data_cfg, 'representation_output_dtype', 'float32'))
    sample_window_size = int(get_config_value(data_cfg, 'sample_window_size', 1))
    sample_stride = int(get_config_value(data_cfg, 'sample_stride', 1))
    max_samples_per_episode = get_config_value(data_cfg, 'max_samples_per_episode', None)
    camera_window_frame_offsets = get_config_value(data_cfg, 'camera_window_frame_offsets', None)
    camera_history_name = get_config_value(data_cfg, 'camera_history_name', None)
    camera_history_frame_offsets = get_config_value(data_cfg, 'camera_history_frame_offsets', None)
    camera_payload_image_size = get_config_value(data_cfg, 'camera_payload_image_size', None)
    max_cached_steps = int(get_config_value(data_cfg, 'max_cached_steps', 0) or 0)
    batch_state_window = bool(get_config_value(data_cfg, 'batch_state_window', True))
    batch_prefetch = bool(get_config_value(data_cfg, 'batch_prefetch', True))
    batch_prefetch_cameras = bool(get_config_value(data_cfg, 'batch_prefetch_cameras', True))
    statistics_state_action_only = bool(get_config_value(data_cfg, 'statistics_state_action_only', False))
    skip_bad_samples = bool(get_config_value(data_cfg, 'skip_bad_samples', False))
    max_skip_attempts = int(get_config_value(data_cfg, 'max_skip_attempts', 32))
    strict = bool(get_config_value(data_cfg, 'strict', True))
    output_format = str(get_config_value(data_cfg, 'output_format', 'canonical')).lower()
    sources_cfg = _to_plain_config(get_config_value(data_cfg, 'sources', None)) or []
    representation_transform = None
    if representation_config is not None:
        from sipai.data.transforms.state_action import compile_state_action_transform
        representation_config = dict(representation_config)
        representation_config['schema'] = _to_plain_config((dataset_manifest.get('extra') or {}).get('schema') if isinstance(dataset_manifest, Mapping) else {}) or {}
        model_dim = int(representation_config.get('model_dim', 0) or 0)
        action_source = str((representation_config.get('action') or {}).get('source', 'stored'))
        dim_policy = str(representation_config.get('dim_policy', 'exact'))
        if dim_policy == 'pad_to_capacity' and action_source != 'none' and (model_dim <= 0):
            raise ValueError('pad_to_capacity state/action representation requires an explicit positive model_dim/action_dim')
        representation_transform = compile_state_action_transform(representation_config, model_dim=model_dim or None, dim_policy=dim_policy, compute_dtype=representation_compute_dtype, output_dtype=representation_output_dtype)
    dataset = CanonicalEmbodiedDataset(dataset_manifest=dataset_manifest, data_root=data_root, reader_backend=reader_backend, reader_options=reader_options, language_source=language_source, action_config=action_config, action_supervision=action_supervision, counterfactual_action_supervision=counterfactual_action_supervision, state_recovery_supervision=state_recovery_supervision, episode_filter=episode_filter, representation_transform=representation_transform, state_action_config=representation_config, sample_window_size=sample_window_size, sample_stride=sample_stride, max_samples_per_episode=max_samples_per_episode, camera_window_frame_offsets=list(camera_window_frame_offsets) if camera_window_frame_offsets is not None else None, camera_history_name=str(camera_history_name) if camera_history_name is not None else None, camera_history_frame_offsets=list(camera_history_frame_offsets) if camera_history_frame_offsets is not None else None, camera_payload_image_size=list(camera_payload_image_size) if camera_payload_image_size is not None else None, max_cached_steps=max_cached_steps, batch_state_window=batch_state_window, batch_prefetch=batch_prefetch, batch_prefetch_cameras=batch_prefetch_cameras, statistics_state_action_only=statistics_state_action_only, skip_bad_samples=skip_bad_samples, max_skip_attempts=max_skip_attempts, strict=strict)
    batch_size = int(get_config_value(data_cfg, 'per_device_batch_size', 1))
    num_workers = int(get_config_value(data_cfg, 'num_workers', 0))
    pin_memory = bool(get_config_value(data_cfg, 'pin_memory', False))
    shuffle = bool(get_config_value(data_cfg, 'shuffle', False))
    drop_last = bool(get_config_value(data_cfg, 'drop_last', False))
    persistent_workers = bool(get_config_value(data_cfg, 'persistent_workers', False))
    prefetch_factor = get_config_value(data_cfg, 'prefetch_factor', None)
    in_order = get_config_value(data_cfg, 'in_order', None)
    worker_init_cfg = _to_plain_config(get_config_value(data_cfg, 'worker_init', None)) or {}
    sampler_cfg = _to_plain_config(get_config_value(data_cfg, 'sampler', None)) or {}
    if sampler_cfg.get('type') != 'MemoryMixtureSampler':
        raise ValueError('This release requires MemoryMixtureSampler.')
    seed = int(sampler_cfg.get('seed', get_config_value(cfg, 'seed', 0)))
    rank = distributed_context.rank if distributed_context is not None and distributed_context.enabled else None
    world_size = distributed_context.world_size if distributed_context is not None and distributed_context.enabled else None
    dataset_weights = _dataset_weights_from_sources(dataset, sources_cfg)
    sampler = MemoryMixtureSampler(dataset, memory_ratio=float(sampler_cfg.get('memory_ratio', 0.5)), epoch_length=sampler_cfg.get('epoch_length', 'auto'), dataset_weights=dataset_weights, memory=_to_plain_config(sampler_cfg.get('memory', {})) or {}, mixture=_to_plain_config(sampler_cfg.get('mixture', {})) or {}, seed=seed, rank=rank, world_size=world_size, batch_size=batch_size, drop_last=bool(sampler_cfg.get('drop_last', False)))
    shuffle = False
    if output_format != 'canonical':
        raise ValueError(f"Unsupported canonical output_format: {output_format!r}; expected 'canonical'.")
    collate_fn = canonical_collate_fn
    dataloader_kwargs = {'batch_size': batch_size, 'shuffle': shuffle if sampler is None else False, 'sampler': sampler, 'num_workers': num_workers, 'pin_memory': pin_memory, 'drop_last': drop_last, 'collate_fn': collate_fn}
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = int(prefetch_factor)
        if in_order is not None:
            if 'in_order' not in inspect.signature(DataLoader).parameters:
                raise ValueError('Configured datasets.vla_data.in_order, but this PyTorch DataLoader has no in_order parameter')
            dataloader_kwargs['in_order'] = bool(in_order)
        if bool(worker_init_cfg.get('limit_threads', False)):
            dataloader_kwargs['worker_init_fn'] = partial(canonical_worker_init_fn, torch_num_threads=worker_init_cfg.get('torch_num_threads', 1), opencv_num_threads=worker_init_cfg.get('opencv_num_threads', 1), disable_opencv_opencl=bool(worker_init_cfg.get('disable_opencv_opencl', True)))
    return DataLoader(dataset, **dataloader_kwargs)

def _dataset_weights_from_sources(dataset: CanonicalEmbodiedDataset, sources_cfg: list[Any]) -> dict[str, float]:
    sources = [source for source in sources_cfg if isinstance(source, Mapping) and source.get('name')]
    if not sources:
        return {}
    if all((str(source.get('sampling_mode', 'ratio')).lower() == 'natural' for source in sources)):
        counts = _source_sample_counts(dataset)
        return {str(source['name']): float(max(1, counts.get(str(source['name']), 0))) for source in sources}
    return {str(source['name']): float(source.get('sampling_weight', source.get('ratio', source.get('weight', 1.0)))) for source in sources}

def _source_sample_counts(dataset: CanonicalEmbodiedDataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (compact_pos, compact) in enumerate(dataset.episode_index):
        start = int(dataset.sample_cumsum[compact_pos])
        end = int(dataset.sample_cumsum[compact_pos + 1])
        if end <= start:
            continue
        episode = dataset.episodes[int(compact.episode_index)]
        source_info = episode.get('source_info') or {}
        name = str(source_info.get('source_dataset') or source_info.get('raw_dataset') or '__unknown__')
        counts[name] = counts.get(name, 0) + int(end - start)
    return counts
