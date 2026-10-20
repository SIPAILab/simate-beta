from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from omegaconf import OmegaConf
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import sipai.models
from sipai.config import SIPAI_CONFIG_SCHEMA_VERSION, load_sipai_config_file, resolve_state_action_statistics_path, normalize_sipai_config
from sipai.data import build_canonical_dataloader, build_or_load_dataset_manifest, load_dataset_manifest, sample_dataset_manifest_episodes
from sipai.engine.checkpoint import STATE_ACTION_STATISTICS_FILENAME
from sipai.engine.distributed import DistributedContext, barrier, destroy_distributed, init_distributed_from_env, resolve_distributed_context_from_env
from sipai.engine.hooks import HistoryHook
from sipai.engine.runners import TrainRunner
from sipai.models import build_model_from_sipai_config
from sipai.models.checkpoint_mappers import load_init_checkpoint
from sipai.utils import ensure_shared_artifact_directory, normalize_shared_artifact_permissions
TRAIN_SUMMARY_SCHEMA_VERSION = 'v0.2'

def _get(node: Any, key: str, default: Any=None) -> Any:
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)

def _to_plain_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True) or {}
    if isinstance(node, Mapping):
        return {key: _to_plain_value(value) for (key, value) in node.items()}
    if hasattr(node, 'items'):
        return {key: _to_plain_value(value) for (key, value) in node.items()}
    return dict(vars(node))

def _to_plain_value(node: Any) -> Any:
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    if isinstance(node, Mapping):
        return {key: _to_plain_value(value) for (key, value) in node.items()}
    if isinstance(node, list):
        return [_to_plain_value(value) for value in node]
    if isinstance(node, tuple):
        return tuple((_to_plain_value(value) for value in node))
    return node

def _source_schema_paths(data_root: Path, sources: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for (index, source) in enumerate(sources):
        if not isinstance(source, Mapping) or not source.get('root'):
            raise ValueError(f'Federated source {index} must define a root before schema projection.')
        source_root = Path(str(source['root'])).expanduser()
        if not source_root.is_absolute():
            source_root = data_root / source_root
        schema_path = source_root / 'schema.json'
        if not schema_path.is_file():
            raise FileNotFoundError(f"Federated source {source.get('name', index)!r} schema does not exist: {schema_path}")
        paths.append(schema_path)
    return paths

def _project_representation_streams(*, data_root: Path, sources: list[Any], source_paths: list[str]) -> list[str]:
    if not source_paths:
        return []
    if sources:
        schema_paths = _source_schema_paths(data_root, sources)
    else:
        schema_path = data_root / 'schema.json'
        schema_paths = [schema_path] if schema_path.is_file() else []
    if not schema_paths:
        return []
    projections: list[list[str]] = []
    for schema_path in schema_paths:
        schema_payload = json.loads(schema_path.read_text(encoding='utf-8'))
        available_state = tuple((schema_payload.get('state') or {}).keys())
        projected: list[str] = []
        for source in source_paths:
            matches = [key for key in available_state if source == key or source.startswith(f'{key}.')]
            if not matches:
                raise ValueError(f'Representation source {source!r} does not match a canonical state stream in {schema_path}.')
            projected.append(max(matches, key=len))
        projections.append(sorted(set(projected)))
    expected = projections[0]
    for (schema_path, projected) in zip(schema_paths[1:], projections[1:]):
        if projected != expected:
            raise ValueError(f'Federated representation fields project to different canonical streams: expected {expected}, got {projected} in {schema_path}.')
    return expected

def _checkpoint_export_settings(cfg: Any) -> tuple[bool, bool, str, str]:
    trainer_cfg = _get(cfg, 'trainer', None)
    authored_train_cfg = _get(cfg, 'train_cfg', None)
    selected_cfg = trainer_cfg if trainer_cfg is not None else authored_train_cfg
    selected_cfg = selected_cfg or {}
    export_enabled = bool(_get(selected_cfg, 'export_final_checkpoint', True))
    default_hooks = _get(cfg, 'default_hooks', {}) or {}
    checkpoint_hook = _get(default_hooks, 'checkpoint', {}) or {}
    default_save_last = bool(_get(checkpoint_hook, 'save_last', True))
    save_final = bool(_get(selected_cfg, 'save_final_checkpoint', _get(selected_cfg, 'save_last', default_save_last)))
    weights = _get(selected_cfg, 'export_final_weights', 'ema')
    if not isinstance(weights, str) or weights not in {'ema', 'raw'}:
        raise ValueError(f"train_cfg.export_final_weights must be 'ema' or 'raw', got {weights!r}.")
    save_interval = int(_get(selected_cfg, 'save_interval', 0) or 0)
    will_save_checkpoint = save_final or save_interval > 0
    if export_enabled and will_save_checkpoint and (weights == 'ema') and (_get(selected_cfg, 'ema_decay', None) is None):
        raise ValueError("Automatic checkpoint export defaults to EMA weights, but EMA is not configured. Set train_cfg.ema_decay, set train_cfg.export_final_weights='raw', or disable train_cfg.export_final_checkpoint.")
    export_dtype = _get(selected_cfg, 'export_final_dtype', 'fp32')
    if not isinstance(export_dtype, str) or export_dtype not in {'fp32', 'bf16', 'fp16'}:
        raise ValueError(f"train_cfg.export_final_dtype must be 'fp32', 'bf16', or 'fp16', got {export_dtype!r}.")
    return (export_enabled, save_final, weights, export_dtype)

def _run_export_subprocess(*, checkpoint_path: Path, output_path: Path, weights: str, export_dtype: str) -> dict[str, Any]:
    command = [sys.executable, str(REPO_ROOT / 'tools' / 'export_checkpoint.py'), '--input', str(checkpoint_path), '--output', str(output_path), '--input-format', 'auto', '--output-format', 'safetensors', '--weights', weights, '--dtype', export_dtype, '--no-portable-companions']
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f'Checkpoint export failed for {checkpoint_path} with exit code {completed.returncode}: {detail}')
    try:
        return dict(json.loads(completed.stdout))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'Checkpoint export returned invalid JSON for {checkpoint_path}: {completed.stdout!r}') from exc

class _AutomaticCheckpointExporter:

    def __init__(self, *, output_dir: Path, experiment_name: str, weights: str, export_dtype: str):
        if not experiment_name or Path(experiment_name).name != experiment_name or experiment_name in {'.', '..'}:
            raise ValueError('experiment.name must be a non-empty single path component for automatic export.')
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.weights = weights
        self.export_dtype = export_dtype
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sipai-checkpoint-export')
        self._jobs: list[tuple[int, Path, Path, Future[dict[str, Any]]]] = []
        self._closed = False

    def schedule(self, checkpoint_path: Path, step: int) -> None:
        if self._closed:
            raise RuntimeError('Automatic checkpoint exporter is already closed.')
        checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=False)
        output_path = (self.output_dir / f'{self.experiment_name}-{int(step)}.safetensors').expanduser().resolve(strict=False)
        future = self._executor.submit(_run_export_subprocess, checkpoint_path=Path(checkpoint_path), output_path=output_path, weights=self.weights, export_dtype=self.export_dtype)
        self._jobs.append((int(step), Path(checkpoint_path), output_path, future))

    def finish(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            for (step, checkpoint_path, output_path, future) in self._jobs:
                summary = future.result()
                records.append({'step': step, 'checkpoint': str(checkpoint_path), 'output': str(output_path), **summary})
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True
        return records

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True

def prepare_state_action_statistics(cfg, *, config_path=None, generate_if_missing=False):
    base_dir = Path(config_path).expanduser().resolve().parent if config_path is not None else None
    statistics_path = resolve_state_action_statistics_path(cfg, config_base_dir=base_dir)
    if statistics_path is not None and (not statistics_path.is_file()):
        raise FileNotFoundError(f'Supply the frozen normalization statistics required by this recipe: {statistics_path}')
    return (normalize_sipai_config(cfg, config_base_dir=base_dir), statistics_path, False)

def load_config(config_path: str | Path, overrides: list[str] | None=None, *, generate_statistics_if_missing: bool=False) -> Any:
    cfg = load_sipai_config_file(config_path, overrides)
    (runtime_cfg, _, _) = prepare_state_action_statistics(cfg, config_path=config_path, generate_if_missing=generate_statistics_if_missing)
    return runtime_cfg

def build_train_dataloader(cfg: Any, *, distributed_context: DistributedContext | None=None) -> Any:
    data_cfg = cfg.datasets.vla_data
    dataset_py = str(_get(data_cfg, 'dataset_py', 'CanonicalEmbodiedDataset'))
    if dataset_py not in {'CanonicalEmbodiedDataset', 'canonical_embodied', 'canonical'}:
        raise ValueError(f'SiPAI tools/train.py currently supports canonical embodied dataloaders only. Got datasets.vla_data.dataset_py={dataset_py!r}.')
    manifest_builder = str(_get(data_cfg, 'manifest_builder', 'simate_unified')).lower()
    if manifest_builder != 'simate_unified':
        raise ValueError(f"SiPAI training now supports only datasets.vla_data.manifest_builder='simate_unified'. Got {manifest_builder!r}. Move legacy converters to parity/debug tools if needed.")
    reader_backend = str(_get(data_cfg, 'reader_backend', 'simate_unified')).lower()
    if reader_backend != 'simate_unified':
        raise ValueError(f"SiPAI training now supports only datasets.vla_data.reader_backend='simate_unified'. Got {reader_backend!r}.")
    data_root = Path(str(_get(data_cfg, 'data_root', _get(data_cfg, 'data_root_dir', '.')))).expanduser()
    sources = _to_plain_value(_get(data_cfg, 'sources', None)) or []
    max_episodes = _get(data_cfg, 'manifest_max_episodes', None)
    max_episodes = int(max_episodes) if max_episodes is not None else None
    manifest_episode_glob = _get(data_cfg, 'manifest_episode_glob', None)
    manifest_split = _get(data_cfg, 'manifest_split', None)
    manifest_cache_path = _get(data_cfg, 'manifest_cache_path', None)
    manifest_cache_policy = str(_get(data_cfg, 'manifest_cache_policy', 'readonly'))
    if manifest_cache_policy != 'readonly' or not manifest_cache_path:
        raise ValueError('This release requires a prepared readonly manifest.')
    manifest_sample_episodes = _get(data_cfg, 'manifest_sample_episodes', None)
    manifest_sample_episodes = int(manifest_sample_episodes) if manifest_sample_episodes is not None else None
    seed = int(_get(cfg, 'seed', _get(_get(cfg, 'experiment', None), 'seed', 42)))
    manifest_sample_seed = int(_get(data_cfg, 'manifest_sample_seed', seed))
    framework_cfg = _get(cfg, 'framework', None)
    backbone_cfg = _get(framework_cfg, 'backbone', {}) or {}
    camera_names = list(_get(backbone_cfg, 'input_cameras', []) or [])
    synthetic_cameras = set(_get(backbone_cfg, 'synthetic_masked_cameras', []) or [])
    camera_names = [name for name in camera_names if name not in synthetic_cameras]
    required_streams = [f'camera.{camera_name}' for camera_name in camera_names]
    action_cfg = _get(data_cfg, 'action_config', {}) or {}
    derivation_cfg = _get(action_cfg, 'derivation', {}) or {}
    state_keys = list(_get(derivation_cfg, 'state_keys', []) or [])
    gripper_state_keys = list(_get(derivation_cfg, 'gripper_state_keys', []) or [])
    required_streams.extend(state_keys)
    required_streams.extend(gripper_state_keys)
    representation_cfg = _get(data_cfg, 'state_action_representation', None)
    representation_action_source: str | None = None
    if representation_cfg is not None:
        source_paths: list[str] = []
        for block_name in ('state', 'action'):
            block = _get(representation_cfg, block_name, {}) or {}
            if block_name == 'action':
                representation_action_source = str(_get(block, 'source', 'stored')).lower()
            fields = list(_get(block, 'fields', []) or [])
            typed_stored_action = block_name == 'action' and str(_get(block, 'source', '')).lower() == 'stored' and bool(fields)
            for field in fields:
                source = _get(field, 'anchor_source', None) if typed_stored_action else _get(field, 'source', None)
                if source:
                    source_paths.append(str(source))
        required_streams.extend(_project_representation_streams(data_root=data_root, sources=sources, source_paths=source_paths))
    required_streams = sorted(set(required_streams))
    action_step_interval = _get(derivation_cfg, 'action_step_interval', None)
    if action_step_interval is not None:
        sample_window_size = int(_get(data_cfg, 'sample_window_size', 1))
        model_cfg = _get(cfg, 'model', None)
        action_head_cfg = _get(model_cfg, 'action_head', None)
        framework_action_cfg = _get(framework_cfg, 'action_model', None)
        action_horizon = int(_get(action_head_cfg, 'horizon', _get(action_head_cfg, 'action_horizon', _get(framework_action_cfg, 'action_horizon', 1))))
        min_window_size = action_horizon * int(action_step_interval) + 1
        if sample_window_size < min_window_size:
            raise ValueError(f'Sequential downsampled action requires sample_window_size >= action_horizon * action_step_interval + 1 ({min_window_size}), got sample_window_size={sample_window_size}, action_horizon={action_horizon}, action_step_interval={action_step_interval}.')
    cache_key = {'builder': manifest_builder, 'data_root': str(data_root), 'max_episodes': max_episodes, 'sample_episodes': manifest_sample_episodes, 'sample_seed': manifest_sample_seed, 'episode_glob': manifest_episode_glob, 'split': manifest_split, 'required_streams': required_streams, 'format': 'simate_unified_sharded_v2'}
    if sources:
        cache_key['sources'] = sources
        cache_key['format'] = 'simate_unified_federated_v1'
        cache_key['schema_compatibility'] = 'required_streams'
        cache_key['action_source'] = representation_action_source

    def build_manifest() -> dict[str, Any]:
        raise ValueError('Supply a prepared manifest with cache_policy=readonly.')
    if manifest_cache_path and distributed_context is not None and distributed_context.enabled:
        if distributed_context.is_main_process:
            manifest = build_or_load_dataset_manifest(cache_path=manifest_cache_path, cache_policy=manifest_cache_policy, builder=build_manifest, cache_key=cache_key)
        barrier(distributed_context)
        if not distributed_context.is_main_process:
            manifest = load_dataset_manifest(manifest_cache_path)
    else:
        manifest = build_or_load_dataset_manifest(cache_path=manifest_cache_path, cache_policy=manifest_cache_policy, builder=build_manifest, cache_key=cache_key)
    dataloader = build_canonical_dataloader(cfg, dataset_manifest=manifest, distributed_context=distributed_context)
    sampler = getattr(dataloader, 'sampler', None)
    sampler_report = {'type': type(sampler).__name__ if sampler is not None else None}
    source_sampling_weights = getattr(sampler, 'source_sampling_weights', None)
    if callable(source_sampling_weights):
        source_sampling_weights = source_sampling_weights()
    if source_sampling_weights is not None:
        sampler_report['source_sampling_weights'] = _to_plain_value(source_sampling_weights)
    return (dataloader, {'dataset_py': dataset_py, 'manifest_builder': manifest_builder, 'manifest_cache_path': str(manifest_cache_path) if manifest_cache_path else None, 'manifest_cache_policy': manifest_cache_policy, 'manifest_episode_glob': manifest_episode_glob, 'manifest_split': manifest_split, 'manifest_max_episodes': max_episodes, 'manifest_sample_episodes': manifest_sample_episodes, 'manifest_sample_seed': manifest_sample_seed, 'required_streams': required_streams, 'data_root': str(data_root), 'sources': sources, 'manifest_episodes': len(manifest.get('episodes', [])), 'dataset_len': len(dataloader.dataset), 'sampler': sampler_report, 'batch_size': int(_get(data_cfg, 'per_device_batch_size', 1)), 'output_format': str(_get(data_cfg, 'output_format', 'canonical'))})

def normalize_optimizer_cfg(cfg: Any) -> dict[str, Any]:
    trainer_cfg = _get(cfg, 'trainer', None)
    optimizer_cfg = _to_plain_dict(_get(trainer_cfg, 'optimizer', {}))
    optimizer_name = optimizer_cfg.pop('name', None)
    if 'type' not in optimizer_cfg and optimizer_name is not None:
        optimizer_cfg['type'] = optimizer_name
    learning_rate = _get(trainer_cfg, 'learning_rate', None)
    if 'lr' not in optimizer_cfg:
        optimizer_cfg['lr'] = float(_get(learning_rate, 'base', 0.0001))
    if 'weight_decay' not in optimizer_cfg:
        optimizer_cfg['weight_decay'] = float(_get(trainer_cfg, 'weight_decay', 0.0))
    paramwise_lrs = {str(key): float(value) for (key, value) in _to_plain_dict(learning_rate).items() if str(key) != 'base'}
    if paramwise_lrs:
        optimizer_cfg['paramwise_lrs'] = paramwise_lrs
    return optimizer_cfg

def normalize_scheduler_cfg(cfg: Any) -> dict[str, Any] | None:
    trainer_cfg = _get(cfg, 'trainer', None)
    scheduler_type = _get(trainer_cfg, 'lr_scheduler_type', None)
    if scheduler_type is None:
        return None
    scheduler_type = str(scheduler_type)
    if scheduler_type.lower() == 'constant':
        return {'type': 'constant'}
    scheduler_cfg = _to_plain_dict(_get(trainer_cfg, 'scheduler', {}))
    scheduler_cfg.setdefault('type', scheduler_type)
    scheduler_specific_kwargs = _to_plain_dict(_get(trainer_cfg, 'scheduler_specific_kwargs', {}))
    scheduler_cfg.update(scheduler_specific_kwargs)
    scheduler_cfg.setdefault('num_warmup_steps', int(_get(trainer_cfg, 'num_warmup_steps', 0)))
    scheduler_cfg.setdefault('num_training_steps', int(_get(trainer_cfg, 'max_train_steps', 1)))
    return scheduler_cfg

def normalize_strategy_cfg(cfg: Any) -> dict[str, Any]:
    engine_cfg = _get(cfg, 'engine', None)
    strategy_cfg = _to_plain_dict(_get(engine_cfg, 'strategy', {}))
    if not strategy_cfg:
        strategy_cfg = {'type': 'FSDPStrategy', 'mode': 'hybrid_shard'}
    trainer_cfg = _get(cfg, 'trainer', None)
    gradient_accumulation_steps = _get(trainer_cfg, 'gradient_accumulation_steps', None)
    if gradient_accumulation_steps is not None and strategy_cfg.get('gradient_accumulation_steps', None) is None:
        strategy_cfg['gradient_accumulation_steps'] = int(gradient_accumulation_steps)
    max_grad_norm = _get(trainer_cfg, 'max_grad_norm', None)
    if max_grad_norm is not None and strategy_cfg.get('max_grad_norm', None) is None:
        strategy_cfg['max_grad_norm'] = float(max_grad_norm)
    ema_decay = _get(trainer_cfg, 'ema_decay', None)
    if ema_decay is not None and strategy_cfg.get('ema_decay', None) is None:
        strategy_cfg['ema_decay'] = float(ema_decay)
    precision = _get(engine_cfg, 'precision', None)
    if precision is not None and strategy_cfg.get('precision', None) is None:
        strategy_cfg['precision'] = str(precision).lower()
    strategy_type = str(strategy_cfg.get('type', 'FSDPStrategy'))
    return strategy_cfg

def build_training_hooks(cfg, *, run_id, run_root):
    trainer_cfg = _get(cfg, 'trainer', None)
    history_hook = HistoryHook() if bool(_get(trainer_cfg, 'record_history', False)) else None
    return ([history_hook] if history_hook is not None else [], history_hook, None)

def checkpoint_backend_from_strategy(strategy_cfg: Mapping[str, Any]) -> str:
    strategy_type = str(strategy_cfg.get('type', 'FSDPStrategy'))
    if strategy_type == 'FSDPStrategy':
        return 'torch_dcp'
    return 'torch_full_state'

def _module_by_dotted_name(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    current: Any = model
    for part in name.split('.'):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current if isinstance(current, torch.nn.Module) else None

def freeze_modules(model: torch.nn.Module, freeze_spec: str | None) -> list[str]:
    if not freeze_spec:
        return []
    aliases = {'vlm': 'backbone.vlm'}
    frozen = []
    module_names = [item.strip() for item in str(freeze_spec).replace(',', ' ').split() if item.strip()]
    for requested_name in module_names:
        module_name = aliases.get(requested_name, requested_name)
        module = _module_by_dotted_name(model, module_name)
        if module is None:
            raise ValueError(f'Cannot freeze module {requested_name!r}; resolved {module_name!r} was not found.')
        for param in module.parameters():
            param.requires_grad_(False)
        frozen.append(module_name)
    return frozen

def _default_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def _authored_config_copy(cfg: Any) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if isinstance(cfg, Mapping):
        return OmegaConf.create(deepcopy(dict(cfg)))
    raise TypeError('Training config must be a mapping or OmegaConf config.')

def _wait_for_preflight_statistics(cfg: Any, *, config_path: str | Path | None, poll_interval_seconds: float=1.0) -> Path | None:
    base_dir = Path(config_path).expanduser().resolve(strict=False).parent if config_path is not None else None
    statistics_path = resolve_state_action_statistics_path(cfg, config_base_dir=base_dir)
    if statistics_path is None:
        return None
    while not statistics_path.exists():
        time.sleep(poll_interval_seconds)
    if not statistics_path.is_file():
        raise ValueError(f'State/action statistics path is not a file: {statistics_path}')
    return statistics_path

def _portable_training_config(authored_cfg: Any, *, has_statistics: bool) -> Any:
    portable = _authored_config_copy(authored_cfg)
    if not has_statistics:
        return portable
    pipeline = OmegaConf.select(portable, 'data.train.pipeline')
    if pipeline is not None:
        pipeline.state_action_statistics = STATE_ACTION_STATISTICS_FILENAME
    return portable

def _atomic_write_text(path: Path, text: str) -> None:
    ensure_shared_artifact_directory(path.parent)
    (fd, temporary_name) = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        normalize_shared_artifact_permissions(temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def _write_training_companions(authored_cfg: Any, *, statistics_path: Path | None, run_dir: Path) -> tuple[Path, Path | None]:
    config_destination = run_dir / 'config.yaml'
    portable_cfg = _portable_training_config(authored_cfg, has_statistics=statistics_path is not None)
    _atomic_write_text(config_destination, OmegaConf.to_yaml(portable_cfg, resolve=False))
    statistics_destination = None
    if statistics_path is not None:
        statistics_destination = run_dir / STATE_ACTION_STATISTICS_FILENAME
        ensure_shared_artifact_directory(statistics_destination.parent)
        if statistics_path.resolve() != statistics_destination.resolve():
            temporary = statistics_destination.with_name(f'.{statistics_destination.name}.tmp')
            shutil.copy2(statistics_path, temporary)
            normalize_shared_artifact_permissions(temporary)
            os.replace(temporary, statistics_destination)
        else:
            normalize_shared_artifact_permissions(statistics_destination)
    return (config_destination, statistics_destination)

def train(cfg: Any, *, device: str | None=None, checkpoint_dir: str | Path | None=None, resume_from: str | Path | None=None, launcher: str='none', dist_backend: str | None=None, config_path: str | Path | None=None, overrides: list[str] | None=None) -> dict[str, Any]:
    authored_cfg = _authored_config_copy(cfg)
    (export_checkpoints, save_final_checkpoint, export_weights, export_dtype) = _checkpoint_export_settings(authored_cfg)
    preinit_context = resolve_distributed_context_from_env(launcher=launcher, backend=dist_backend)
    statistics_path: Path | None = None
    statistics_generated = False
    if preinit_context.is_main_process:
        (cfg, statistics_path, statistics_generated) = prepare_state_action_statistics(authored_cfg, config_path=config_path, generate_if_missing=True)
    else:
        statistics_path = _wait_for_preflight_statistics(authored_cfg, config_path=config_path)
        (cfg, statistics_path, _) = prepare_state_action_statistics(authored_cfg, config_path=config_path, generate_if_missing=False)
    distributed_context = init_distributed_from_env(launcher=launcher, backend=dist_backend)
    automatic_exporter: _AutomaticCheckpointExporter | None = None
    try:
        (export_checkpoints, save_final_checkpoint, export_weights, export_dtype) = _checkpoint_export_settings(cfg)
        base_seed = int(_get(cfg, 'seed', _get(_get(cfg, 'experiment', None), 'seed', 42)))
        _set_global_seed(base_seed)
        model = build_model_from_sipai_config(cfg)
        trainer_cfg = cfg.trainer
        resume_path = resume_from or _get(trainer_cfg, 'resume_from', None)
        initialization_report = None
        init_checkpoint_cfg = _get(trainer_cfg, 'init_checkpoint', None)
        if resume_path is None and init_checkpoint_cfg is not None:
            initialization_report = load_init_checkpoint(model, _to_plain_dict(init_checkpoint_cfg)).to_dict()
        frozen_modules = freeze_modules(model, _get(cfg.trainer, 'freeze_modules', ''))
        _set_global_seed((base_seed + int(distributed_context.rank)) % 2 ** 32)
        (dataloader, data_report) = build_train_dataloader(cfg, distributed_context=distributed_context)
        max_steps = int(_get(trainer_cfg, 'max_train_steps', 1))
        run_root = Path(str(_get(cfg, 'run_root_dir', 'results/Checkpoints')))
        run_id = str(_get(cfg, 'run_id', 'sipai_train'))
        resolved_checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else run_root / run_id / 'checkpoints'
        run_dir = resolved_checkpoint_dir.parent if resolved_checkpoint_dir.name == 'checkpoints' else resolved_checkpoint_dir
        config_artifact_path = run_dir / 'config.yaml'
        statistics_artifact_path = run_dir / STATE_ACTION_STATISTICS_FILENAME if statistics_path is not None else None
        if distributed_context.is_main_process:
            (config_artifact_path, statistics_artifact_path) = _write_training_companions(authored_cfg, statistics_path=statistics_path, run_dir=run_dir)
        barrier(distributed_context)
        resolved_device = _default_device(device)
        if distributed_context.enabled and torch.cuda.is_available() and (device is None):
            resolved_device = f'cuda:{distributed_context.local_rank}'
        (hooks, history_hook, wandb_hook) = build_training_hooks(cfg, run_id=run_id, run_root=run_root)
        strategy_cfg = normalize_strategy_cfg(cfg)
        checkpoint_backend = checkpoint_backend_from_strategy(strategy_cfg)
        if export_checkpoints and distributed_context.is_main_process:
            experiment_name = str(_get(_get(authored_cfg, 'experiment', None), 'name', run_id))
            automatic_exporter = _AutomaticCheckpointExporter(output_dir=run_dir / 'exported_ckpts', experiment_name=experiment_name, weights=export_weights, export_dtype=export_dtype)
        runner = TrainRunner(model=model, dataloader=dataloader, optimizer=normalize_optimizer_cfg(cfg), scheduler=normalize_scheduler_cfg(cfg), max_steps=max_steps, hooks=hooks, checkpoint_dir=resolved_checkpoint_dir, save_interval=int(_get(trainer_cfg, 'save_interval', 0) or 0), resume_from=resume_path, device=resolved_device, distributed_context=distributed_context, strategy=strategy_cfg, logging_frequency=int(_get(trainer_cfg, 'logging_frequency', 25) or 0), checkpoint_extra=None, defer_final_interval_checkpoint=save_final_checkpoint, checkpoint_saved_callback=automatic_exporter.schedule if automatic_exporter is not None else None)
        result = runner.train()
        checkpoint_path = None
        if save_final_checkpoint:
            checkpoint_extra = {}
            if initialization_report is not None:
                checkpoint_extra['initialization'] = initialization_report
            checkpoint_path = runner.trainer.save_checkpoint(extra=checkpoint_extra or None)
        export_summaries: list[dict[str, Any]] = []
        if automatic_exporter is not None:
            export_summaries = automatic_exporter.finish()
            automatic_exporter = None
        total_params = sum((param.numel() for param in model.parameters()))
        trainable_params = sum((param.numel() for param in model.parameters() if param.requires_grad))
        summary = {'schema_version': TRAIN_SUMMARY_SCHEMA_VERSION, 'config_schema_version': SIPAI_CONFIG_SCHEMA_VERSION, 'status': 'completed', 'run_id': run_id, 'device': resolved_device, 'completed_steps': result['completed_steps'], 'checkpoint': str(checkpoint_path) if checkpoint_path is not None else None, 'checkpoint_info': {'backend': checkpoint_backend, 'dir': str(resolved_checkpoint_dir), 'final': str(checkpoint_path) if checkpoint_path is not None else None, 'save_interval': int(_get(trainer_cfg, 'save_interval', 0) or 0), 'save_final': bool(_get(trainer_cfg, 'save_final_checkpoint', True)), 'resume_from': str(resume_path) if resume_path is not None else None, 'config': str(config_artifact_path), 'state_action_statistics': str(statistics_artifact_path) if statistics_artifact_path is not None else None, 'statistics_generated': bool(statistics_generated), 'export_checkpoints': export_checkpoints, 'export_weights': export_weights, 'export_dtype': export_dtype, 'export_dir': str(run_dir / 'exported_ckpts') if export_checkpoints else None}, 'exports': export_summaries, 'entrypoint': {'name': 'tools/train.py', 'config_path': str(config_path) if config_path is not None else None, 'overrides': list(overrides or []), 'launcher': launcher, 'dist_backend': dist_backend}, 'frozen_modules': frozen_modules, 'distributed': {'enabled': distributed_context.enabled, 'rank': distributed_context.rank, 'local_rank': distributed_context.local_rank, 'world_size': distributed_context.world_size, 'backend': distributed_context.backend}, 'strategy': strategy_cfg, 'parameters': {'total': total_params, 'trainable': trainable_params}, 'data': data_report, 'initialization': initialization_report}
        last_outputs = result.get('last_outputs') or {}
        if 'action_loss' in last_outputs and hasattr(last_outputs['action_loss'], 'detach'):
            summary['last_action_loss'] = float(last_outputs['action_loss'].detach().cpu())
        if history_hook is not None:
            summary['history'] = history_hook.history
        return summary
    finally:
        if automatic_exporter is not None:
            automatic_exporter.close()
        destroy_distributed(distributed_context)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SiPAI single-process training entrypoint.')
    parser.add_argument('--config', '--config-yaml', dest='config', required=True, help='Path to SiPAI YAML config.')
    parser.add_argument('--device', default=None, help='Training device, e.g. cuda:0 or cpu. Defaults to cuda:0 if available.')
    parser.add_argument('--checkpoint-dir', default=None, help='Override checkpoint output directory.')
    parser.add_argument('--resume-from', default=None, help='Resume model/optimizer/scheduler state from a checkpoint.')
    parser.add_argument('--launcher', default='none', choices=['none', 'pytorch', 'torchrun', 'env'], help='Distributed launcher. Use torchrun with --launcher torchrun for multi-process training.')
    parser.add_argument('--dist-backend', default=None, help='Distributed backend override, e.g. nccl or gloo.')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None, help='OmegaConf dotlist overrides, e.g. --opts train_cfg.max_steps=3 train_dataloader.num_workers=0')
    parser.add_argument('--summary-json', default=None, help='Optional rank-0 JSON path for the resolved training summary.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = load_sipai_config_file(args.config, args.opts)
    summary = train(cfg, device=args.device, checkpoint_dir=args.checkpoint_dir, resume_from=args.resume_from, launcher=args.launcher, dist_backend=args.dist_backend, config_path=args.config, overrides=args.opts)
    if summary['distributed']['rank'] == 0:
        if args.summary_json:
            summary_path = Path(args.summary_json)
            ensure_shared_artifact_directory(summary_path.parent)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
            normalize_shared_artifact_permissions(summary_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
