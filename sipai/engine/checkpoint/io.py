from __future__ import annotations
import hashlib
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import torch
from sipai.utils import ensure_shared_artifact_directory, normalize_shared_artifact_permissions, normalize_shared_artifact_tree
try:
    import numpy as np
except Exception:
    np = None
CHECKPOINT_MANIFEST_FORMAT_VERSION = '1.0'
CHECKPOINT_MANIFEST_NAME = 'manifest.json'
STATE_ACTION_STATISTICS_FILENAME = 'state_action_statistics.json'

@dataclass(frozen=True)
class _DistributedContext:
    rank: int
    world_size: int
    process_group: Any | None
    backend: str

class _OpaquePayload:

    def __init__(self, value: Any) -> None:
        self.value = value

class _TrainerState:

    def __init__(self, value: Mapping[str, Any] | None=None) -> None:
        self.value = dict(value or {})

    def state_dict(self) -> dict[str, Any]:
        return {'payload': _OpaquePayload(self.value)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        payload = state_dict['payload']
        self.value = dict(payload.value)

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model

def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if np is not None:
        state['numpy'] = np.random.get_state()
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state

def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state and np is not None:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])

def _seed_rng_state(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)

def _expanded_rank_seed(source_state: Mapping[str, Any], *, saved_world_size: int, rank: int) -> int:
    torch_state = source_state.get('torch')
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError('Checkpoint RNG state is missing the required torch RNG tensor.')
    digest = hashlib.sha256()
    digest.update(b'sipai-dcp-expanded-rank-rng-v1\x00')
    digest.update(saved_world_size.to_bytes(8, byteorder='big', signed=False))
    digest.update(rank.to_bytes(8, byteorder='big', signed=False))
    cpu_state = torch_state.detach().to(device='cpu', dtype=torch.uint8).contiguous().view(-1)
    digest.update(bytes(cpu_state.tolist()))
    return int.from_bytes(digest.digest()[:8], byteorder='big', signed=False) & (1 << 63) - 1

def _restore_rank_rng_state(rng_by_rank: list[Mapping[str, Any]], *, rank: int, saved_world_size: int) -> tuple[dict[str, Any], str]:
    if len(rng_by_rank) != saved_world_size:
        raise ValueError(f'Checkpoint RNG state count does not match its manifest world_size: rng_states={len(rng_by_rank)}, world_size={saved_world_size}.')
    if rank < 0:
        raise ValueError(f'Distributed rank must be non-negative, got {rank}.')
    if rank < saved_world_size:
        state = dict(rng_by_rank[rank])
        _restore_rng_state(state)
        return (state, 'exact')
    source_state = rng_by_rank[rank % saved_world_size]
    seed = _expanded_rank_seed(source_state, saved_world_size=saved_world_size, rank=rank)
    _seed_rng_state(seed)
    return (_rng_state(), 'derived_for_expanded_world')

def _context_value(context: Any, name: str, default: Any) -> Any:
    if context is None:
        return default
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)

def _resolve_distributed_context(distributed_context: Any | None) -> _DistributedContext:
    process_group = _context_value(distributed_context, 'process_group', None)
    backend = str(_context_value(distributed_context, 'checkpoint_backend', 'pytorch'))
    rank = _context_value(distributed_context, 'rank', None)
    world_size = _context_value(distributed_context, 'world_size', None)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank(process_group) if rank is None else rank
        world_size = torch.distributed.get_world_size(process_group) if world_size is None else world_size
    return _DistributedContext(rank=int(0 if rank is None else rank), world_size=int(1 if world_size is None else world_size), process_group=process_group, backend=backend)

def _barrier(context: _DistributedContext) -> None:
    if context.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier(group=context.process_group)

def _all_rank_rng_states(context: _DistributedContext) -> list[dict[str, Any]]:
    local_state = _rng_state()
    if context.world_size <= 1:
        return [local_state]
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RuntimeError('world_size > 1 requires an initialized torch.distributed process group')
    states: list[dict[str, Any] | None] = [None] * context.world_size
    torch.distributed.all_gather_object(states, local_state, group=context.process_group)
    return [state for state in states if state is not None]

def write_checkpoint_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    payload = _validate_checkpoint_manifest(manifest)
    checkpoint_dir = ensure_shared_artifact_directory(path)
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    normalize_shared_artifact_permissions(manifest_path)
    return manifest_path

def load_checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    manifest_path = checkpoint_path if checkpoint_path.name == CHECKPOINT_MANIFEST_NAME else checkpoint_path / CHECKPOINT_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    return _validate_checkpoint_manifest(payload)

def _validate_checkpoint_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    required = {'format_version', 'backend', 'format', 'step', 'world_size'}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Checkpoint manifest is missing required fields: {', '.join(missing)}")
    if payload['format_version'] != CHECKPOINT_MANIFEST_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint manifest format_version: {payload['format_version']!r}")
    if not isinstance(payload['backend'], str) or not payload['backend']:
        raise ValueError('Checkpoint manifest backend must be a non-empty string.')
    if not isinstance(payload['format'], str) or not payload['format']:
        raise ValueError('Checkpoint manifest format must be a non-empty string.')
    if not isinstance(payload['world_size'], int) or payload['world_size'] <= 0:
        raise ValueError('Checkpoint manifest world_size must be a positive integer.')
    if payload['step'] is not None and (not isinstance(payload['step'], int)):
        raise ValueError('Checkpoint manifest step must be an integer or null.')
    return payload

def save_dcp_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None=None, scheduler: Any | None=None, step: int=0, extra: Mapping[str, Any] | None=None, distributed_context: Any | None=None, checkpoint_backend: str | None=None, ema: Any | None=None) -> Path:
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
    except ImportError as exc:
        raise ImportError('PyTorch DCP checkpointing requires torch.distributed.checkpoint.') from exc
    checkpoint_path = Path(path)
    context = _resolve_distributed_context(distributed_context)
    if 'state_action_representation' in (extra or {}):
        raise ValueError('Normalization statistics must not be embedded in DCP checkpoint metadata; save config.yaml and state_action_statistics.json beside the checkpoint instead.')
    ensure_shared_artifact_directory(checkpoint_path)
    optimizers: Any = optimizer if optimizer is not None else []
    (model_state, optimizer_state) = get_state_dict(model, optimizers)
    trainer_state = {'step': int(step), 'extra': dict(extra or {}), 'scheduler': scheduler.state_dict() if scheduler is not None else None, 'rng_by_rank': _all_rank_rng_states(context)}
    state_dict = {'model': model_state, 'optimizer': optimizer_state, 'trainer': _TrainerState(trainer_state)}
    if ema is not None:
        state_dict['ema'] = ema
    dcp.save(state_dict, checkpoint_id=checkpoint_path, process_group=context.process_group)
    if context.rank == 0:
        manifest = {'format_version': CHECKPOINT_MANIFEST_FORMAT_VERSION, 'backend': checkpoint_backend or context.backend, 'format': 'torch_dcp', 'step': int(step), 'world_size': context.world_size}
        if ema is not None:
            manifest['ema'] = {'enabled': True, 'decay': float(ema.decay)}
        write_checkpoint_manifest(checkpoint_path, manifest)
    _barrier(context)
    if context.rank == 0:
        normalize_shared_artifact_tree(checkpoint_path)
    return checkpoint_path

def load_dcp_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None=None, scheduler: Any | None=None, distributed_context: Any | None=None, restore_rng: bool=True, ema: Any | None=None) -> dict[str, Any]:
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_model_state_dict, set_optimizer_state_dict, set_state_dict
        from torch.distributed.checkpoint.stateful import Stateful
    except ImportError as exc:
        raise ImportError('PyTorch DCP checkpointing requires torch.distributed.checkpoint.') from exc
    checkpoint_path = Path(path)
    manifest = load_checkpoint_manifest(checkpoint_path)
    if manifest['format'] not in {'torch_dcp', 'pytorch_dcp', 'dcp'}:
        raise ValueError(f"Expected a PyTorch DCP manifest, got format={manifest['format']!r}")
    context = _resolve_distributed_context(distributed_context)
    ema_manifest = manifest.get('ema', {})
    checkpoint_has_ema = bool(isinstance(ema_manifest, Mapping) and ema_manifest.get('enabled', False))
    if checkpoint_has_ema != (ema is not None):
        raise ValueError(f'EMA resume configuration does not match the checkpoint: checkpoint_enabled={checkpoint_has_ema}, configured_enabled={ema is not None}.')
    optimizers: Any = optimizer if optimizer is not None else []
    (model_state, optimizer_state) = get_state_dict(model, optimizers)
    trainer = _TrainerState()
    state_dict = {'model': model_state, 'optimizer': optimizer_state, 'trainer': trainer}
    if ema is not None:
        state_dict['ema'] = ema
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    sparse_optimizer_parameters = _validate_sparse_dcp_optimizer_state(state_dict, metadata, planner_type=DefaultLoadPlanner, stateful_type=Stateful)
    dcp.load(state_dict, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=bool(sparse_optimizer_parameters)), process_group=context.process_group)
    if sparse_optimizer_parameters:
        loaded_optimizer_state = optimizer_state.get('state', {})
        for parameter_name in sparse_optimizer_parameters:
            loaded_optimizer_state.pop(parameter_name, None)
        if context.rank == 0:
            warnings.warn(f'Restored a legacy sparse DCP optimizer state for {len(sparse_optimizer_parameters)} parameter(s); model, EMA, trainer, and partial per-parameter optimizer states were still checked strictly.', RuntimeWarning, stacklevel=2)
    if sparse_optimizer_parameters:
        set_model_state_dict(model, model_state, options=StateDictOptions(strict=True))
        set_optimizer_state_dict(model, optimizers, optimizer_state, options=StateDictOptions(strict=False))
    else:
        set_state_dict(model, optimizers, model_state_dict=model_state, optim_state_dict=optimizer_state)
    trainer_state = dict(trainer.value)
    if scheduler is not None and trainer_state.get('scheduler') is not None:
        scheduler.load_state_dict(trainer_state['scheduler'])
    rng_by_rank = trainer_state.get('rng_by_rank', [])
    restored_rng = None
    rng_restore_mode = 'not_present'
    if restore_rng and rng_by_rank:
        (restored_rng, rng_restore_mode) = _restore_rank_rng_state(rng_by_rank, rank=context.rank, saved_world_size=int(manifest['world_size']))
    elif rng_by_rank:
        rng_restore_mode = 'disabled'
        if context.rank < len(rng_by_rank):
            restored_rng = rng_by_rank[context.rank]
    return {'step': int(trainer_state.get('step', manifest['step'])), 'extra': dict(trainer_state.get('extra', {})), 'scheduler': trainer_state.get('scheduler'), 'rng': restored_rng, 'rng_restore_mode': rng_restore_mode, 'manifest': manifest}

def _validate_sparse_dcp_optimizer_state(state_dict: Mapping[str, Any], metadata: Any, *, planner_type: type[Any], stateful_type: type[Any]) -> tuple[str, ...]:
    materialized_state = {key: value.state_dict() if isinstance(value, stateful_type) else value for (key, value) in state_dict.items()}
    planner = planner_type(allow_partial_load=True)
    planner.set_up_planner(materialized_state, metadata, is_coordinator=False)
    planner.create_local_plan()
    checkpoint_keys = {str(key) for key in metadata.state_dict_metadata}
    target_keys = {str(key) for key in planner.state_dict}
    missing_keys = target_keys - checkpoint_keys
    if not missing_keys:
        return ()
    mappings = planner.mappings
    missing_by_parameter: dict[str, set[str]] = {}
    invalid_missing: set[str] = set()
    for key in missing_keys:
        path = mappings.get(key, ())
        if len(path) < 4 or path[0] != 'optimizer' or path[1] != 'state' or (not isinstance(path[2], str)):
            invalid_missing.add(key)
            continue
        missing_by_parameter.setdefault(path[2], set()).add(key)
    partial_parameters: list[str] = []
    for parameter_name in missing_by_parameter:
        expected_keys = {key for (key, path) in mappings.items() if len(path) >= 4 and path[0] == 'optimizer' and (path[1] == 'state') and (path[2] == parameter_name)}
        if expected_keys & checkpoint_keys:
            partial_parameters.append(parameter_name)
    if invalid_missing or partial_parameters:
        details: list[str] = []
        if invalid_missing:
            details.append(f'non-optimizer keys={sorted(invalid_missing)[:8]}')
        if partial_parameters:
            details.append(f'partial optimizer parameters={sorted(partial_parameters)[:8]}')
        raise RuntimeError('DCP checkpoint is missing required state. Sparse compatibility only permits an entire per-parameter optimizer state to be absent; ' + '; '.join(details))
    return tuple(sorted(missing_by_parameter))

def save_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None=None, scheduler: Any | None=None, step: int=0, extra: Mapping[str, Any] | None=None, save_rng: bool=True, ema: Any | None=None) -> Path:
    if 'state_action_representation' in (extra or {}):
        raise ValueError('Normalization statistics must not be embedded in checkpoint metadata; save config.yaml and state_action_statistics.json beside the checkpoint instead.')
    checkpoint_path = Path(path)
    ensure_shared_artifact_directory(checkpoint_path.parent)
    state = {'model': _unwrap_model(model).state_dict(), 'step': int(step), 'extra': dict(extra or {})}
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if save_rng:
        state['rng'] = _rng_state()
    if ema is not None:
        state['ema'] = ema.state_dict()
    torch.save(state, checkpoint_path)
    normalize_shared_artifact_permissions(checkpoint_path)
    return checkpoint_path

def load_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None=None, scheduler: Any | None=None, map_location: str | torch.device='cpu', restore_rng: bool=True, ema: Any | None=None) -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    checkpoint_has_ema = 'ema' in state
    if checkpoint_has_ema != (ema is not None):
        raise ValueError(f'EMA resume configuration does not match the checkpoint: checkpoint_enabled={checkpoint_has_ema}, configured_enabled={ema is not None}.')
    _unwrap_model(model).load_state_dict(state['model'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
    if restore_rng and 'rng' in state:
        _restore_rng_state(state['rng'])
    if ema is not None:
        ema.load_state_dict(state['ema'])
    return state
