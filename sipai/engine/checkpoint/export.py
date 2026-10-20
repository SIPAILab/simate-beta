from __future__ import annotations
import inspect
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from sipai.engine.checkpoint.io import CHECKPOINT_MANIFEST_NAME, STATE_ACTION_STATISTICS_FILENAME, load_checkpoint_manifest
from sipai.utils import ensure_shared_artifact_directory, normalize_shared_artifact_permissions
EXPORT_CHECKPOINT_SCHEMA_VERSION = 'v0.1'

def infer_checkpoint_format(path: str | Path) -> str:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        manifest_path = checkpoint_path / CHECKPOINT_MANIFEST_NAME
        if manifest_path.is_file():
            manifest = load_checkpoint_manifest(checkpoint_path)
            manifest_format = str(manifest['format'])
            if manifest_format in {'torch_dcp', 'pytorch_dcp', 'dcp'}:
                return 'torch_dcp'
            if manifest_format == 'sharded_safetensors':
                return 'sharded_safetensors'
            raise ValueError(f'Unsupported checkpoint manifest format: {manifest_format!r}')
        if (checkpoint_path / '.metadata').is_file():
            return 'torch_dcp'
        if (checkpoint_path / 'model.safetensors.index.json').is_file():
            return 'sharded_safetensors'
        raise ValueError(f'Cannot infer checkpoint format for directory without {CHECKPOINT_MANIFEST_NAME}: {checkpoint_path}. Pass checkpoint_format explicitly for a legacy checkpoint.')
    if checkpoint_path.is_file():
        if checkpoint_path.suffix.lower() == '.safetensors':
            return 'safetensors'
        return 'torch_full_state'
    raise FileNotFoundError(f'Checkpoint path does not exist: {checkpoint_path}')

def extract_model_state_dict(payload: Any, *, weights: str='raw') -> dict[str, Any]:
    if weights not in {'ema', 'raw'}:
        raise ValueError(f'Unsupported weights selection {weights!r}; expected ema or raw.')
    raw_state: dict[str, Any] | None = None
    if isinstance(payload, Mapping):
        if 'model' in payload and isinstance(payload['model'], Mapping):
            raw_state = dict(payload['model'])
        elif 'state_dict' in payload and isinstance(payload['state_dict'], Mapping):
            raw_state = dict(payload['state_dict'])
        elif payload and all((torch.is_tensor(value) for value in payload.values())):
            raw_state = dict(payload)
    if raw_state is not None:
        if weights == 'raw':
            return raw_state
        ema = payload.get('ema') if isinstance(payload, Mapping) else None
        shadow = ema.get('shadow') if isinstance(ema, Mapping) else None
        if not isinstance(shadow, Mapping):
            raise ValueError('The checkpoint does not contain EMA weights.')
        unknown = sorted(set(shadow) - set(raw_state))
        if unknown:
            raise ValueError(f'EMA checkpoint contains parameters absent from the model state: {unknown[:4]}.')
        raw_state.update(shadow)
        return raw_state
    raise ValueError('Unsupported checkpoint payload. Expected a dict with `model`, `state_dict`, or a plain tensor state_dict.')

def load_torch_state_dict(path: str | Path, *, map_location: str | torch.device='cpu', weights: str='raw') -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return extract_model_state_dict(payload, weights=weights)

def _extract_torch_checkpoint_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    if isinstance(payload.get('metadata'), Mapping):
        metadata['metadata'] = dict(payload['metadata'])
    if isinstance(payload.get('extra'), Mapping):
        metadata['extra'] = dict(payload['extra'])
    if 'step' in payload:
        metadata['step'] = int(payload['step'])
    return metadata

def load_torch_checkpoint_metadata(path: str | Path, *, map_location: str | torch.device='cpu') -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return _extract_torch_checkpoint_metadata(payload)

def load_dcp_state_dict(path: str | Path, *, map_location: str | torch.device='cpu', weights: str='raw', tensor_dtype: torch.dtype | None=None) -> dict[str, Any]:
    if torch.device(map_location).type != 'cpu':
        raise ValueError("DCP weight export currently supports map_location='cpu' only.")
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
        from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
    except ImportError as exc:
        raise ImportError('PyTorch DCP export requires torch.distributed.checkpoint.') from exc
    if weights not in {'ema', 'raw'}:
        raise ValueError(f'Unsupported weights selection {weights!r}; expected ema or raw.')
    reader = FileSystemReader(Path(path))
    metadata = reader.read_metadata()
    raw_metadata = _dcp_tensor_metadata_by_name(metadata, prefix='model.', tensor_metadata_type=TensorStorageMetadata, bytes_metadata_type=BytesStorageMetadata)
    if not raw_metadata:
        raise ValueError(f'DCP checkpoint does not contain model tensor metadata: {path}')
    selected_metadata = dict(raw_metadata)
    if weights == 'ema':
        ema_metadata = _dcp_tensor_metadata_by_name(metadata, prefix='ema.shadow.', tensor_metadata_type=TensorStorageMetadata, bytes_metadata_type=BytesStorageMetadata)
        if not ema_metadata:
            raise ValueError('The checkpoint does not contain EMA weights.')
        unknown = sorted(set(ema_metadata) - set(raw_metadata))
        if unknown:
            raise ValueError(f'EMA checkpoint contains parameters absent from the model state: {unknown[:4]}.')
        mismatched = sorted((name for (name, (_, ema_tensor_metadata)) in ema_metadata.items() if tuple(ema_tensor_metadata.size) != tuple(raw_metadata[name][1].size)))
        if mismatched:
            raise ValueError(f'EMA checkpoint tensor shapes differ from the model state: {mismatched[:4]}.')
        selected_metadata.update(ema_metadata)
    targets: dict[str, torch.Tensor] = {}
    for (fqn, tensor_metadata) in selected_metadata.values():
        properties = tensor_metadata.properties
        if properties.layout != torch.strided:
            raise ValueError(f'DCP export requires strided model tensors; {fqn!r} uses {properties.layout}.')
        dtype = properties.dtype
        if tensor_dtype is not None and dtype.is_floating_point:
            dtype = tensor_dtype
        targets[fqn] = torch.empty(tuple(tensor_metadata.size), dtype=dtype, device='cpu')
    load_kwargs: dict[str, Any] = {'storage_reader': reader, 'planner': DefaultLoadPlanner(flatten_state_dict=False)}
    if 'no_dist' in inspect.signature(dcp.load).parameters:
        load_kwargs['no_dist'] = True
    dcp.load(targets, **load_kwargs)
    return {name: targets[fqn] for (name, (fqn, _)) in selected_metadata.items()}

def _dcp_tensor_metadata_by_name(metadata: Any, *, prefix: str, tensor_metadata_type: type[Any], bytes_metadata_type: type[Any]) -> dict[str, tuple[str, Any]]:
    selected: dict[str, tuple[str, Any]] = {}
    for (fqn, tensor_metadata) in metadata.state_dict_metadata.items():
        fqn = str(fqn)
        if not fqn.startswith(prefix):
            continue
        name = fqn[len(prefix):]
        if not name:
            raise ValueError(f'DCP checkpoint contains an empty tensor name under {prefix!r}.')
        if isinstance(tensor_metadata, bytes_metadata_type):
            raise ValueError(f'DCP Safetensors export does not support non-tensor model state: {fqn!r}.')
        if not isinstance(tensor_metadata, tensor_metadata_type):
            raise ValueError(f'DCP checkpoint uses unsupported model metadata for {fqn!r}.')
        selected[name] = (fqn, tensor_metadata)
    return selected

def load_safetensors_state_dict(path: str | Path, *, map_location: str | torch.device='cpu') -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError('Loading Safetensors requires the `safetensors` package.') from exc
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Safetensors checkpoint does not exist: {checkpoint_path}')
    return dict(load_file(str(checkpoint_path), device=str(map_location)))

def load_sharded_safetensors_state_dict(path: str | Path, *, map_location: str | torch.device='cpu') -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError('Loading Safetensors requires the `safetensors` package.') from exc
    checkpoint_dir = Path(path)
    index_path = checkpoint_dir / 'model.safetensors.index.json'
    if not index_path.is_file():
        raise FileNotFoundError(f'Missing Safetensors index: {index_path}')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f'Invalid Safetensors index without a non-empty weight_map: {index_path}')
    state_dict: dict[str, torch.Tensor] = {}
    for filename in sorted(set((str(value) for value in weight_map.values()))):
        shard_path = checkpoint_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f'Missing Safetensors shard referenced by index: {shard_path}')
        state_dict.update(load_file(str(shard_path), device=str(map_location)))
    missing = sorted(set((str(name) for name in weight_map)) - set(state_dict))
    if missing:
        raise ValueError(f'Safetensors shards are missing indexed tensors: {missing[:20]}')
    return state_dict

def load_checkpoint_state_dict(path: str | Path, *, checkpoint_format: str='auto', tag: str | None=None, map_location: str | torch.device='cpu', weights: str='raw', tensor_dtype: torch.dtype | None=None) -> tuple[dict[str, Any], str]:
    resolved_format = infer_checkpoint_format(path) if checkpoint_format == 'auto' else checkpoint_format
    if weights == 'ema' and resolved_format in {'safetensors', 'sharded_safetensors'}:
        raise ValueError('EMA weight selection requires a training checkpoint that contains EMA state.')
    if resolved_format in {'torch_full_state', 'full', 'state_dict'}:
        return (load_torch_state_dict(path, map_location=map_location, weights=weights), 'torch_full_state')
    if resolved_format in {'torch_dcp', 'pytorch_dcp', 'dcp'}:
        return (load_dcp_state_dict(path, map_location=map_location, weights=weights, tensor_dtype=tensor_dtype), 'torch_dcp')
    if resolved_format == 'safetensors' and Path(path).is_file():
        return (load_safetensors_state_dict(path, map_location=map_location), 'safetensors')
    if resolved_format in {'sharded_safetensors', 'safetensors'}:
        return (load_sharded_safetensors_state_dict(path, map_location=map_location), 'sharded_safetensors')
    raise ValueError(f'Unsupported checkpoint format: {checkpoint_format!r}')

def export_checkpoint(input_path: str | Path, output_path: str | Path, *, checkpoint_format: str='auto', output_format: str='safetensors', tag: str | None=None, map_location: str | torch.device='cpu', metadata: Mapping[str, Any] | None=None, export_dtype: str='fp32', config_path: str | Path | None=None, statistics_path: str | Path | None=None, weights: str='raw', copy_portable_companions: bool=True) -> dict[str, Any]:
    input_checkpoint = Path(input_path)
    requested_format = infer_checkpoint_format(input_checkpoint) if checkpoint_format == 'auto' else checkpoint_format
    if requested_format in {'torch_full_state', 'full', 'state_dict'}:
        torch_payload = torch.load(input_checkpoint, map_location=map_location, weights_only=False)
        state_dict = extract_model_state_dict(torch_payload, weights=weights)
        resolved_format = 'torch_full_state'
        source_metadata = _extract_torch_checkpoint_metadata(torch_payload)
    else:
        load_dtype = _export_torch_dtype(export_dtype) if output_format == 'safetensors' else None
        (state_dict, resolved_format) = load_checkpoint_state_dict(input_checkpoint, checkpoint_format=requested_format, tag=tag, map_location=map_location, weights=weights, tensor_dtype=load_dtype)
        source_metadata = {}
    if resolved_format == 'torch_dcp' and (input_checkpoint / CHECKPOINT_MANIFEST_NAME).is_file():
        source_metadata = {'source_manifest': load_checkpoint_manifest(input_path)}
    output = Path(output_path)
    export_metadata = {'schema_version': EXPORT_CHECKPOINT_SCHEMA_VERSION, 'source': str(Path(input_path)), 'source_format': resolved_format, 'output_format': output_format, 'tag': tag, 'weights': weights, **source_metadata, **dict(metadata or {})}
    export_metadata = _without_embedded_representation(export_metadata)
    if output_format == 'full':
        ensure_shared_artifact_directory(output.parent)
        payload: Any = {'model': state_dict, 'metadata': export_metadata}
        torch.save(payload, output)
        normalize_shared_artifact_permissions(output)
    elif output_format == 'state_dict':
        ensure_shared_artifact_directory(output.parent)
        payload = state_dict
        torch.save(payload, output)
        normalize_shared_artifact_permissions(output)
    elif output_format == 'wrapped_state_dict':
        ensure_shared_artifact_directory(output.parent)
        payload = {'state_dict': state_dict, 'metadata': export_metadata}
        torch.save(payload, output)
        normalize_shared_artifact_permissions(output)
    elif output_format == 'safetensors':
        state_dict = _convert_export_dtype(state_dict, export_dtype)
        _write_safetensors(output, state_dict, export_metadata)
    else:
        raise ValueError(f'Unsupported output format {output_format!r}; expected full, state_dict, wrapped_state_dict, or safetensors.')
    output_dir = output.parent
    companions = _copy_portable_companions(input_path=Path(input_path), output_dir=output_dir, config_path=Path(config_path) if config_path is not None else None, statistics_path=Path(statistics_path) if statistics_path is not None else None) if copy_portable_companions else {}
    return {'schema_version': EXPORT_CHECKPOINT_SCHEMA_VERSION, 'status': 'completed', 'source': str(Path(input_path)), 'source_format': resolved_format, 'output': str(output), 'output_format': output_format, 'num_tensors': len(state_dict), 'num_parameters': int(sum((value.numel() for value in state_dict.values() if torch.is_tensor(value)))), 'tag': tag, 'weights': weights, 'dtype': export_dtype if output_format == 'safetensors' else None, 'config': str(companions.get('config')) if companions.get('config') is not None else None, 'state_action_statistics': str(companions.get('state_action_statistics')) if companions.get('state_action_statistics') is not None else None}

def _convert_export_dtype(state_dict: Mapping[str, Any], export_dtype: str) -> dict[str, torch.Tensor]:
    dtype = _export_torch_dtype(export_dtype)
    converted: dict[str, torch.Tensor] = {}
    for (name, value) in state_dict.items():
        if not torch.is_tensor(value):
            raise ValueError(f'Safetensors export requires tensor values; {name!r} is {type(value).__name__}.')
        tensor = value.detach()
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        if tensor.is_floating_point() and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        converted[name] = tensor
    _clone_shared_export_tensors(converted)
    return converted

def _export_torch_dtype(export_dtype: str) -> torch.dtype:
    dtype_by_name = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    if export_dtype not in dtype_by_name:
        raise ValueError(f'Unsupported export dtype {export_dtype!r}; expected fp32, bf16, or fp16.')
    return dtype_by_name[export_dtype]

def _clone_shared_export_tensors(state_dict: dict[str, torch.Tensor]) -> None:
    owners: dict[tuple[str, int | None, int, int], str] = {}
    for (name, tensor) in state_dict.items():
        if tensor.numel() == 0 or tensor.device.type == 'meta':
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, int(storage.data_ptr()), int(storage.nbytes()))
        if key in owners:
            state_dict[name] = tensor.clone()
        else:
            owners[key] = name

def _copy_portable_companions(*, input_path: Path, output_dir: Path, config_path: Path | None, statistics_path: Path | None) -> dict[str, Path]:
    ensure_shared_artifact_directory(output_dir)
    run_dir = _checkpoint_run_dir(input_path)
    resolved_config = config_path or _first_existing(run_dir / 'config.yaml', input_path / 'config.yaml')
    resolved_statistics = statistics_path or _first_existing(run_dir / STATE_ACTION_STATISTICS_FILENAME, input_path / STATE_ACTION_STATISTICS_FILENAME, run_dir / 'dataset_statistics.json', input_path / 'dataset_statistics.json')
    copied: dict[str, Path] = {}
    if resolved_config is not None:
        destination = output_dir / 'config.yaml'
        _copy_portable_config(resolved_config, destination, has_state_action_statistics=resolved_statistics is not None)
        copied['config'] = destination
    if resolved_statistics is not None:
        destination = output_dir / STATE_ACTION_STATISTICS_FILENAME
        _copy_if_different(resolved_statistics, destination)
        copied['state_action_statistics'] = destination
    return copied

def _without_embedded_representation(metadata: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(metadata)
    extra = cleaned.get('extra')
    if isinstance(extra, Mapping):
        cleaned_extra = dict(extra)
        cleaned_extra.pop('state_action_representation', None)
        cleaned['extra'] = cleaned_extra
    nested = cleaned.get('metadata')
    if isinstance(nested, Mapping):
        cleaned['metadata'] = _without_embedded_representation(nested)
    return cleaned

def _copy_portable_config(source: Path, destination: Path, *, has_state_action_statistics: bool) -> None:
    from omegaconf import OmegaConf
    from sipai.config import load_sipai_config_file
    config = load_sipai_config_file(source)
    public_pipeline = OmegaConf.select(config, 'data.train.pipeline')
    runtime_data = OmegaConf.select(config, 'datasets.vla_data')
    if has_state_action_statistics and public_pipeline is not None:
        public_pipeline.state_action_statistics = STATE_ACTION_STATISTICS_FILENAME
    elif has_state_action_statistics and runtime_data is not None:
        runtime_data.state_action_statistics = STATE_ACTION_STATISTICS_FILENAME
    ensure_shared_artifact_directory(destination.parent)
    OmegaConf.save(config=config, f=destination, resolve=False)
    normalize_shared_artifact_permissions(destination)

def _checkpoint_run_dir(input_path: Path) -> Path:
    return input_path.parent.parent if input_path.parent.name == 'checkpoints' else input_path.parent

def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)

def _copy_if_different(source: Path, destination: Path) -> None:
    ensure_shared_artifact_directory(destination.parent)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    normalize_shared_artifact_permissions(destination)

def _write_safetensors(output: Path, state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError('Safetensors export requires the `safetensors` package.') from exc
    if not state_dict:
        raise ValueError('Cannot export an empty state_dict to Safetensors.')
    if output.suffix.lower() != '.safetensors':
        raise ValueError('Safetensors output path must end with `.safetensors`.')
    ensure_shared_artifact_directory(output.parent)
    file_metadata = {'format': 'pt', 'source_format': str(metadata.get('source_format', 'unknown')), 'schema_version': EXPORT_CHECKPOINT_SCHEMA_VERSION}
    save_file(dict(state_dict), str(output), metadata=file_metadata)
    normalize_shared_artifact_permissions(output)

def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_ready(item) for item in value))
    return value
