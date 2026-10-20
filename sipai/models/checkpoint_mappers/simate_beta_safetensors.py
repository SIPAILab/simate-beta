from __future__ import annotations
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from sipai.models.action_heads.simate_beta import SimateBetaFlowMatchingActionHead
from sipai.models.backbones.simate_beta import SimateBetaPrefixBackbone
from sipai.models.frameworks.vla.composable_vla import ComposableVLAFramework
from sipai.registry import CHECKPOINT_LOADERS
SIPAI_NATIVE_ABI = 'sipai_simate_beta_native_state_dict_v1'

@dataclass(frozen=True)
class SimateBetaLoadReport:
    path: Path
    config_path: Path | None
    source_format: str
    tensor_count: int
    parameter_count: int
    assigned: bool
    loader_type: str = 'SipaiNativeSafetensors'
    initialized_keys: tuple[str, ...] = ()
    supplemental_path: Path | None = None
    supplemental_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {'type': self.loader_type, 'path': str(self.path), 'config_path': str(self.config_path) if self.config_path is not None else None, 'source_format': self.source_format, 'tensor_count': self.tensor_count, 'parameter_count': self.parameter_count, 'assigned': self.assigned, 'initialized_keys': list(self.initialized_keys), 'supplemental_path': str(self.supplemental_path) if self.supplemental_path is not None else None, 'supplemental_keys': list(self.supplemental_keys)}

@CHECKPOINT_LOADERS.register_module(name='SipaiNativeSafetensors')
def load_sipai_native_safetensors(target: ComposableVLAFramework, path: str | Path, *, device: str | torch.device='cpu', initialize_missing_prefixes: Sequence[str]=(), supplemental_checkpoint: str | Path | None=None, supplemental_prefixes: Sequence[str]=()) -> SimateBetaLoadReport:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError('Loading simate-beta weights requires the `safetensors` package.') from exc
    _validate_target(target)
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / 'model.safetensors'
    if checkpoint.suffix != '.safetensors':
        raise ValueError(f'Native SiPAI loader only accepts a .safetensors file or directory, got {checkpoint}.')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Native SiPAI Safetensors checkpoint does not exist: {checkpoint}')
    source = load_file(str(checkpoint), device=str(torch.device(device)))
    target_state = target.state_dict()
    missing_prefixes = _validate_prefixes(initialize_missing_prefixes, name='initialize_missing_prefixes')
    fill_prefixes = _validate_prefixes(supplemental_prefixes, name='supplemental_prefixes')
    if (supplemental_checkpoint is None) != (not fill_prefixes):
        raise ValueError('supplemental_checkpoint and supplemental_prefixes must be configured together.')
    overlapping_prefixes = sorted(set(missing_prefixes) & set(fill_prefixes))
    if overlapping_prefixes:
        raise ValueError(f'Missing prefixes cannot be both initialized and loaded from the supplemental checkpoint: {overlapping_prefixes}.')
    missing = sorted(set(target_state) - set(source))
    unexpected = sorted(set(source) - set(target_state))
    disallowed_missing = [key for key in missing if not any((key.startswith(prefix) for prefix in (*missing_prefixes, *fill_prefixes)))]
    unmatched_prefixes = [prefix for prefix in (*missing_prefixes, *fill_prefixes) if not any((key.startswith(prefix) for key in missing))]
    if disallowed_missing or unexpected or unmatched_prefixes:
        raise ValueError(f'Native SiPAI state_dict key mismatch: missing={missing}, unexpected={unexpected}.')
    initialized_keys = sorted((key for key in missing if any((key.startswith(prefix) for prefix in missing_prefixes))))
    supplemental_keys = sorted((key for key in missing if any((key.startswith(prefix) for prefix in fill_prefixes))))
    supplemental_path: Path | None = None
    if supplemental_checkpoint is not None:
        supplemental_path = Path(supplemental_checkpoint)
        if supplemental_path.is_dir():
            supplemental_path = supplemental_path / 'model.safetensors'
        if supplemental_path.suffix != '.safetensors':
            raise ValueError(f'Native SiPAI supplemental checkpoint must be a .safetensors file or directory, got {supplemental_path}.')
        if not supplemental_path.is_file():
            raise FileNotFoundError(f'Native SiPAI supplemental Safetensors checkpoint does not exist: {supplemental_path}')
        supplemental_source = load_file(str(supplemental_path), device=str(torch.device(device)))
        selected_supplemental = {key: tensor for (key, tensor) in supplemental_source.items() if any((key.startswith(prefix) for prefix in fill_prefixes))}
        supplemental_missing = sorted(set(supplemental_keys) - set(selected_supplemental))
        supplemental_unexpected = sorted(set(selected_supplemental) - set(supplemental_keys))
        if supplemental_missing or supplemental_unexpected:
            raise ValueError(f'Native SiPAI supplemental state_dict key mismatch: missing={supplemental_missing}, unexpected={supplemental_unexpected}.')
        source.update(selected_supplemental)
    has_meta = any((tensor.device.type == 'meta' for tensor in target_state.values()))
    if initialized_keys and has_meta:
        raise ValueError(f'Native SiPAI cannot preserve model initialization for missing meta parameters: missing={initialized_keys}.')
    prepared: dict[str, torch.Tensor] = {}
    for (key, tensor) in source.items():
        expected = target_state[key]
        if tensor.shape != expected.shape:
            raise ValueError(f'Native SiPAI shape mismatch for {key!r}: source={tuple(tensor.shape)}, target={tuple(expected.shape)}.')
        destination = torch.device(device) if expected.device.type == 'meta' else expected.device
        prepared[key] = tensor.to(device=destination, dtype=expected.dtype).contiguous()
    result = target.load_state_dict(prepared, strict=not initialized_keys, assign=has_meta)
    if sorted(result.missing_keys) != initialized_keys or result.unexpected_keys:
        raise RuntimeError(f'Native SiPAI load result does not match the validated compatibility contract: missing={result.missing_keys}, unexpected={result.unexpected_keys}.')
    return SimateBetaLoadReport(path=checkpoint, config_path=None, source_format=SIPAI_NATIVE_ABI, tensor_count=len(prepared), parameter_count=sum((tensor.numel() for tensor in prepared.values())), assigned=has_meta, loader_type='SipaiNativeSafetensors', initialized_keys=tuple(initialized_keys), supplemental_path=supplemental_path, supplemental_keys=tuple(supplemental_keys))

def _validate_prefixes(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f'{name} must be a sequence of non-empty prefixes.')
    prefixes = tuple((str(prefix).strip() for prefix in value))
    if any((not prefix for prefix in prefixes)) or len(set(prefixes)) != len(prefixes):
        raise ValueError(f'{name} must contain unique non-empty prefixes.')
    return prefixes

def _validate_target(target: ComposableVLAFramework) -> None:
    if not isinstance(target, ComposableVLAFramework):
        raise TypeError('simate-beta target must be a ComposableVLAFramework.')
    if not isinstance(target.backbone, SimateBetaPrefixBackbone):
        raise ValueError('simate-beta checkpoint ABI requires backbone=SimateBetaPrefixBackbone.')
    if not isinstance(target.action_head, SimateBetaFlowMatchingActionHead):
        raise ValueError('simate-beta checkpoint ABI requires action_head=SimateBetaFlowMatchingActionHead.')
__all__ = ['SIPAI_NATIVE_ABI', 'SimateBetaLoadReport', 'load_sipai_native_safetensors']
