from __future__ import annotations
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from sipai.data.schema import CanonicalBatch
from sipai.models.task_modules.simate_beta.contracts import SimateBetaPreparedInputs
SIMATE_BETA_IMAGE_AUGMENTATION_MODES = frozenset({'openpi', 'robodojo_memory'})

@dataclass(frozen=True)
class SimateBetaColorAugmentation:
    apply: bool
    brightness: float
    contrast: float
    hue: float = 0.0

def validate_simate_beta_image_augmentation_mode(mode: str) -> str:
    normalized = str(mode)
    if normalized not in SIMATE_BETA_IMAGE_AUGMENTATION_MODES:
        raise ValueError(f'Unsupported SimateBeta image augmentation mode {normalized!r}; expected one of {sorted(SIMATE_BETA_IMAGE_AUGMENTATION_MODES)}.')
    return normalized

def validate_canonical_sections(batch: CanonicalBatch, *, require_actions: bool) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    if not isinstance(batch, Mapping):
        raise TypeError('SimateBeta canonical batch must be a mapping.')
    missing = [key for key in ('inputs', 'targets', 'masks', 'meta') if key not in batch]
    if missing:
        raise KeyError(f'CanonicalBatch missing sections: {missing}')
    (inputs, targets, masks) = (batch['inputs'], batch['targets'], batch['masks'])
    if not isinstance(inputs, Mapping) or not isinstance(targets, Mapping) or (not isinstance(masks, Mapping)):
        raise TypeError('CanonicalBatch inputs, targets, and masks must be mappings.')
    language = inputs.get('language')
    if isinstance(language, torch.Tensor) or isinstance(language, (str, bytes)) or (not isinstance(language, Sequence)):
        raise TypeError('CanonicalBatch inputs.language must be a batch sequence of instructions.')
    batch_size = len(language)
    if batch_size <= 0:
        raise ValueError('CanonicalBatch must contain at least one sample.')
    if require_actions and ('actions' not in targets or targets['actions'] is None):
        raise KeyError('CanonicalBatch targets.actions is required.')
    return (inputs, masks, batch_size)

def batch_values(value: object, batch_size: int, *, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(f'CanonicalBatch {name} must have batch dimension {batch_size}.')
        return list(value.unbind(0))
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(f'CanonicalBatch {name} must have batch dimension {batch_size}.')
        return list(value)
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        values = list(value)
        if len(values) != batch_size:
            raise ValueError(f'CanonicalBatch {name} must contain {batch_size} samples, got {len(values)}.')
        return values
    raise TypeError(f'CanonicalBatch {name} must be a batched Tensor, ndarray, or sequence.')

def fixed_sequence_values(value: object, expected_size: int, *, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise TypeError(f'CanonicalBatch {name} must contain a sequence axis.')
        values = list(value.unbind(0))
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise TypeError(f'CanonicalBatch {name} must contain a sequence axis.')
        values = list(value)
    elif isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        values = list(value)
    else:
        raise TypeError(f'CanonicalBatch {name} must be a Tensor, ndarray, or sequence.')
    if len(values) != expected_size:
        raise ValueError(f'CanonicalBatch {name} must contain {expected_size} values, got {len(values)}.')
    return values

def numeric_payload(payload: object, *, name: str) -> np.ndarray:
    if isinstance(payload, Mapping):
        if 'values' not in payload:
            raise KeyError(f"Canonical {name} payload is missing 'values'.")
        payload = payload['values']
    if isinstance(payload, torch.Tensor):
        payload = payload.detach().cpu().numpy()
    array = np.asarray(payload)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f'Canonical {name} payload must be numeric.')
    return array

def canonical_window_anchor(payload: object, *, name: str) -> object:
    if isinstance(payload, Sequence) and (not isinstance(payload, (str, bytes, np.ndarray, torch.Tensor))):
        values = list(payload)
        if not values:
            raise ValueError(f'Canonical SimateBeta {name} window must not be empty.')
        first = values[0]
        if name.startswith('camera') or isinstance(first, (Mapping, np.ndarray, torch.Tensor, Image.Image)):
            return first
        if name == 'state' and np.asarray(values).ndim >= 2:
            return first
    return payload

def canonical_instructions(value: object, batch_size: int) -> tuple[str, ...]:
    instructions: list[str] = []
    for (index, item) in enumerate(batch_values(value, batch_size, name='language')):
        if item is None:
            raise ValueError(f'CanonicalBatch language[{index}] is required for SimateBeta.')
        if isinstance(item, bytes):
            item = item.decode('utf-8')
        elif not isinstance(item, str):
            item = item.item() if hasattr(item, 'item') else str(item)
        instructions.append(item)
    return tuple(instructions)

def clean_instruction(instruction: str) -> str:
    return instruction.strip().replace('_', ' ').replace('\n', ' ')

def lookup_state_key(mapping: Mapping[str, Any], key: str) -> Any:
    for candidate in (key, key.removeprefix('state.'), key.removeprefix('state/')):
        if candidate in mapping:
            return mapping[candidate]
    current: Any = mapping
    for component in key.replace('/', '.').split('.'):
        if component == 'state' and isinstance(current, Mapping) and (component not in current):
            continue
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(f'Structured Canonical state is missing configured key {key!r}.')
        current = current[component]
    return current

def flatten_state_payload(payload: object, state_keys: tuple[str, ...] | None) -> np.ndarray:
    if isinstance(payload, Mapping) and 'values' in payload:
        return numeric_payload(payload, name='state').astype(np.float32, copy=False).reshape(-1)
    if isinstance(payload, Mapping):
        if state_keys is None:
            raise ValueError('Structured Canonical state requires explicit SimateBeta state_keys/state_order.')
        parts = [numeric_payload(lookup_state_key(payload, key), name=f'state {key!r}').astype(np.float32, copy=False).reshape(-1) for key in state_keys]
        return np.concatenate(parts)
    return numeric_payload(payload, name='state').astype(np.float32, copy=False).reshape(-1)

def resolve_state_keys(state_keys: Sequence[str] | None, state_statistics: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    if state_keys is not None:
        return tuple(state_keys)
    statistics = state_statistics or {}
    order = statistics.get('state_order')
    if order is None and isinstance(statistics.get('state'), Mapping):
        order = statistics['state'].get('state_order')
    if order is None:
        return None
    keys = tuple(order)
    if not keys or any((not isinstance(key, str) or not key for key in keys)):
        raise ValueError('SimateBeta statistics state_order must contain non-empty strings.')
    return keys

def canonical_state(value: object, batch_size: int, device: torch.device, *, state_keys: tuple[str, ...] | None) -> torch.Tensor:
    if value is None:
        raise ValueError('CanonicalBatch inputs.state is required when discrete_state_input=true.')
    if isinstance(value, torch.Tensor):
        if value.ndim != 2 or value.shape[0] != batch_size:
            raise ValueError('CanonicalBatch tensor state must have shape [B,D].')
        return value.to(device=device, dtype=torch.float32)
    rows = [flatten_state_payload(canonical_window_anchor(item, name='state'), state_keys) for item in batch_values(value, batch_size, name='state')]
    dimensions = {row.shape for row in rows}
    if len(dimensions) != 1 or not rows or rows[0].ndim != 1 or (rows[0].size == 0):
        raise ValueError('CanonicalBatch state rows must share one non-empty flat dimension.')
    return torch.from_numpy(np.stack(rows).astype(np.float32, copy=False)).to(device)

def state_quantiles(state_statistics: Mapping[str, Any] | None, state_keys: tuple[str, ...] | None, state_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if state_statistics is None:
        raise ValueError('SimateBeta discrete state input requires q01/q99 state statistics.')
    statistics = state_statistics.get('statistics', state_statistics)
    state_stats = statistics.get('state', statistics)
    if isinstance(state_stats, Mapping) and 'q01' in state_stats and ('q99' in state_stats):
        q01 = np.asarray(state_stats['q01'], dtype=np.float32).reshape(-1)
        q99 = np.asarray(state_stats['q99'], dtype=np.float32).reshape(-1)
    elif isinstance(state_stats, Mapping) and state_keys is not None:
        q01_parts: list[np.ndarray] = []
        q99_parts: list[np.ndarray] = []
        for key in state_keys:
            item = lookup_state_key(state_stats, key)
            if not isinstance(item, Mapping) or 'q01' not in item or 'q99' not in item:
                raise ValueError(f'SimateBeta state statistics for {key!r} require q01 and q99.')
            q01_parts.append(np.asarray(item['q01'], dtype=np.float32).reshape(-1))
            q99_parts.append(np.asarray(item['q99'], dtype=np.float32).reshape(-1))
        q01 = np.concatenate(q01_parts)
        q99 = np.concatenate(q99_parts)
    else:
        raise ValueError('SimateBeta state statistics require q01 and q99 values.')
    if q01.shape != (state_dim,) or q99.shape != (state_dim,):
        raise ValueError(f'SimateBeta state statistics dimension must be {state_dim}, got q01={q01.shape}, q99={q99.shape}.')
    if not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError('SimateBeta state q01/q99 statistics must contain only finite values.')
    if np.any(q99 < q01):
        raise ValueError('SimateBeta state q99 statistics must be greater than or equal to q01.')
    return (torch.from_numpy(q01).to(device), torch.from_numpy(q99).to(device))

def tokenize_prompts(prompts: tuple[str, ...], *, tokenizer: Any, max_token_len: int, discrete_state_input: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if tokenizer is None:
        raise ValueError('SimateBeta canonical preprocessing requires a SentencePiece tokenizer.')
    rows: list[list[int]] = []
    masks: list[list[bool]] = []
    for prompt in prompts:
        tokens = list(tokenizer.encode(prompt, add_bos=True))
        if not discrete_state_input:
            tokens += list(tokenizer.encode('\n'))
        if len(tokens) > max_token_len:
            warnings.warn(f'SimateBeta token length {len(tokens)} exceeds max_token_len={max_token_len}; truncating.', stacklevel=2)
        tokens = [int(token) for token in tokens[:max_token_len]]
        valid = len(tokens)
        rows.append(tokens + [0] * (max_token_len - valid))
        masks.append([True] * valid + [False] * (max_token_len - valid))
    return (torch.tensor(rows, dtype=torch.int64, device=device), torch.tensor(masks, dtype=torch.bool, device=device))

def prepare_canonical_prefix(batch: CanonicalBatch, *, input_cameras: Sequence[str], synthetic_masked_cameras: Sequence[str], discrete_state_input: bool, state_is_normalized: bool=False, state_tokenization_domain: str | None=None, max_token_len: int, tokenizer: Any, state_statistics: Mapping[str, Any] | None, state_keys: Sequence[str] | None, device: torch.device | str, train: bool=False, generator: torch.Generator | None=None, history_camera: str | None=None, history_frames: int=0, image_augmentation_mode: str='openpi') -> SimateBetaPreparedInputs:
    if not isinstance(train, bool):
        raise TypeError('SimateBeta canonical image train flag must be bool.')
    if generator is not None and (not isinstance(generator, torch.Generator)):
        raise TypeError('SimateBeta canonical image generator must be torch.Generator or None.')
    image_augmentation_mode = validate_simate_beta_image_augmentation_mode(image_augmentation_mode)
    if history_camera is None and history_frames != 0:
        raise ValueError('SimateBeta history_frames must be zero when history_camera is disabled.')
    if history_camera is not None and history_frames <= 0:
        raise ValueError('SimateBeta history_frames must be positive when history_camera is configured.')
    (inputs, masks, batch_size) = validate_canonical_sections(batch, require_actions=False)
    target_device = torch.device(device)
    instructions = canonical_instructions(inputs.get('language'), batch_size)
    if train and image_augmentation_mode == 'robodojo_memory':
        metadata = (batch.get('meta') or {}).get('meta_data') or [None] * batch_size
        pair_augmentations: dict[str, SimateBetaColorAugmentation] = {}
        sampled_augmentations: list[SimateBetaColorAugmentation] = []
        for batch_index in range(batch_size):
            row = metadata[batch_index] if batch_index < len(metadata) else None
            raw_pair_id = row.get('counterfactual_pair_id') if isinstance(row, Mapping) else None
            pair_id = '' if raw_pair_id is None else str(raw_pair_id)
            if pair_id:
                augmentation = pair_augmentations.get(pair_id)
                if augmentation is None:
                    augmentation = sample_simate_beta_memory_color_augmentation(generator=generator)
                    pair_augmentations[pair_id] = augmentation
            else:
                augmentation = sample_simate_beta_memory_color_augmentation(generator=generator)
            sampled_augmentations.append(augmentation)
        shared_color_augmentations: tuple[SimateBetaColorAugmentation | None, ...] = tuple(sampled_augmentations)
    else:
        shared_color_augmentations = (None,) * batch_size
    canonical_images = inputs.get('images')
    if not isinstance(canonical_images, Mapping):
        raise TypeError('CanonicalBatch inputs.images must be a camera mapping.')
    canonical_image_masks = masks.get('images', {})
    if not isinstance(canonical_image_masks, Mapping):
        raise TypeError('CanonicalBatch masks.images must be a camera mapping when present.')
    synthetic_cameras = set(synthetic_masked_cameras)
    images: dict[str, torch.Tensor] = {}
    image_masks: dict[str, torch.Tensor] = {}
    for camera in input_cameras:
        if camera not in canonical_images:
            if camera not in synthetic_cameras:
                raise KeyError(f'CanonicalBatch is missing required SimateBeta camera {camera!r}.')
            images[camera] = torch.full((batch_size, 224, 224, 3), -1.0, device=target_device)
            image_masks[camera] = torch.zeros(batch_size, dtype=torch.bool, device=target_device)
            continue
        if camera in canonical_image_masks:
            mask_values = batch_values(canonical_image_masks[camera], batch_size, name=f'camera mask {camera!r}')
            valid = [bool(value.item()) if hasattr(value, 'item') else bool(value) for value in mask_values]
        else:
            valid = [True] * batch_size
        payloads = batch_values(canonical_images[camera], batch_size, name=f'camera {camera!r}')
        prepared = [prepare_simate_beta_image(canonical_window_anchor(payload, name=f'camera {camera!r}'), camera_name=camera, train=train, generator=generator, image_augmentation_mode=image_augmentation_mode, color_augmentation=shared_color_augmentations[batch_index]) if is_valid else np.full((224, 224, 3), -1.0, dtype=np.float32) for (batch_index, (payload, is_valid)) in enumerate(zip(payloads, valid, strict=True))]
        images[camera] = torch.from_numpy(np.stack(prepared)).to(device=target_device, dtype=torch.float32)
        image_masks[camera] = torch.tensor(valid, dtype=torch.bool, device=target_device)
    history_images: dict[str, torch.Tensor] = {}
    history_image_masks: dict[str, torch.Tensor] = {}
    if history_camera is not None:
        canonical_history_images = inputs.get('history_images')
        if not isinstance(canonical_history_images, Mapping):
            raise TypeError('CanonicalBatch inputs.history_images must be a camera mapping.')
        if history_camera not in canonical_history_images:
            raise KeyError(f'CanonicalBatch is missing required SimateBeta history camera {history_camera!r}.')
        canonical_history_masks = masks.get('history_images', {})
        if not isinstance(canonical_history_masks, Mapping):
            raise TypeError('CanonicalBatch masks.history_images must be a camera mapping.')
        history_payload_rows = batch_values(canonical_history_images[history_camera], batch_size, name=f'history camera {history_camera!r}')
        history_mask_rows = batch_values(canonical_history_masks[history_camera], batch_size, name=f'history camera mask {history_camera!r}') if history_camera in canonical_history_masks else [None] * batch_size
        prepared_history_rows: list[np.ndarray] = []
        valid_history_rows: list[list[bool]] = []
        for (batch_index, (payload_row, mask_row)) in enumerate(zip(history_payload_rows, history_mask_rows, strict=True)):
            row_name = f'history camera {history_camera!r}[{batch_index}]'
            payload_slots = fixed_sequence_values(payload_row, history_frames, name=row_name)
            if mask_row is None:
                valid_slots = [payload is not None for payload in payload_slots]
            else:
                mask_slots = fixed_sequence_values(mask_row, history_frames, name=f'{row_name} mask')
                valid_slots = [bool(value.item()) if hasattr(value, 'item') else bool(value) for value in mask_slots]
            if any((valid and payload is None for (payload, valid) in zip(payload_slots, valid_slots, strict=True))):
                raise ValueError(f'Canonical SimateBeta history camera {history_camera!r}[{batch_index}] marks a missing frame as valid.')
            prepared_slots = [prepare_simate_beta_image(payload, camera_name=history_camera, train=train, generator=generator, image_augmentation_mode=image_augmentation_mode, color_augmentation=shared_color_augmentations[batch_index]) if valid else np.full((224, 224, 3), -1.0, dtype=np.float32) for (payload, valid) in zip(payload_slots, valid_slots, strict=True)]
            prepared_history_rows.append(np.stack(prepared_slots))
            valid_history_rows.append(valid_slots)
        history_images[history_camera] = torch.from_numpy(np.stack(prepared_history_rows)).to(device=target_device, dtype=torch.float32)
        history_image_masks[history_camera] = torch.tensor(valid_history_rows, dtype=torch.bool, device=target_device)
    resolved_state_keys = resolve_state_keys(state_keys, state_statistics)
    normalized_state: torch.Tensor | None = None
    state_bin_ids: torch.Tensor | None = None
    if state_tokenization_domain is None:
        state_tokenization_domain = 'shared_unit_interval' if state_is_normalized else 'legacy_q01_q99'
    if state_tokenization_domain not in {'legacy_q01_q99', 'shared_unit_interval'}:
        raise ValueError(f'Unsupported SimateBeta state tokenization domain: {state_tokenization_domain!r}')
    if state_is_normalized and state_tokenization_domain != 'shared_unit_interval':
        raise ValueError('state_is_normalized conflicts with state_tokenization_domain')
    if discrete_state_input:
        raw_state = canonical_state(inputs.get('state'), batch_size, target_device, state_keys=resolved_state_keys)
        if state_tokenization_domain == 'shared_unit_interval':
            normalized_state = raw_state
        else:
            (q01, q99) = state_quantiles(state_statistics, resolved_state_keys, raw_state.shape[-1], target_device)
            normalized_state = (raw_state - q01) / (q99 - q01 + 1e-06) * 2 - 1
        bin_edges = np.linspace(-1, 1, 257, dtype=np.float64)[:-1]
        bins = np.digitize(normalized_state.detach().cpu().numpy(), bins=bin_edges) - 1
        state_bin_ids = torch.from_numpy(bins.astype(np.int64, copy=False)).to(target_device)
        prompts = tuple((f"Task: {clean_instruction(instruction)}, State: {' '.join((str(int(value)) for value in state_ids.tolist()))};\nAction: " for (instruction, state_ids) in zip(instructions, state_bin_ids, strict=True)))
    else:
        prompts = tuple((clean_instruction(instruction) for instruction in instructions))
    (token_ids, token_mask) = tokenize_prompts(prompts, tokenizer=tokenizer, max_token_len=max_token_len, discrete_state_input=discrete_state_input, device=target_device)
    return SimateBetaPreparedInputs(images=images, image_masks=image_masks, history_images=history_images, history_image_masks=history_image_masks, normalized_state=normalized_state, state_bin_ids=state_bin_ids, prompts=prompts, token_ids=token_ids, token_mask=token_mask)

def prepare_simate_beta_image(payload: object, *, camera_name: str='', train: bool=False, generator: torch.Generator | None=None, image_augmentation_mode: str='openpi', color_augmentation: SimateBetaColorAugmentation | None=None) -> np.ndarray:
    image_augmentation_mode = validate_simate_beta_image_augmentation_mode(image_augmentation_mode)
    if isinstance(payload, Mapping):
        if 'array' not in payload:
            raise KeyError("Canonical image payload is missing 'array'.")
        payload = payload['array']
    if isinstance(payload, Image.Image):
        array = np.asarray(payload)
    elif isinstance(payload, torch.Tensor):
        array = payload.detach().cpu().numpy()
    else:
        array = np.asarray(payload)
    if array.ndim != 3:
        raise ValueError(f'Canonical SimateBeta image must have rank 3, got shape {array.shape}.')
    if array.shape[-1] != 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f'Canonical SimateBeta image must have three channels, got shape {array.shape}.')
    if array.dtype == np.uint8:
        image = torch.from_numpy(np.array(array, copy=True, order='C')).float() / 255.0 * 2.0 - 1.0
    elif np.issubdtype(array.dtype, np.floating):
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all() or np.any((array < -1.0) | (array > 1.0)):
            raise ValueError('Canonical SimateBeta floating image values must be finite and in [-1, 1].')
        image = torch.from_numpy(np.array(array, copy=True, order='C'))
    else:
        raise TypeError(f'Canonical SimateBeta image must use uint8 or floating dtype, got {array.dtype}.')
    if train and image_augmentation_mode == 'robodojo_memory':
        if color_augmentation is None:
            color_augmentation = sample_simate_beta_memory_color_augmentation(generator=generator)
        image = apply_simate_beta_color_augmentation(image, color_augmentation)
    if image.shape[:2] != (224, 224):
        image = resize_with_pad_torch(image, 224, 224)
    if train and image_augmentation_mode == 'openpi':
        image = augment_simate_beta_image(image, camera_name=camera_name, generator=generator)
    return image.cpu().numpy()

def resize_with_pad_torch(images: torch.Tensor, height: int, width: int, mode: str='bilinear') -> torch.Tensor:
    squeeze_batch = images.dim() == 3
    if squeeze_batch:
        images = images.unsqueeze(0)
    images = images.permute(0, 3, 1, 2)
    (_, _, current_height, current_width) = images.shape
    ratio = max(current_width / width, current_height / height)
    (resized_height, resized_width) = (int(current_height / ratio), int(current_width / ratio))
    resized = F.interpolate(images, size=(resized_height, resized_width), mode=mode, align_corners=False)
    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        fill_value = 0
    else:
        resized = resized.clamp(-1.0, 1.0)
        fill_value = -1.0
    (pad_h, extra_h) = divmod(height - resized_height, 2)
    (pad_w, extra_w) = divmod(width - resized_width, 2)
    padded = F.pad(resized, (pad_w, pad_w + extra_w, pad_h, pad_h + extra_h), value=fill_value)
    padded = padded.permute(0, 2, 3, 1)
    return padded.squeeze(0) if squeeze_batch else padded

def augment_simate_beta_image(image: torch.Tensor, *, camera_name: str, generator: torch.Generator | None) -> torch.Tensor:
    augmented = image / 2.0 + 0.5
    if 'wrist' not in camera_name:
        (height, width) = augmented.shape[:2]
        crop_size = (int(height * 0.95), int(width * 0.95))
        offsets = simate_beta_uniform_tensor((-float(height - crop_size[0]) / 2.0, -float(width - crop_size[1]) / 2.0), (float(height - crop_size[0]) / 2.0, float(width - crop_size[1]) / 2.0), generator=generator)
        augmented = augmax_geometric_transform(augmented, crop_size=crop_size, crop_offsets=(float(offsets[0]), float(offsets[1])), angle_degrees=simate_beta_uniform(-5.0, 5.0, generator=generator))
    amounts = simate_beta_uniform_tensor((-0.3, -0.4, -0.1, -0.5), (0.3, 0.4, 0.1, 0.5), generator=generator)
    return augmax_color_jitter(augmented, brightness=float(amounts[0]), contrast=float(amounts[1]), hue=float(amounts[2]), apply=bool(torch.rand((), generator=generator).item() < 0.5)) * 2.0 - 1.0

def sample_simate_beta_memory_color_augmentation(*, generator: torch.Generator | None) -> SimateBetaColorAugmentation:
    amounts = simate_beta_uniform_tensor((-0.15, -0.2), (0.15, 0.2), generator=generator)
    return SimateBetaColorAugmentation(apply=bool(torch.rand((), generator=generator).item() < 0.5), brightness=float(amounts[0]), contrast=float(amounts[1]), hue=0.0)

def apply_simate_beta_color_augmentation(image: torch.Tensor, augmentation: SimateBetaColorAugmentation) -> torch.Tensor:
    return augmax_color_jitter(image / 2.0 + 0.5, brightness=augmentation.brightness, contrast=augmentation.contrast, hue=augmentation.hue, apply=augmentation.apply) * 2.0 - 1.0

def augmax_geometric_transform(image: torch.Tensor, *, crop_size: tuple[int, int], crop_offsets: tuple[float, float], angle_degrees: float) -> torch.Tensor:
    (height, width) = image.shape[:2]
    (crop_height, crop_width) = crop_size
    (y_coords, x_coords) = torch.meshgrid(torch.arange(height, dtype=torch.float32, device=image.device) - (height / 2.0 - 0.5), torch.arange(width, dtype=torch.float32, device=image.device) - (width / 2.0 - 0.5), indexing='ij')
    theta = math.radians(angle_degrees)
    source_y = crop_height / height * (math.cos(theta) * y_coords + math.sin(theta) * x_coords) + crop_offsets[0]
    source_x = crop_width / width * (-math.sin(theta) * y_coords + math.cos(theta) * x_coords) + crop_offsets[1]
    source_y += height / 2.0 - 0.5
    source_x += width / 2.0 - 0.5
    grid = torch.stack([source_x * (2.0 / (width - 1)) - 1.0, source_y * (2.0 / (height - 1)) - 1.0], dim=-1).unsqueeze(0)
    transformed = F.grid_sample(image.permute(2, 0, 1).unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return transformed.squeeze(0).permute(1, 2, 0)

def augmax_color_jitter(image: torch.Tensor, *, brightness: float, contrast: float, hue: float, apply: bool) -> torch.Tensor:
    if not apply:
        return image
    (value, argmax) = torch.max(image, dim=-1)
    minimum = torch.min(image, dim=-1).values
    channel_range = value - minimum
    (red, green, blue) = image.unbind(dim=-1)
    numerator = torch.stack((green - blue, blue - red, red - green), dim=-1)
    safe_range = torch.where(channel_range == 0.0, torch.ones_like(channel_range), channel_range)
    hsv_hue = (2.0 * argmax.float() + torch.gather(numerator, -1, argmax.unsqueeze(-1)).squeeze(-1) / safe_range) / 6.0
    hsv_hue = torch.where(channel_range == 0.0, torch.zeros_like(hsv_hue), hsv_hue)
    safe_value = torch.where(value == 0.0, torch.ones_like(value), value)
    saturation = torch.where(value == 0.0, torch.zeros_like(value), channel_range / safe_value)
    value = value * (1.0 + brightness) if brightness < 0.0 else value * (1.0 - brightness) + brightness
    slant = math.tan((contrast + 1.0) * (math.pi / 4.0))
    (lower, upper) = (slant / (2.0 * (1.0 + slant)), 1.0 - slant / (2.0 * (1.0 + slant)))
    value = torch.where(value < lower, value / slant, torch.where(value > upper, value / slant + 1.0 - 1.0 / slant, slant * (value - 0.5) + 0.5))
    coefficients = torch.tensor((5.0, 3.0, 1.0), dtype=image.dtype, device=image.device)
    k = torch.remainder(coefficients + (hsv_hue + hue).unsqueeze(-1) * 6.0, 6.0)
    color_shape = torch.clamp(torch.minimum(k, 4.0 - k), min=0.0, max=1.0)
    return value.unsqueeze(-1) - value.unsqueeze(-1) * saturation.unsqueeze(-1) * color_shape

def simate_beta_uniform(low: float, high: float, *, generator: torch.Generator | None) -> float:
    return low + (high - low) * torch.rand(1, generator=generator).item()

def simate_beta_uniform_tensor(low: Sequence[float], high: Sequence[float], *, generator: torch.Generator | None) -> torch.Tensor:
    low_tensor = torch.tensor(tuple(low), dtype=torch.float32)
    high_tensor = torch.tensor(tuple(high), dtype=torch.float32)
    return low_tensor + (high_tensor - low_tensor) * torch.rand(low_tensor.shape, generator=generator)
__all__ = ['SIMATE_BETA_IMAGE_AUGMENTATION_MODES', 'SimateBetaColorAugmentation', 'apply_simate_beta_color_augmentation', 'augmax_color_jitter', 'augmax_geometric_transform', 'augment_simate_beta_image', 'batch_values', 'canonical_instructions', 'canonical_state', 'canonical_window_anchor', 'clean_instruction', 'flatten_state_payload', 'lookup_state_key', 'numeric_payload', 'prepare_canonical_prefix', 'prepare_simate_beta_image', 'resize_with_pad_torch', 'resolve_state_keys', 'sample_simate_beta_memory_color_augmentation', 'state_quantiles', 'tokenize_prompts', 'validate_canonical_sections', 'validate_simate_beta_image_augmentation_mode']
