from __future__ import annotations
from dataclasses import dataclass
import torch
from sipai.models.base_modules.gemma import LayerKVCache
SIMATE_BETA_VOCAB_SIZE = 257152
SIMATE_BETA_VISION_VARIANT = 'siglip_so400m_patch14_224'
SIMATE_BETA_VLM_VARIANT = 'gemma_2b'
SIMATE_BETA_ACTION_VARIANT = 'gemma_300m'

@dataclass(frozen=True)
class SimateBetaPreparedInputs:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    history_images: dict[str, torch.Tensor]
    history_image_masks: dict[str, torch.Tensor]
    normalized_state: torch.Tensor | None
    state_bin_ids: torch.Tensor | None
    prompts: tuple[str, ...]
    token_ids: torch.Tensor
    token_mask: torch.Tensor

@dataclass(frozen=True)
class TokenBlock:
    embeddings: torch.Tensor
    valid_mask: torch.Tensor
    block_starts: torch.Tensor
    condition: torch.Tensor | None = None

@dataclass(frozen=True)
class SimatePrefixContext:
    input_mask: torch.Tensor
    layer_kv: tuple[LayerKVCache, ...]
__all__ = ['SIMATE_BETA_ACTION_VARIANT', 'SIMATE_BETA_VISION_VARIANT', 'SIMATE_BETA_VLM_VARIANT', 'SIMATE_BETA_VOCAB_SIZE', 'SimateBetaPreparedInputs', 'SimatePrefixContext', 'TokenBlock']
