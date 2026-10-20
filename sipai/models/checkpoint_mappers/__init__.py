from collections.abc import Mapping
from typing import Any
import torch
from sipai.registry import CHECKPOINT_LOADERS
from .simate_beta_safetensors import SIPAI_NATIVE_ABI, SimateBetaLoadReport, load_sipai_native_safetensors

def load_init_checkpoint(target: torch.nn.Module, config: Mapping[str, Any]) -> SimateBetaLoadReport:
    return CHECKPOINT_LOADERS.build(config, target=target)
