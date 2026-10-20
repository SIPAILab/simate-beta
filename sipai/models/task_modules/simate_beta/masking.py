from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class BlockAttentionLayout:
    input_mask: torch.Tensor
    block_starts: torch.Tensor
    attention_mask: torch.Tensor
    positions: torch.Tensor

def build_block_causal_attention(input_mask: torch.Tensor, block_starts: torch.Tensor) -> BlockAttentionLayout:
    if input_mask.ndim != 2 or block_starts.shape != (input_mask.shape[1],):
        raise ValueError('Expected input_mask [B,T] and block_starts [T].')
    if input_mask.dtype != torch.bool or block_starts.dtype != torch.bool:
        raise TypeError('input_mask and block_starts must use bool dtype.')
    expanded_block_starts = block_starts.expand(input_mask.shape[0], -1)
    cumulative_blocks = torch.cumsum(expanded_block_starts.int(), dim=1)
    block_attention = cumulative_blocks[:, None, :] <= cumulative_blocks[:, :, None]
    valid_attention = input_mask[:, None, :] * input_mask[:, :, None]
    attention_mask = torch.logical_and(block_attention, valid_attention)
    positions = torch.cumsum(input_mask.int(), dim=1) - 1
    return BlockAttentionLayout(input_mask=input_mask, block_starts=block_starts, attention_mask=attention_mask, positions=positions)
__all__ = ['BlockAttentionLayout', 'build_block_causal_attention']
