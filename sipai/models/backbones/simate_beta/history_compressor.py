from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from sipai.models.base_modules.gemma import resolve_torch_dtype

def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1

class TokenAutoencoderResidualBlock(nn.Module):

    def __init__(self, channels: int, *, dtype: torch.dtype=torch.float32, device: torch.device | str | None=None) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels, dtype=dtype, device=device)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dtype=dtype, device=device)
        self.norm2 = nn.GroupNorm(groups, channels, dtype=dtype, device=device)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, dtype=dtype, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden

class SimateBetaVisualTokenEncoder(nn.Module):

    def __init__(self, *, token_dim: int=2048, hidden_dim: int=512, input_grid_size: int=16, latent_grid_size: int=4, dtype: str | torch.dtype='bfloat16', device: torch.device | str | None=None) -> None:
        super().__init__()
        if min(token_dim, hidden_dim, input_grid_size, latent_grid_size) <= 0:
            raise ValueError('Token-autoencoder dimensions must be positive.')
        if input_grid_size % latent_grid_size != 0:
            raise ValueError('input_grid_size must be divisible by latent_grid_size.')
        ratio = input_grid_size // latent_grid_size
        if ratio & ratio - 1:
            raise ValueError('Token-autoencoder spatial compression ratio must be a power of two.')
        param_dtype = resolve_torch_dtype(dtype)
        self.token_dim = int(token_dim)
        self.input_grid_size = int(input_grid_size)
        self.latent_grid_size = int(latent_grid_size)
        self.input_norm = nn.LayerNorm(token_dim, dtype=param_dtype, device=device)
        self.input_projection = nn.Conv2d(token_dim, hidden_dim, 1, dtype=param_dtype, device=device)
        self.stages = nn.ModuleList()
        for _ in range(int(math.log2(ratio))):
            self.stages.append(nn.Sequential(TokenAutoencoderResidualBlock(hidden_dim, dtype=param_dtype, device=device), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, dtype=param_dtype, device=device)))
        self.bottleneck = TokenAutoencoderResidualBlock(hidden_dim, dtype=param_dtype, device=device)
        self.output_projection = nn.Conv2d(hidden_dim, token_dim, 1, dtype=param_dtype, device=device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.input_grid_size * self.input_grid_size
        if tokens.ndim != 3 or tokens.shape[1:] != (expected_tokens, self.token_dim):
            raise ValueError(f'SimateBetaVisualTokenEncoder expects tokens with shape [B,{expected_tokens},{self.token_dim}], got {tuple(tokens.shape)}.')
        hidden = self.input_norm(tokens)
        hidden = hidden.reshape(tokens.shape[0], self.input_grid_size, self.input_grid_size, self.token_dim).permute(0, 3, 1, 2)
        hidden = self.input_projection(hidden)
        for stage in self.stages:
            hidden = stage(hidden)
        hidden = self.output_projection(self.bottleneck(hidden))
        return hidden.permute(0, 2, 3, 1).reshape(tokens.shape[0], self.latent_grid_size * self.latent_grid_size, self.token_dim)

__all__ = ['SimateBetaVisualTokenEncoder', 'TokenAutoencoderResidualBlock']
