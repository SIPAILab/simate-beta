from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sipai.data.schema import CanonicalBatch
from sipai.models.base_modules.gemma import GemmaConfig, GemmaDecoderLayer, GemmaRMSNorm, LayerKVCache, resolve_torch_dtype
from sipai.models.base_modules.siglip import SigLIPVisionEncoder
from sipai.models.task_modules.simate_beta import SIMATE_BETA_VISION_VARIANT, SIMATE_BETA_VLM_VARIANT, SIMATE_BETA_VOCAB_SIZE, SimateBetaPreparedInputs, SimatePrefixContext, TokenBlock, prepare_canonical_prefix, validate_simate_beta_image_augmentation_mode
from sipai.models.task_modules.simate_beta.masking import build_block_causal_attention
from sipai.registry import BACKBONES

@BACKBONES.register_module()
class SimateBetaPrefixBackbone(nn.Module):

    def __init__(self, *, vision_variant: str=SIMATE_BETA_VISION_VARIANT, transformer_variant: str=SIMATE_BETA_VLM_VARIANT, input_cameras: Sequence[str]=('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'), synthetic_masked_cameras: Sequence[str]=(), history_camera: str | None=None, history_frames: int=0, history_pool_size: int=4, history_compression: str='average_pool', history_compressor_hidden_dim: int=512, image_augmentation_mode: str='openpi', discrete_state_input: bool=True, state_is_normalized: bool=False, state_tokenization_domain: str | None=None, max_token_len: int=200, dtype: str | torch.dtype='bfloat16', vision_encoder: nn.Module | None=None, token_embedding: nn.Module | None=None, init_device: torch.device | str | None=None, tokenizer_model_path: str | Path | None=None, tokenizer: Any | None=None, state_statistics: Mapping[str, Any] | None=None, state_keys: Sequence[str] | None=None, gradient_checkpointing: bool=False, layers: Sequence[nn.Module] | None=None, final_norms: Sequence[nn.Module] | None=None, gemma_config: GemmaConfig | None=None) -> None:
        super().__init__()
        if transformer_variant != SIMATE_BETA_VLM_VARIANT:
            raise ValueError(f'SimateBetaPrefixBackbone only supports transformer_variant={SIMATE_BETA_VLM_VARIANT!r}.')
        if vision_variant != SIMATE_BETA_VISION_VARIANT:
            raise ValueError(f'SimateBetaPrefixBackbone only supports vision_variant={SIMATE_BETA_VISION_VARIANT!r}.')
        self.config = gemma_config or GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
        self.param_dtype = resolve_torch_dtype(dtype)
        self.compute_dtype = self.param_dtype
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.vision_variant = vision_variant
        self.transformer_variant = transformer_variant
        self.input_cameras = tuple(input_cameras)
        self.synthetic_masked_cameras = tuple(synthetic_masked_cameras)
        self.history_camera = str(history_camera) if history_camera is not None else None
        self.history_frames = int(history_frames)
        self.history_pool_size = int(history_pool_size)
        self.history_compression = str(history_compression)
        if self.history_compression not in {'average_pool', 'ae'}:
            raise ValueError("SimateBeta history_compression must be 'average_pool' or 'ae'.")
        if self.history_compression == 'ae' and self.history_camera is None:
            raise ValueError('SimateBeta AE history compression requires history_camera.')
        self.image_augmentation_mode = validate_simate_beta_image_augmentation_mode(image_augmentation_mode)
        self.discrete_state_input = bool(discrete_state_input)
        self.state_is_normalized = bool(state_is_normalized)
        self.state_tokenization_domain = state_tokenization_domain
        self.max_token_len = int(max_token_len)
        if not self.input_cameras or len(set(self.input_cameras)) != len(self.input_cameras):
            raise ValueError('SimateBetaPrefixBackbone input_cameras must be non-empty and unique.')
        if not set(self.synthetic_masked_cameras).issubset(self.input_cameras):
            raise ValueError('SimateBetaPrefixBackbone synthetic cameras must also appear in input_cameras.')
        if self.history_camera is None and self.history_frames != 0:
            raise ValueError('SimateBetaPrefixBackbone history_frames must be zero when history_camera is disabled.')
        if self.history_camera is not None:
            if self.history_camera not in self.input_cameras:
                raise ValueError('SimateBetaPrefixBackbone history_camera must also appear in input_cameras.')
            if self.history_frames <= 0:
                raise ValueError('SimateBetaPrefixBackbone history_frames must be positive when history_camera is configured.')
            if self.history_pool_size <= 0:
                raise ValueError('SimateBetaPrefixBackbone history_pool_size must be positive when history_camera is configured.')
        if self.max_token_len <= 0:
            raise ValueError('SimateBetaPrefixBackbone max_token_len must be positive.')
        self.vision_encoder = vision_encoder or SigLIPVisionEncoder(dtype=self.param_dtype, device=init_device)
        self.history_compressor: nn.Module | None = None
        if self.history_compression == 'ae':
            from .history_compressor import SimateBetaVisualTokenEncoder
            vision_config = self.vision_encoder.config
            self.history_compressor = SimateBetaVisualTokenEncoder(token_dim=self.config.width, hidden_dim=history_compressor_hidden_dim, input_grid_size=math.isqrt(vision_config.num_patches), latent_grid_size=self.history_pool_size, dtype=self.param_dtype, device=init_device)
        self.token_embedding = token_embedding or nn.Embedding(SIMATE_BETA_VOCAB_SIZE, self.config.width, dtype=self.param_dtype, device=init_device)
        if token_embedding is None:
            nn.init.normal_(self.token_embedding.weight)
        self.layers = nn.ModuleList(tuple(layers) if layers is not None else (GemmaDecoderLayer(self.config, dtype=self.param_dtype, device=init_device) for _ in range(self.config.depth)))
        self.final_norms = nn.ModuleList(tuple(final_norms) if final_norms is not None else (GemmaRMSNorm(self.config.width, dtype=self.param_dtype, device=init_device),))
        self._tokenizer: Any | None = None
        self._state_statistics: Mapping[str, Any] | None = None
        self._state_keys: tuple[str, ...] | None = None
        self.configure_preprocessing(tokenizer_model_path=tokenizer_model_path, tokenizer=tokenizer, state_statistics=state_statistics, state_keys=state_keys)

    def configure_preprocessing(self, *, tokenizer_model_path: str | Path | None=None, tokenizer: Any | None=None, state_statistics: Mapping[str, Any] | None=None, state_keys: Sequence[str] | None=None) -> None:
        if tokenizer is not None and tokenizer_model_path is not None:
            raise ValueError('Provide either tokenizer or tokenizer_model_path, not both.')
        if tokenizer_model_path is not None:
            path = Path(tokenizer_model_path)
            if not path.is_file():
                raise FileNotFoundError(f'simate-beta tokenizer model does not exist: {path}')
            try:
                import sentencepiece
                tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))
            except Exception as error:
                raise ValueError(f'Failed to load simate-beta SentencePiece model {path}: {error}') from error
        if tokenizer is not None:
            self._tokenizer = tokenizer
        if state_statistics is not None:
            self._state_statistics = state_statistics
        if state_keys is not None:
            self._state_keys = tuple(state_keys)

    def set_compute_dtype(self, dtype: str | torch.dtype) -> None:
        self.compute_dtype = resolve_torch_dtype(dtype)

    def model_device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else torch.device('cpu')

    def encode(self, batch: CanonicalBatch, *, mode: str | None=None, generator: torch.Generator | None=None, **kwargs: Any) -> SimatePrefixContext:
        del kwargs
        prepared = self.prepare_canonical_prefix(batch, device=self.model_device(), train=self.training and mode != 'predict', generator=generator)
        return self.build_prefix_context(self.embed_prepared_prefix(prepared))

    def prepare_canonical_prefix(self, batch: CanonicalBatch, *, device: torch.device | str, train: bool=False, generator: torch.Generator | None=None) -> SimateBetaPreparedInputs:
        return prepare_canonical_prefix(batch, input_cameras=self.input_cameras, synthetic_masked_cameras=self.synthetic_masked_cameras, discrete_state_input=self.discrete_state_input, state_is_normalized=self.state_is_normalized, state_tokenization_domain=self.state_tokenization_domain, max_token_len=self.max_token_len, tokenizer=self._tokenizer, state_statistics=self._state_statistics, state_keys=self._state_keys, device=device, train=train, generator=generator, history_camera=self.history_camera, history_frames=self.history_frames, image_augmentation_mode=self.image_augmentation_mode)

    def embed_prepared_prefix(self, prepared: SimateBetaPreparedInputs) -> TokenBlock:
        return self.embed_prefix(images=prepared.images, image_masks=prepared.image_masks, history_images=prepared.history_images, history_image_masks=prepared.history_image_masks, token_ids=prepared.token_ids, token_mask=prepared.token_mask)

    def embed_prefix(self, *, images: Mapping[str, torch.Tensor], image_masks: Mapping[str, torch.Tensor], history_images: Mapping[str, torch.Tensor] | None=None, history_image_masks: Mapping[str, torch.Tensor] | None=None, token_ids: torch.Tensor | None=None, token_mask: torch.Tensor | None=None) -> TokenBlock:
        token_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        batch_size: int | None = None
        for camera in self.input_cameras:
            image = images[camera]
            image_mask = image_masks[camera]
            batch_size = image.shape[0] if batch_size is None else batch_size
            encoded = self.vision_encoder(image)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            token_parts.append(encoded.to(self.compute_dtype))
            mask_parts.append(image_mask[:, None].expand(-1, encoded.shape[1]))
        if self.history_camera is not None:
            if history_images is None or self.history_camera not in history_images:
                raise KeyError(f'Missing SimateBeta history images for camera {self.history_camera!r}.')
            if history_image_masks is None or self.history_camera not in history_image_masks:
                raise KeyError(f'Missing SimateBeta history mask for camera {self.history_camera!r}.')
            history = history_images[self.history_camera]
            history_mask = history_image_masks[self.history_camera]
            if history.ndim != 5:
                raise ValueError('SimateBeta history images must have shape [B,T,H,W,C].')
            if history.shape[:2] != (batch_size, self.history_frames):
                raise ValueError(f'SimateBeta history images must start with shape [{batch_size},{self.history_frames}], got {tuple(history.shape[:2])}.')
            if history_mask.shape != (batch_size, self.history_frames):
                raise ValueError(f'SimateBeta history mask must have shape [{batch_size},{self.history_frames}], got {tuple(history_mask.shape)}.')
            if history_mask.dtype != torch.bool:
                raise TypeError('SimateBeta history mask must use bool dtype.')
            flat_history = history.reshape(batch_size * self.history_frames, *history.shape[2:])
            flat_history_mask = history_mask.reshape(-1)
            valid_indices = torch.nonzero(flat_history_mask, as_tuple=False).squeeze(-1)
            reference_tokens = token_parts[0]
            if valid_indices.numel() > 0:
                encoded_history = self.vision_encoder(flat_history.index_select(0, valid_indices))
                if isinstance(encoded_history, tuple):
                    encoded_history = encoded_history[0]
                if encoded_history.shape[1:] != reference_tokens.shape[1:]:
                    raise ValueError('SimateBeta current and history images must produce the same vision token shape.')
                encoded_history = self._compress_history_tokens(encoded_history)
                scattered_history = encoded_history.new_zeros(batch_size * self.history_frames, encoded_history.shape[1], encoded_history.shape[2]).index_copy(0, valid_indices, encoded_history)
            else:
                scattered_history = reference_tokens.new_zeros(batch_size * self.history_frames, self.history_pool_size * self.history_pool_size, reference_tokens.shape[2])
            history_token_count = scattered_history.shape[1]
            history_tokens = scattered_history.reshape(batch_size, self.history_frames * history_token_count, scattered_history.shape[-1])
            expanded_history_mask = history_mask[:, :, None].expand(-1, -1, history_token_count).reshape(batch_size, self.history_frames * history_token_count)
            token_parts.insert(0, history_tokens.to(self.compute_dtype))
            mask_parts.insert(0, expanded_history_mask)
        if token_ids is not None:
            if token_mask is None:
                raise ValueError('token_mask is required with token_ids.')
            prompt_tokens = self.token_embedding(token_ids) * math.sqrt(self.config.width)
            token_parts.append(prompt_tokens.to(self.compute_dtype))
            mask_parts.append(token_mask)
        if not token_parts or batch_size is None:
            raise ValueError('SimateBetaPrefixBackbone requires at least one camera token block.')
        embeddings = torch.cat(token_parts, dim=1)
        valid_mask = torch.cat(mask_parts, dim=1)
        return TokenBlock(embeddings=embeddings, valid_mask=valid_mask, block_starts=torch.zeros(embeddings.shape[1], dtype=torch.bool, device=embeddings.device))

    def _compress_history_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.history_compressor is None:
            return self._pool_history_tokens(tokens)
        tokens = tokens.to(self.history_compressor.input_norm.weight.dtype)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.history_compressor, tokens, use_reentrant=False)
        return self.history_compressor(tokens)

    def _pool_history_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError('SimateBeta history vision tokens must have shape [B,N,D].')
        token_count = int(tokens.shape[1])
        spatial = math.isqrt(token_count)
        if spatial * spatial != token_count:
            raise ValueError(f'SimateBeta history vision token count must form a square grid, got {token_count}.')
        if self.history_pool_size > spatial:
            raise ValueError(f'SimateBeta history_pool_size cannot exceed the encoded history grid size: pool={self.history_pool_size}, encoded={spatial}.')
        hidden = int(tokens.shape[2])
        grid = tokens.reshape(-1, spatial, spatial, hidden).permute(0, 3, 1, 2)
        grid = F.adaptive_avg_pool2d(grid, output_size=(self.history_pool_size, self.history_pool_size))
        return grid.permute(0, 2, 3, 1).reshape(-1, self.history_pool_size * self.history_pool_size, hidden)

    def build_prefix_context(self, prefix: TokenBlock) -> SimatePrefixContext:
        layout = build_block_causal_attention(prefix.valid_mask, prefix.block_starts)
        hidden_states = prefix.embeddings
        layer_kv: list[LayerKVCache] = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():

                def run_layer(hidden: torch.Tensor, positions: torch.Tensor, attention_mask: torch.Tensor, module: nn.Module=layer):
                    return module(hidden, positions, attention_mask)
                (hidden_states, cache) = checkpoint(run_layer, hidden_states, layout.positions, layout.attention_mask, use_reentrant=False)
            else:
                (hidden_states, cache) = layer(hidden_states, layout.positions, layout.attention_mask)
            layer_kv.append(cache)
        if self.final_norms:
            self.final_norms[0](hidden_states, None)
        return SimatePrefixContext(input_mask=layout.input_mask, layer_kv=tuple(layer_kv))
__all__ = ['SimateBetaPrefixBackbone']
