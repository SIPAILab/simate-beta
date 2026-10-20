from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sipai.data.schema import CanonicalBatch
from sipai.models.base_modules.gemma import AdaptiveGemmaRMSNorm, GemmaConfig, GemmaDecoderLayer, LayerKVCache, resolve_torch_dtype
from sipai.models.task_modules.simate_beta import SIMATE_BETA_ACTION_VARIANT, SimatePrefixContext, TokenBlock
from sipai.models.task_modules.simate_beta.masking import build_block_causal_attention
from sipai.models.task_modules.simate_beta.processing import batch_values, numeric_payload, validate_canonical_sections
from sipai.registry import ACTION_HEADS

def _autocast_disabled(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)

def _posemb_sincos(positions: torch.Tensor, embedding_dim: int, *, min_period: float=0.004, max_period: float=4.0) -> torch.Tensor:
    if embedding_dim % 2 != 0:
        raise ValueError(f'embedding_dim ({embedding_dim}) must be divisible by 2.')
    fractions = torch.linspace(0.0, 1.0, embedding_dim // 2, device=positions.device, dtype=torch.float32)
    periods = min_period * (max_period / min_period) ** fractions
    radians = torch.einsum('i,j->ij', positions.float(), 1.0 / periods * 2 * torch.pi)
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1).to(positions.dtype)

def _validate_action_normalization(config: Mapping[str, Any] | None, *, action_dim: int) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError('SimateBeta action_normalization must be a mapping.')
    mode = str(config.get('mode', 'none')).lower()
    if mode in {'', 'none'}:
        return None
    if mode != 'q01_q99':
        raise ValueError(f'SimateBeta action_normalization only supports q01_q99, got {mode!r}.')
    statistics = config.get('statistics', config)
    if not isinstance(statistics, Mapping):
        raise TypeError('SimateBeta q01_q99 action normalization requires a statistics mapping.')
    try:
        q01 = tuple((float(value) for value in statistics.get('q01', ())))
        q99 = tuple((float(value) for value in statistics.get('q99', ())))
    except (TypeError, ValueError) as error:
        raise ValueError('SimateBeta q01/q99 action statistics must be numeric sequences.') from error
    if not q01 or len(q01) != len(q99) or len(q01) > action_dim:
        raise ValueError(f'SimateBeta q01/q99 action statistics must have one shared non-empty dimension not exceeding action_dim={action_dim}; got q01={len(q01)}, q99={len(q99)}.')
    if not all((np.isfinite(value) for value in (*q01, *q99))):
        raise ValueError('SimateBeta q01/q99 action statistics must contain only finite values.')
    if any((high < low for (low, high) in zip(q01, q99, strict=True))):
        raise ValueError('SimateBeta q99 action statistics must be greater than or equal to q01.')
    return {'mode': mode, 'statistics': {'q01': q01, 'q99': q99}}

def _normalize_actions(actions: torch.Tensor, config: Mapping[str, Any] | None) -> torch.Tensor:
    if config is None:
        return actions
    statistics = config['statistics']
    q01 = actions.new_tensor(statistics['q01'])
    q99 = actions.new_tensor(statistics['q99'])
    if actions.shape[-1] < q01.numel():
        raise ValueError(f'CanonicalBatch action dimension is smaller than SimateBeta action normalization statistics: actions={actions.shape[-1]}, statistics={q01.numel()}.')
    normalized = actions.clone()
    normalized[..., :q01.numel()] = (actions[..., :q01.numel()] - q01) / (q99 - q01 + 1e-06) * 2 - 1
    return normalized

def _sample_beta_1p5_1(batch_size: int, *, device: torch.device, generator: torch.Generator | None=None) -> torch.Tensor:
    return torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator).pow(2.0 / 3.0)

def _make_flow_matching_sample(actions: torch.Tensor, *, noise: torch.Tensor | None=None, time: torch.Tensor | None=None, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = actions.float()
    if noise is None:
        noise = torch.randn(actions.shape, device=actions.device, dtype=torch.float32, generator=generator)
    else:
        if noise.shape != actions.shape or not noise.is_floating_point() or noise.device != actions.device:
            raise ValueError('SimateBeta noise must match the floating-point actions shape and device.')
        noise = noise.float()
    if time is None:
        time = _sample_beta_1p5_1(actions.shape[0], device=actions.device, generator=generator) * 0.999 + 0.001
    else:
        if time.shape != (actions.shape[0],) or not time.is_floating_point() or time.device != actions.device:
            raise ValueError('SimateBeta time must be floating point with shape [batch] on the actions device.')
        time = time.float()
    with _autocast_disabled(actions.device):
        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1.0 - time_expanded) * actions
        target_velocity = noise - actions
    return (noisy_actions, target_velocity, time)

def _validate_counterfactual_pair_flow(config: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {'enabled': False, 'pair_id_key': 'counterfactual_pair_id', 'training_mode': 'common_rollout_solver_grid', 'common_loss_weight': 0.5, 'rollout_loss_weight': 0.5}
    if config is None:
        return defaults
    if not isinstance(config, Mapping):
        raise TypeError('counterfactual_pair_flow must be a mapping')
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f'Unsupported counterfactual_pair_flow key(s): {sorted(unknown)}')
    training_mode = str(config.get('training_mode', defaults['training_mode']))
    if training_mode != 'common_rollout_solver_grid':
        raise ValueError("counterfactual_pair_flow.training_mode must be 'common_rollout_solver_grid'")
    common_weight = float(config.get('common_loss_weight', defaults['common_loss_weight']))
    rollout_weight = float(config.get('rollout_loss_weight', defaults['rollout_loss_weight']))
    if not math.isfinite(common_weight) or not math.isfinite(rollout_weight) or common_weight <= 0.0 or (rollout_weight <= 0.0) or (not math.isclose(common_weight + rollout_weight, 1.0, rel_tol=0.0, abs_tol=1e-08)):
        raise ValueError('counterfactual_pair_flow common/rollout loss weights must be positive, finite, and sum to one')
    return {'enabled': bool(config.get('enabled', True)), 'pair_id_key': str(config.get('pair_id_key', defaults['pair_id_key'])), 'training_mode': training_mode, 'common_loss_weight': common_weight, 'rollout_loss_weight': rollout_weight}

def _counterfactual_pair_groups(batch: CanonicalBatch, *, pair_id_key: str) -> dict[str, tuple[int, int]]:
    metadata = (batch.get('meta') or {}).get('meta_data') or ()
    pair_groups: dict[str, list[int]] = {}
    for (index, row) in enumerate(metadata):
        raw_pair_id = row.get(pair_id_key) if isinstance(row, Mapping) else None
        pair_id = '' if raw_pair_id is None else str(raw_pair_id)
        if pair_id:
            pair_groups.setdefault(pair_id, []).append(index)
    malformed = sorted((pair_id for (pair_id, indices) in pair_groups.items() if len(indices) != 2))
    if malformed:
        raise ValueError(f'Counterfactual pair IDs must have exactly two rows inside a training batch; malformed IDs={malformed}')
    for (pair_id, indices) in pair_groups.items():
        roles = {str(metadata[index].get('counterfactual_action_sample_kind', '')) for index in indices if isinstance(metadata[index], Mapping)}
        if roles != {'decision_positive', 'decision_negative'}:
            raise ValueError(f'Counterfactual pair {pair_id!r} must contain one decision_positive and one decision_negative row')
    return {pair_id: (indices[0], indices[1]) for (pair_id, indices) in pair_groups.items()}

def _counterfactual_solver_grid_time(solver_node_index: int, *, num_inference_steps: int) -> float:
    if num_inference_steps < 2:
        raise ValueError('Counterfactual solver-grid supervision requires num_inference_steps >= 2')
    if not 0 <= solver_node_index < num_inference_steps:
        raise ValueError('solver_node_index is outside the deployed inference grid')
    return 1.0 - solver_node_index / float(num_inference_steps)

def _counterfactual_common_state(actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor, pair_groups: Mapping[str, tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if actions.shape != noise.shape or time.shape != (actions.shape[0],):
        raise ValueError('Counterfactual common-state inputs have incompatible shapes')
    shared_noise = noise.float().clone()
    endpoints = actions.float().clone()
    pair_mask = torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
    for (first, second) in pair_groups.values():
        shared_noise[second] = shared_noise[first]
        midpoint = 0.5 * (actions[first].float() + actions[second].float())
        endpoints[first] = midpoint
        endpoints[second] = midpoint
        pair_mask[first] = True
        pair_mask[second] = True
    with _autocast_disabled(actions.device):
        expanded_time = time.float()[:, None, None]
        state = expanded_time * shared_noise + (1.0 - expanded_time) * endpoints
    return (state, pair_mask, shared_noise)

def _conditional_velocity_target(state: torch.Tensor, actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    if state.shape != actions.shape or time.shape != (actions.shape[0],):
        raise ValueError('Conditional velocity inputs have incompatible shapes')
    if bool((time <= 0).any().item()):
        raise ValueError('Conditional velocity target requires strictly positive flow time')
    with _autocast_disabled(actions.device):
        return (state.float() - actions.float()) / time.float()[:, None, None]

def _global_counterfactual_pair_present(local_present: bool, device: torch.device) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_present
    flag = torch.tensor(int(local_present), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())

def _append_prefix_rows(prefix: SimatePrefixContext, row_mask: torch.Tensor) -> SimatePrefixContext:
    row_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
    return SimatePrefixContext(input_mask=torch.cat([prefix.input_mask, prefix.input_mask.index_select(0, row_indices)], dim=0), layer_kv=tuple((LayerKVCache(key=torch.cat([cached.key, cached.key.index_select(0, row_indices)], dim=0), value=torch.cat([cached.value, cached.value.index_select(0, row_indices)], dim=0)) for cached in prefix.layer_kv)))

def _append_suffix_rows(causal: TokenBlock, solver_path: TokenBlock) -> TokenBlock:
    if not torch.equal(causal.block_starts, solver_path.block_starts):
        raise ValueError('Causal and solver-path suffix layouts differ')
    if (causal.condition is None) != (solver_path.condition is None):
        raise ValueError('Causal and solver-path suffix conditions differ')
    condition = None if causal.condition is None else torch.cat([causal.condition, solver_path.condition], dim=0)
    return TokenBlock(embeddings=torch.cat([causal.embeddings, solver_path.embeddings], dim=0), valid_mask=torch.cat([causal.valid_mask, solver_path.valid_mask], dim=0), block_starts=causal.block_starts, condition=condition)

def _validate_action_loss_weight_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {'enabled': False, 'key': 'value_weight', 'default': 1.0, 'normalize': 'mean_one', 'clip': None}
    if not isinstance(config, Mapping):
        raise TypeError('SimateBeta action_loss_weight must be a mapping.')
    enabled = bool(config.get('enabled', True))
    key = str(config.get('key', 'value_weight'))
    default = float(config.get('default', 1.0))
    normalize = str(config.get('normalize', 'mean_one')).lower()
    if normalize not in {'none', 'mean_one'}:
        raise ValueError(f"Unsupported SimateBeta action_loss_weight.normalize={normalize!r}; expected 'none' or 'mean_one'.")
    clip = config.get('clip', None)
    if clip is not None:
        if not isinstance(clip, Sequence) or isinstance(clip, (str, bytes)) or len(clip) != 2:
            raise ValueError('SimateBeta action_loss_weight.clip must be a [min, max] sequence.')
        clip = (float(clip[0]), float(clip[1]))
        if not torch.isfinite(torch.tensor(clip)).all().item() or clip[0] < 0 or clip[1] < clip[0]:
            raise ValueError('SimateBeta action_loss_weight.clip must be finite and satisfy 0 <= min <= max.')
    if default < 0:
        raise ValueError('SimateBeta action_loss_weight.default must be non-negative.')
    return {'enabled': enabled, 'key': key, 'default': default, 'normalize': normalize, 'clip': clip}

def _loss_value_weights(batch: CanonicalBatch, *, config: Mapping[str, Any], batch_size: int, device: torch.device, horizon: int | None=None) -> torch.Tensor | None:
    if not bool(config.get('enabled', False)):
        return None
    key = str(config.get('key', 'value_weight'))
    default = float(config.get('default', 1.0))
    meta = batch.get('meta', {}) if isinstance(batch, Mapping) else {}
    raw_values = meta.get(key)
    if raw_values is None:
        metadata_rows = meta.get('meta_data')
        if metadata_rows is not None:
            rows = list(batch_values(metadata_rows, batch_size, name='meta_data'))
            raw_values = [row.get(key, default) if isinstance(row, Mapping) else default for row in rows]
    if raw_values is None:
        raw_values = [default] * batch_size
    if isinstance(raw_values, torch.Tensor):
        weights = raw_values.to(device=device, dtype=torch.float32)
    else:
        values = list(batch_values(raw_values, batch_size, name=key))
        rows = [torch.as_tensor(value, device=device, dtype=torch.float32) for value in values]
        vector_lengths = {int(row.numel()) for row in rows if row.ndim == 1}
        if any((row.ndim > 1 for row in rows)) or len(vector_lengths) > 1:
            raise ValueError('SimateBeta per-sample action loss weights must be scalars or equal-length vectors.')
        if vector_lengths:
            resolved_horizon = next(iter(vector_lengths))
            if horizon is not None and resolved_horizon != int(horizon):
                raise ValueError(f'SimateBeta action loss weight horizon must be {horizon}, got {resolved_horizon}.')
            rows = [row.expand(resolved_horizon) if row.ndim == 0 else row for row in rows]
            weights = torch.stack(rows)
        else:
            weights = torch.stack(rows)
    if weights.ndim not in {1, 2} or weights.shape[0] != batch_size:
        raise ValueError('SimateBeta action loss value weights must have shape [B] or [B,H].')
    if weights.ndim == 2 and horizon is not None and (weights.shape[1] != int(horizon)):
        raise ValueError(f'SimateBeta action loss weight horizon must be {horizon}, got {weights.shape[1]}.')
    if torch.any(weights < 0) or not bool(torch.isfinite(weights).all().item()):
        raise ValueError('SimateBeta action loss value weights must be finite and non-negative.')
    clip = config.get('clip')
    if clip is not None:
        weights = weights.clamp(min=float(clip[0]), max=float(clip[1]))
    if str(config.get('normalize', 'mean_one')).lower() == 'mean_one':
        positive = weights > 0
        if bool(positive.any().item()):
            weights = weights / weights[positive].mean().clamp_min(1e-08)
    return weights

def _flow_matching_loss(predicted_velocity: torch.Tensor, target_velocity: torch.Tensor, *, value_weight: torch.Tensor | None=None) -> torch.Tensor:
    with _autocast_disabled(predicted_velocity.device):
        per_token_loss = torch.mean(torch.square(predicted_velocity.float() - target_velocity.float()), dim=-1)
        if value_weight is None:
            loss = per_token_loss.mean()
        else:
            weights = value_weight.float()
            if weights.ndim == 1:
                weights = weights[:, None].expand_as(per_token_loss)
            if weights.shape != per_token_loss.shape:
                raise ValueError(f'SimateBeta action loss value weights must broadcast to [B,H]; got weights={tuple(value_weight.shape)}, loss={tuple(per_token_loss.shape)}.')
            positive = weights > 0
            positive_count = positive.sum(dim=1)
            per_sample_weight_sum = weights.sum(dim=1)
            per_sample_loss = (per_token_loss * weights).sum(dim=1) / per_sample_weight_sum.clamp_min(1e-08)
            per_sample_weight = per_sample_weight_sum / positive_count.clamp_min(1)
            loss = (per_sample_loss * per_sample_weight).sum() / per_sample_weight.sum().clamp_min(1e-08)
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(f'SimateBeta action_loss must be a finite scalar, got {loss}.')
    return loss

@ACTION_HEADS.register_module()
class SimateBetaFlowMatchingActionHead(nn.Module):

    def __init__(self, *, transformer_variant: str=SIMATE_BETA_ACTION_VARIANT, action_dim: int=32, action_horizon: int=50, num_inference_steps: int=10, action_normalization: Mapping[str, Any] | None=None, action_loss_weight: Mapping[str, Any] | None=None, counterfactual_pair_flow: Mapping[str, Any] | None=None, dtype: str | torch.dtype='bfloat16', init_device: torch.device | str | None=None, gemma_config: GemmaConfig | None=None, gradient_checkpointing: bool=False, layers: Sequence[nn.Module] | None=None, final_norms: Sequence[nn.Module] | None=None) -> None:
        super().__init__()
        if transformer_variant != SIMATE_BETA_ACTION_VARIANT:
            raise ValueError(f'SimateBetaFlowMatchingActionHead only supports transformer_variant={SIMATE_BETA_ACTION_VARIANT!r}.')
        self.config = gemma_config or GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256, adaptive_norm=True)
        self.param_dtype = resolve_torch_dtype(dtype)
        self.compute_dtype = self.param_dtype
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.transformer_variant = transformer_variant
        self.action_dim = _positive_int('action_dim', action_dim)
        self.action_horizon = _positive_int('action_horizon', action_horizon)
        self.num_inference_steps = _positive_int('num_inference_steps', num_inference_steps)
        self.action_normalization = _validate_action_normalization(action_normalization, action_dim=self.action_dim)
        self.action_loss_weight = _validate_action_loss_weight_config(action_loss_weight)
        self.counterfactual_pair_flow = _validate_counterfactual_pair_flow(counterfactual_pair_flow)
        if self.counterfactual_pair_flow['enabled'] and self.num_inference_steps < 2:
            raise ValueError('Counterfactual solver-grid supervision requires num_inference_steps >= 2')
        self._counterfactual_solver_grid_step = 0
        self.action_in_proj = nn.Linear(self.action_dim, self.config.width, dtype=self.param_dtype, device=init_device)
        self.time_mlp_in = nn.Linear(self.config.width, self.config.width, dtype=self.param_dtype, device=init_device)
        self.time_mlp_out = nn.Linear(self.config.width, self.config.width, dtype=self.param_dtype, device=init_device)
        self.action_out_proj = nn.Linear(self.config.width, self.action_dim, dtype=self.param_dtype, device=init_device)
        self.layers = nn.ModuleList(tuple(layers) if layers is not None else (GemmaDecoderLayer(self.config, dtype=self.param_dtype, device=init_device) for _ in range(self.config.depth)))
        self.final_norms = nn.ModuleList(tuple(final_norms) if final_norms is not None else (AdaptiveGemmaRMSNorm(self.config.width, dtype=self.param_dtype, device=init_device),))
        for projection in (self.action_in_proj, self.time_mlp_in, self.time_mlp_out, self.action_out_proj):
            nn.init.normal_(projection.weight, std=0.02)
            nn.init.zeros_(projection.bias)

    def set_compute_dtype(self, dtype: str | torch.dtype) -> None:
        self.compute_dtype = resolve_torch_dtype(dtype)

    def prepare_canonical_actions(self, batch: CanonicalBatch, *, device: torch.device | str) -> torch.Tensor:
        (_, _, batch_size) = validate_canonical_sections(batch, require_actions=True)
        actions = batch['targets']['actions']
        target_device = torch.device(device)
        if isinstance(actions, torch.Tensor):
            tensor = actions
            if tensor.ndim != 3 or tensor.shape[0] != batch_size:
                raise ValueError('CanonicalBatch tensor actions must have shape [B,H,D].')
            if not tensor.is_floating_point():
                tensor = tensor.float()
            tensor = tensor.to(device=target_device, dtype=torch.float32)
        else:
            rows = [numeric_payload(payload, name='action') for payload in batch_values(actions, batch_size, name='actions')]
            shapes = {row.shape for row in rows}
            if len(shapes) != 1 or not rows or rows[0].ndim != 2:
                raise ValueError('CanonicalBatch action rows must share shape [H,D].')
            tensor = torch.from_numpy(np.stack(rows).astype(np.float32, copy=False)).to(target_device)
        if tensor.shape[1] != self.action_horizon:
            raise ValueError(f'CanonicalBatch action horizon must be {self.action_horizon}, got {tensor.shape[1]}.')
        if tensor.shape[2] > self.action_dim:
            raise ValueError(f'CanonicalBatch action dimension {tensor.shape[2]} exceeds SimateBeta action_dim={self.action_dim}.')
        tensor = _normalize_actions(tensor, self.action_normalization)
        return F.pad(tensor, (0, self.action_dim - tensor.shape[2])) if tensor.shape[2] < self.action_dim else tensor

    def embed_suffix(self, noisy_actions: torch.Tensor, timestep: torch.Tensor) -> TokenBlock:
        with _autocast_disabled(noisy_actions.device):
            action_tokens = F.linear(noisy_actions.float(), self.action_in_proj.weight.float(), self.action_in_proj.bias.float())
            condition = F.linear(_posemb_sincos(timestep.float(), self.config.width), self.time_mlp_in.weight.float(), self.time_mlp_in.bias.float())
            condition = F.silu(condition)
            condition = F.linear(condition, self.time_mlp_out.weight.float(), self.time_mlp_out.bias.float())
            condition = F.silu(condition)
        batch_size = noisy_actions.shape[0]
        block_starts = torch.zeros(self.action_horizon, dtype=torch.bool, device=noisy_actions.device)
        block_starts[0] = True
        return TokenBlock(embeddings=action_tokens.to(self.compute_dtype), valid_mask=torch.ones(batch_size, self.action_horizon, dtype=torch.bool, device=noisy_actions.device), block_starts=block_starts, condition=condition.to(self.compute_dtype))

    def project_velocity(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with _autocast_disabled(hidden_states.device):
            return F.linear(hidden_states.float(), self.action_out_proj.weight.float(), self.action_out_proj.bias.float())

    def _rollout_to_solver_node(self, encoded: SimatePrefixContext, noise: torch.Tensor, *, solver_node_index: int) -> torch.Tensor:
        _counterfactual_solver_grid_time(solver_node_index, num_inference_steps=self.num_inference_steps)
        state = noise.float()
        step_size = 1.0 / self.num_inference_steps
        with torch.no_grad():
            for preceding_index in range(solver_node_index):
                time_value = 1.0 - preceding_index / float(self.num_inference_steps)
                rollout_time = torch.full((state.shape[0],), time_value, dtype=torch.float32, device=state.device)
                velocity = self._predict_velocity(encoded, self.embed_suffix(state, rollout_time))
                with _autocast_disabled(state.device):
                    state = state - step_size * velocity.float()
        return state.detach()

    def compute_loss(self, encoded: SimatePrefixContext, *, batch: CanonicalBatch, noise: torch.Tensor | None=None, time: torch.Tensor | None=None, generator: torch.Generator | None=None, **kwargs: Any) -> Mapping[str, torch.Tensor]:
        del kwargs
        actions = self.prepare_canonical_actions(batch, device=encoded.input_mask.device)
        ordinary_noise = torch.randn(actions.shape, device=actions.device, dtype=torch.float32, generator=generator) if noise is None else noise
        (ordinary_state, ordinary_target, ordinary_time) = _make_flow_matching_sample(actions, noise=ordinary_noise, time=time, generator=generator)
        value_weight = _loss_value_weights(batch, config=self.action_loss_weight, batch_size=actions.shape[0], device=actions.device, horizon=actions.shape[1])
        pair_groups = _counterfactual_pair_groups(batch, pair_id_key=str(self.counterfactual_pair_flow['pair_id_key'])) if bool(self.counterfactual_pair_flow['enabled']) else {}
        global_pair_present = _global_counterfactual_pair_present(bool(pair_groups), actions.device)
        if not global_pair_present:
            velocity = self._predict_velocity(encoded, self.embed_suffix(ordinary_state, ordinary_time))
            return {'action_loss': _flow_matching_loss(velocity, ordinary_target, value_weight=value_weight)}
        solver_node_index = self._counterfactual_solver_grid_step % self.num_inference_steps
        self._counterfactual_solver_grid_step += 1
        solver_time_value = _counterfactual_solver_grid_time(solver_node_index, num_inference_steps=self.num_inference_steps)
        solver_time = torch.full((actions.shape[0],), solver_time_value, dtype=torch.float32, device=actions.device)
        (common_state, pair_mask, shared_noise) = _counterfactual_common_state(actions, ordinary_noise, solver_time, pair_groups)
        rollout_state = self._rollout_to_solver_node(encoded, shared_noise, solver_node_index=solver_node_index)
        common_target = _conditional_velocity_target(common_state, actions, solver_time)
        rollout_target = _conditional_velocity_target(rollout_state, actions, solver_time)
        primary_state = ordinary_state.clone()
        primary_time = ordinary_time.clone()
        primary_target = ordinary_target.clone()
        primary_state[pair_mask] = common_state[pair_mask]
        primary_time[pair_mask] = solver_time[pair_mask]
        primary_target[pair_mask] = common_target[pair_mask]
        primary_suffix = self.embed_suffix(primary_state, primary_time)
        rollout_suffix = self.embed_suffix(rollout_state, solver_time)
        all_rows = torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
        combined_prefix = _append_prefix_rows(encoded, all_rows)
        combined_suffix = _append_suffix_rows(primary_suffix, rollout_suffix)
        combined_velocity = self._predict_velocity(combined_prefix, combined_suffix)
        base_weight = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.float32) if value_weight is None else value_weight[:, None].expand(actions.shape[:2]) if value_weight.ndim == 1 else value_weight
        primary_weight = base_weight.clone()
        primary_weight[pair_mask] *= float(self.counterfactual_pair_flow['common_loss_weight'])
        rollout_weight = torch.zeros_like(base_weight)
        rollout_weight[pair_mask] = base_weight[pair_mask] * float(self.counterfactual_pair_flow['rollout_loss_weight'])
        combined_target = torch.cat([primary_target, rollout_target], dim=0)
        combined_weight = torch.cat([primary_weight, rollout_weight], dim=0)
        return {'action_loss': _flow_matching_loss(combined_velocity, combined_target, value_weight=combined_weight)}

    @torch.inference_mode()
    def predict_action(self, encoded: SimatePrefixContext, *, batch: CanonicalBatch | None=None, num_steps: int | None=None, noise: torch.Tensor | None=None, generator: torch.Generator | None=None, **kwargs: Any) -> Mapping[str, torch.Tensor]:
        del batch, kwargs
        steps = self.num_inference_steps if num_steps is None else _positive_int('num_steps', num_steps)
        batch_size = encoded.input_mask.shape[0]
        device = encoded.input_mask.device
        shape = (batch_size, self.action_horizon, self.action_dim)
        x_t = torch.randn(shape, device=device, dtype=torch.float32, generator=generator) if noise is None else noise.to(device=device, dtype=torch.float32)
        dt = -1.0 / steps
        time_value = 1.0
        while time_value >= -dt / 2:
            time = torch.full((batch_size,), time_value, dtype=torch.float32, device=device)
            velocity = self._predict_velocity(encoded, self.embed_suffix(x_t, time))
            with _autocast_disabled(device):
                x_t = x_t + dt * velocity.float()
            time_value += dt
        return {'normalized_actions': x_t}

    def _predict_velocity(self, prefix: SimatePrefixContext, suffix: TokenBlock) -> torch.Tensor:
        suffix_layout = build_block_causal_attention(suffix.valid_mask, suffix.block_starts)
        prefix_to_suffix = prefix.input_mask[:, None, :].expand(-1, suffix.embeddings.shape[1], -1)
        attention_mask = torch.cat([prefix_to_suffix, suffix_layout.attention_mask], dim=-1)
        positions = prefix.input_mask.sum(dim=-1)[:, None] + torch.cumsum(suffix.valid_mask.int(), dim=-1) - 1
        hidden_states = suffix.embeddings
        for (layer, cached) in zip(self.layers, prefix.layer_kv, strict=True):
            args = (hidden_states, positions, attention_mask, cached, suffix.condition)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                (hidden_states, _) = checkpoint(layer, *args, use_reentrant=False)
            else:
                (hidden_states, _) = layer(*args)
        (hidden_states, _) = self.final_norms[0](hidden_states, suffix.condition)
        return self.project_velocity(hidden_states[:, -self.action_horizon:])

def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}.')
    return value
__all__ = ['SimateBetaFlowMatchingActionHead']
