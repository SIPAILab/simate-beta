from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Mapping
import numpy as np
NORMALIZATION_MODES = frozenset({'none', 'mean_std', 'min_max', 'q01_q99'})
_STATISTIC_NAMES = ('mean', 'std', 'min', 'max', 'q01', 'q99')

@dataclass(frozen=True)
class NormalizationStatistics:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    min: np.ndarray | None = None
    max: np.ndarray | None = None
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f'Normalization statistic {item.name!r} must be a one-dimensional array')
            array = np.array(array, dtype=np.float64, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, item.name, array)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | NormalizationStatistics, *, mode: str | None=None, semantic_dim: int | None=None) -> NormalizationStatistics:
        if isinstance(value, cls):
            result = value
        else:
            if not isinstance(value, Mapping):
                raise ValueError('Normalization statistics must be a mapping')
            payload = value.get('statistics', value)
            if not isinstance(payload, Mapping):
                raise ValueError('Normalization statistics payload must be a mapping')
            result = cls(**{name: payload.get(name) for name in _STATISTIC_NAMES})
        if mode is not None:
            if semantic_dim is None:
                semantic_dim = result.inferred_dim(mode)
            result.validate(mode=mode, semantic_dim=semantic_dim)
        elif semantic_dim is not None:
            result.validate_available_shapes(semantic_dim)
        return result

    def to_dict(self) -> dict[str, list[float]]:
        return {name: getattr(self, name).tolist() for name in _STATISTIC_NAMES if getattr(self, name) is not None}

    def inferred_dim(self, mode: str) -> int:
        required = _required_statistics(_validate_mode(mode))
        for name in required:
            value = getattr(self, name)
            if value is not None:
                return int(value.shape[0])
        raise ValueError(f"Normalization mode {mode!r} requires statistics: {', '.join(required)}")

    def validate_available_shapes(self, semantic_dim: int) -> None:
        semantic_dim = _validate_semantic_dim(semantic_dim)
        for name in _STATISTIC_NAMES:
            value = getattr(self, name)
            if value is None:
                continue
            if value.shape != (semantic_dim,):
                raise ValueError(f'Normalization statistic {name!r} must have shape ({semantic_dim},), got {value.shape}')
            if not np.all(np.isfinite(value)):
                raise ValueError(f'Normalization statistic {name!r} contains non-finite values')

    def validate(self, *, mode: str, semantic_dim: int) -> None:
        mode = _validate_mode(mode)
        semantic_dim = _validate_semantic_dim(semantic_dim)
        if mode == 'none':
            return
        required = _required_statistics(mode)
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Normalization mode {mode!r} requires statistic array(s): {', '.join(missing)}")
        for name in required:
            value = getattr(self, name)
            assert value is not None
            if value.shape != (semantic_dim,):
                raise ValueError(f'Normalization statistic {name!r} must have shape ({semantic_dim},), got {value.shape}')
            if not np.all(np.isfinite(value)):
                raise ValueError(f'Normalization statistic {name!r} contains non-finite values')
        if mode == 'mean_std':
            assert self.std is not None
            if np.any(self.std < 0.0):
                raise ValueError('mean_std statistics require std >= 0 in every channel')
            return
        assert self.min is not None and self.max is not None
        if np.any(self.max < self.min):
            raise ValueError(f'{mode} statistics require max >= min in every channel')
        if mode == 'q01_q99':
            assert self.q01 is not None and self.q99 is not None
            if np.any(self.q99 < self.q01):
                raise ValueError('q01_q99 statistics require q99 >= q01 in every channel')
            collapsed_nonconstant = (self.q99 == self.q01) & (self.max > self.min)
            if np.any(collapsed_nonconstant):
                channels = np.flatnonzero(collapsed_nonconstant).tolist()
                raise ValueError('q01_q99 interval collapsed for non-constant channel(s): ' + ', '.join((str(channel) for channel in channels)))

def normalize_values(values: object, *, mode: str, statistics: Mapping[str, Any] | NormalizationStatistics | None=None, semantic_dim: int | None=None) -> np.ndarray:
    return _apply(values, mode=mode, statistics=statistics, semantic_dim=semantic_dim, inverse=False)

def unnormalize_values(values: object, *, mode: str, statistics: Mapping[str, Any] | NormalizationStatistics | None=None, semantic_dim: int | None=None) -> np.ndarray:
    return _apply(values, mode=mode, statistics=statistics, semantic_dim=semantic_dim, inverse=True)
normalize = normalize_values
unnormalize = unnormalize_values

def _apply(values: object, *, mode: str, statistics: Mapping[str, Any] | NormalizationStatistics | None, semantic_dim: int | None, inverse: bool) -> np.ndarray:
    mode = _validate_mode(mode)
    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError('Normalization input must have at least one dimension')
    if semantic_dim is None:
        semantic_dim = int(array.shape[-1])
    semantic_dim = _validate_semantic_dim(semantic_dim)
    if array.shape[-1] != semantic_dim:
        raise ValueError(f'Normalization input last dimension must be semantic_dim={semantic_dim}, got {array.shape}')
    work = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(work)):
        raise ValueError('Normalization input contains non-finite values')
    if mode == 'none':
        return _restore_float_dtype(work, array)
    if statistics is None:
        raise ValueError(f'Normalization mode {mode!r} requires statistics')
    stats = NormalizationStatistics.from_dict(statistics, mode=mode, semantic_dim=semantic_dim)
    if mode == 'mean_std':
        assert stats.mean is not None and stats.std is not None
        constant = stats.std == 0.0
        if inverse:
            result = work * stats.std + stats.mean
            result = np.where(constant, stats.mean, result)
        else:
            result = np.zeros_like(work)
            np.divide(work - stats.mean, stats.std, out=result, where=~constant)
    else:
        lower = stats.min if mode == 'min_max' else stats.q01
        upper = stats.max if mode == 'min_max' else stats.q99
        assert lower is not None and upper is not None
        constant = stats.max == stats.min if mode == 'q01_q99' else upper == lower
        assert constant is not None
        span = upper - lower
        if inverse:
            result = (work + 1.0) * span / 2.0 + lower
            result = np.where(constant, lower if mode == 'min_max' else stats.min, result)
        else:
            result = np.zeros_like(work)
            np.divide(2.0 * (work - lower), span, out=result, where=~constant)
            result -= np.where(constant, 0.0, 1.0)
    if not np.all(np.isfinite(result)):
        raise ValueError('Normalization produced non-finite values')
    return _restore_float_dtype(result, array)

def _required_statistics(mode: str) -> tuple[str, ...]:
    if mode == 'mean_std':
        return ('mean', 'std')
    if mode == 'min_max':
        return ('min', 'max')
    if mode == 'q01_q99':
        return ('q01', 'q99', 'min', 'max')
    return ()

def _validate_mode(mode: str) -> str:
    resolved = str(mode).lower()
    if resolved not in NORMALIZATION_MODES:
        raise ValueError(f'Unsupported normalization mode: {mode!r}')
    return resolved

def _validate_semantic_dim(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError('semantic_dim must be an integer >= 1')
    return int(value)

def _restore_float_dtype(result: np.ndarray, source: np.ndarray) -> np.ndarray:
    if np.issubdtype(source.dtype, np.floating):
        return result.astype(source.dtype, copy=False)
    return result
__all__ = ['NORMALIZATION_MODES', 'NormalizationStatistics', 'normalize', 'normalize_values', 'unnormalize', 'unnormalize_values']
