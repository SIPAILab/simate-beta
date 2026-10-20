from __future__ import annotations
import numpy as np

def joint_difference(target: object, reference: object, *, difference: tuple[str, ...], period: tuple[float | None, ...]) -> np.ndarray:
    target_array = _joint_vector(target, len(difference), 'target joint')
    reference_array = _joint_vector(reference, len(difference), 'reference joint')
    result = target_array - reference_array
    for (index, mode) in enumerate(difference):
        if mode == 'linear':
            continue
        if mode != 'shortest_periodic':
            raise ValueError(f'Unsupported joint difference mode: {mode!r}')
        channel_period = period[index]
        if channel_period is None or channel_period <= 0.0:
            raise ValueError(f'Periodic joint channel {index} requires a positive period')
        result[index] = (result[index] + channel_period / 2.0) % channel_period - channel_period / 2.0
    return result

def compose_joint(delta: object, reference: object, *, difference: tuple[str, ...], period: tuple[float | None, ...], canonical_min: tuple[float | None, ...]) -> np.ndarray:
    delta_array = _joint_vector(delta, len(difference), 'joint delta')
    reference_array = _joint_vector(reference, len(difference), 'reference joint')
    return canonicalize_joint(reference_array + delta_array, difference=difference, period=period, canonical_min=canonical_min)

def canonicalize_joint(value: object, *, difference: tuple[str, ...], period: tuple[float | None, ...], canonical_min: tuple[float | None, ...]) -> np.ndarray:
    result = _joint_vector(value, len(difference), 'joint').copy()
    for (index, mode) in enumerate(difference):
        if mode == 'linear':
            continue
        if mode != 'shortest_periodic':
            raise ValueError(f'Unsupported joint difference mode: {mode!r}')
        channel_period = period[index]
        lower = canonical_min[index]
        if channel_period is None or channel_period <= 0.0 or lower is None:
            raise ValueError(f'Periodic joint channel {index} has incomplete canonical metadata')
        result[index] = (result[index] - lower) % channel_period + lower
    return result

def _joint_vector(value: object, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f'{label.capitalize()} must have shape ({size},), got {vector.shape}')
    if not np.all(np.isfinite(vector)):
        raise ValueError(f'{label.capitalize()} contains non-finite values')
    return vector
