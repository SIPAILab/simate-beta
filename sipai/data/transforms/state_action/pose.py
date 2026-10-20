from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .rotation import validate_rotation_matrix

@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

def make_pose(position: object, rotation: object) -> Pose:
    position_array = np.asarray(position, dtype=np.float64)
    if position_array.shape != (3,):
        raise ValueError(f'Pose position must have shape (3,), got {position_array.shape}')
    if not np.all(np.isfinite(position_array)):
        raise ValueError('Pose position contains non-finite values')
    return Pose(position=position_array, rotation=validate_rotation_matrix(rotation))

def pose_difference(target: Pose, reference: Pose, *, axes: str) -> Pose:
    if axes == 'parent':
        return make_pose(target.position - reference.position, target.rotation @ reference.rotation.T)
    if axes == 'anchor':
        reference_rotation_inverse = reference.rotation.T
        return make_pose(reference_rotation_inverse @ (target.position - reference.position), reference_rotation_inverse @ target.rotation)
    if axes == 'parent_position_anchor_rotation':
        return make_pose(target.position - reference.position, reference.rotation.T @ target.rotation)
    raise ValueError(f'Unsupported pose delta axes: {axes!r}')

def compose_pose(delta: Pose, reference: Pose, *, axes: str) -> Pose:
    if axes == 'parent':
        return make_pose(reference.position + delta.position, delta.rotation @ reference.rotation)
    if axes == 'anchor':
        return make_pose(reference.position + reference.rotation @ delta.position, reference.rotation @ delta.rotation)
    if axes == 'parent_position_anchor_rotation':
        return make_pose(reference.position + delta.position, reference.rotation @ delta.rotation)
    raise ValueError(f'Unsupported pose delta axes: {axes!r}')

def identity_pose() -> Pose:
    return make_pose(np.zeros(3), np.eye(3))
