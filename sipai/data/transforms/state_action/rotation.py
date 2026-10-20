from __future__ import annotations
from typing import Final
import numpy as np
_NORM_EPS: Final = 1e-12
_HEMISPHERE_TOL: Final = 1e-12
_ROT6D_EPS: Final = 1e-08
_MATRIX_ATOL: Final = 1e-06
ROTATION_DIMS: Final = {'quat_xyzw': 4, 'quat_wxyz': 4, 'rotvec': 3, 'rot6d': 6}

def rotation_dim(encoding: str) -> int:
    try:
        return ROTATION_DIMS[encoding]
    except KeyError as exc:
        raise ValueError(f'Unsupported rotation encoding: {encoding!r}') from exc

def canonicalize_quaternion(quaternion: object) -> np.ndarray:
    quat = _finite_vector(quaternion, 4, 'quaternion')
    norm = float(np.linalg.norm(quat))
    if norm <= _NORM_EPS:
        raise ValueError('Quaternion norm is zero or too small')
    quat = quat / norm
    should_flip = bool(quat[3] < 0.0)
    if quat[3] == 0.0:
        for component in quat[:3]:
            if abs(float(component)) > _HEMISPHERE_TOL:
                should_flip = bool(component < 0.0)
                break
    if should_flip:
        quat = -quat
    return quat

def quaternion_multiply(left: object, right: object) -> np.ndarray:
    (x1, y1, z1, w1) = canonicalize_quaternion(left)
    (x2, y2, z2, w2) = canonicalize_quaternion(right)
    return canonicalize_quaternion(np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], dtype=np.float64))

def quaternion_inverse(quaternion: object) -> np.ndarray:
    quat = canonicalize_quaternion(quaternion)
    return canonicalize_quaternion(np.array([-quat[0], -quat[1], -quat[2], quat[3]]))

def quaternion_to_matrix(quaternion: object) -> np.ndarray:
    (x, y, z, w) = canonicalize_quaternion(quaternion)
    return np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]], dtype=np.float64)

def matrix_to_quaternion(matrix: object) -> np.ndarray:
    rotation = validate_rotation_matrix(matrix)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array([(rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale])
    elif rotation[0, 0] >= rotation[1, 1] and rotation[0, 0] >= rotation[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
        if scale <= _NORM_EPS:
            raise ValueError('Rotation matrix cannot be converted to a quaternion')
        quat = np.array([0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale])
    elif rotation[1, 1] >= rotation[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
        if scale <= _NORM_EPS:
            raise ValueError('Rotation matrix cannot be converted to a quaternion')
        quat = np.array([(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale])
    else:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
        if scale <= _NORM_EPS:
            raise ValueError('Rotation matrix cannot be converted to a quaternion')
        quat = np.array([(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale])
    return canonicalize_quaternion(quat)

def rotvec_to_matrix(rotvec: object) -> np.ndarray:
    vector = _finite_vector(rotvec, 3, 'rotation vector')
    theta = float(np.linalg.norm(vector))
    skew = _skew(vector)
    if theta <= 1e-08:
        theta2 = theta * theta
        a = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return validate_rotation_matrix(np.eye(3) + a * skew + b * (skew @ skew))

def matrix_to_rotvec(matrix: object) -> np.ndarray:
    quat = matrix_to_quaternion(matrix)
    vector = quat[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= _NORM_EPS:
        return np.zeros(3, dtype=np.float64)
    theta = 2.0 * np.arctan2(vector_norm, float(quat[3]))
    return vector * (theta / vector_norm)

def matrix_to_rot6d(matrix: object) -> np.ndarray:
    rotation = validate_rotation_matrix(matrix)
    return np.concatenate((rotation[:, 0], rotation[:, 1]))

def rot6d_to_matrix(rot6d: object) -> np.ndarray:
    value = _finite_vector(rot6d, 6, 'rotation 6D')
    first = value[:3]
    second = value[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm <= _ROT6D_EPS:
        raise ValueError('Rotation 6D first column is degenerate')
    column0 = first / first_norm
    orthogonal = second - float(np.dot(second, column0)) * column0
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    if orthogonal_norm <= _ROT6D_EPS:
        raise ValueError('Rotation 6D columns are collinear or degenerate')
    column1 = orthogonal / orthogonal_norm
    column2 = np.cross(column0, column1)
    return validate_rotation_matrix(np.column_stack((column0, column1, column2)))

def decode_rotation(value: object, encoding: str) -> np.ndarray:
    if encoding == 'quat_xyzw':
        return quaternion_to_matrix(value)
    if encoding == 'quat_wxyz':
        quaternion = _finite_vector(value, 4, 'quaternion')
        return quaternion_to_matrix(quaternion[[1, 2, 3, 0]])
    if encoding == 'rotvec':
        return rotvec_to_matrix(value)
    if encoding == 'rot6d':
        return rot6d_to_matrix(value)
    raise ValueError(f'Unsupported rotation encoding: {encoding!r}')

def encode_rotation(matrix: object, encoding: str) -> np.ndarray:
    if encoding == 'quat_xyzw':
        return matrix_to_quaternion(matrix)
    if encoding == 'quat_wxyz':
        quaternion = matrix_to_quaternion(matrix)
        return quaternion[[3, 0, 1, 2]]
    if encoding == 'rotvec':
        return matrix_to_rotvec(matrix)
    if encoding == 'rot6d':
        return matrix_to_rot6d(matrix)
    raise ValueError(f'Unsupported rotation encoding: {encoding!r}')

def rotation_identity(encoding: str) -> np.ndarray:
    return encode_rotation(np.eye(3), encoding)

def validate_rotation_matrix(matrix: object) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f'Rotation matrix must have shape (3, 3), got {rotation.shape}')
    if not np.all(np.isfinite(rotation)):
        raise ValueError('Rotation matrix contains non-finite values')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError('Rotation matrix is not orthogonal')
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not np.isclose(determinant, 1.0, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError('Rotation matrix must have determinant +1')
    return rotation

def _finite_vector(value: object, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f'{label.capitalize()} must have shape ({size},), got {vector.shape}')
    if not np.all(np.isfinite(vector)):
        raise ValueError(f'{label.capitalize()} contains non-finite values')
    return vector

def _skew(vector: np.ndarray) -> np.ndarray:
    (x, y, z) = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
