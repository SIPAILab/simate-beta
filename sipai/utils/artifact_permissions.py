from __future__ import annotations
import os
import stat
from pathlib import Path
SHARED_ARTIFACT_DIRECTORY_MODE = 1535
SHARED_ARTIFACT_FILE_MODE = 438

def ensure_shared_artifact_directory(path: str | Path) -> Path:
    directory = Path(path)
    missing_directories: list[Path] = []
    candidate = directory
    while not candidate.exists():
        missing_directories.append(candidate)
        candidate = candidate.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created_directory in reversed(missing_directories):
        normalize_shared_artifact_permissions(created_directory)
    if not missing_directories:
        normalize_shared_artifact_permissions(directory)
    return directory

def normalize_shared_artifact_permissions(path: str | Path) -> Path:
    artifact = Path(path)
    file_stat = artifact.lstat()
    if stat.S_ISLNK(file_stat.st_mode):
        return artifact
    if stat.S_ISDIR(file_stat.st_mode):
        target_mode = SHARED_ARTIFACT_DIRECTORY_MODE
    elif stat.S_ISREG(file_stat.st_mode):
        target_mode = SHARED_ARTIFACT_FILE_MODE
    else:
        return artifact
    if stat.S_IMODE(file_stat.st_mode) != target_mode:
        os.chmod(artifact, target_mode, follow_symlinks=False)
    return artifact

def normalize_shared_artifact_tree(path: str | Path) -> Path:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        return normalize_shared_artifact_permissions(root)
    for (current_root, directory_names, file_names) in os.walk(root, followlinks=False):
        current = Path(current_root)
        normalize_shared_artifact_permissions(current)
        for name in directory_names:
            child = current / name
            if not child.is_symlink():
                normalize_shared_artifact_permissions(child)
        for name in file_names:
            child = current / name
            if not child.is_symlink():
                normalize_shared_artifact_permissions(child)
    return root
__all__ = ['SHARED_ARTIFACT_DIRECTORY_MODE', 'SHARED_ARTIFACT_FILE_MODE', 'ensure_shared_artifact_directory', 'normalize_shared_artifact_permissions', 'normalize_shared_artifact_tree']
