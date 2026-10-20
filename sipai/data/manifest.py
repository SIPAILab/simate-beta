from __future__ import annotations
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping
from sipai.utils import ensure_shared_artifact_directory, normalize_shared_artifact_permissions

def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_shared_artifact_directory(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)
        f.write('\n')
    normalize_shared_artifact_permissions(path)

def sample_dataset_manifest_episodes(manifest: Mapping[str, Any], *, sample_size: int | None, seed: int) -> dict[str, Any]:
    out = deepcopy(dict(manifest))
    episodes = list(out.get('episodes', []))
    if sample_size is None or sample_size >= len(episodes):
        return out
    rng = random.Random(int(seed))
    selected_indices = sorted(rng.sample(range(len(episodes)), int(sample_size)))
    out['episodes'] = [episodes[index] for index in selected_indices]
    extra = dict(out.get('extra') or {})
    extra['episode_subset'] = {'sample_size': int(sample_size), 'seed': int(seed), 'source_num_episodes': len(episodes), 'selected_indices': selected_indices}
    out['extra'] = extra
    return out

def build_or_load_dataset_manifest(*, cache_path: str | Path | None, cache_policy: str, builder: Callable[[], dict[str, Any]], cache_key: Mapping[str, Any] | None=None) -> dict[str, Any]:
    policy = (cache_policy or 'auto').lower()
    if cache_path is None:
        return builder()
    path = Path(cache_path)
    if policy == 'auto' and path.exists():
        cached = load_dataset_manifest(path)
        if cache_key is None or _cache_key(cached) == dict(cache_key):
            return cached
    if policy == 'readonly':
        return load_dataset_manifest(path)
    if policy not in {'auto', 'rebuild'}:
        raise ValueError(f'Unsupported manifest cache policy: {cache_policy}')
    manifest = builder()
    if cache_key is not None:
        manifest = _with_cache_key(manifest, cache_key)
    save_dataset_manifest(manifest, path)
    return manifest

def _cache_key(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    extra = manifest.get('extra') if isinstance(manifest, Mapping) else None
    if not isinstance(extra, Mapping):
        return None
    key = extra.get('cache_key')
    return dict(key) if isinstance(key, Mapping) else None

def _with_cache_key(manifest: Mapping[str, Any], cache_key: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(manifest))
    extra = dict(out.get('extra') or {})
    extra['cache_key'] = dict(cache_key)
    out['extra'] = extra
    return out
