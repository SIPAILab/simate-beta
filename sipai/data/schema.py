from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence, TypedDict

class StreamRef(TypedDict, total=False):
    index: int
    index_range: list[int]
    timestamp: float
    time_range: list[float]

class Observation(TypedDict):
    cameras: dict[str, StreamRef]
    tactile: dict[str, StreamRef]
    state: Optional[dict[str, Any]]

class PerceptionGT(TypedDict):
    depth: dict[str, Any]
    segmentation: dict[str, Any]
    objects: list[Any]
    keypoints: list[Any]
    scene_graph: Optional[Any]
    occupancy: Optional[Any]
    source: Optional[str]
    confidence: Optional[float]
    extra: dict[str, Any]

class Step(TypedDict):
    timestamp: float
    observation: Observation
    action: Optional[dict[str, Any]]
    perception_gt: PerceptionGT
    sub_task_id: Optional[str]
    meta_data: dict[str, Any]
    extra: dict[str, Any]

class StorageManifest(TypedDict):
    backend: str
    root: str
    streams: dict[str, dict[str, Any]]
    extra: dict[str, Any]

class EpisodeManifest(TypedDict):
    episode_id: str
    storage: StorageManifest
    task: dict[str, Any]
    timeline: dict[str, Any]
    sub_tasks: list[dict[str, Any]]
    step_index: dict[str, Any]
    step_overrides: list[dict[str, Any]]
    steps: list[Step]
    meta_data: dict[str, Any]
    source_info: dict[str, Any]
    extra: dict[str, Any]

class DatasetManifest(TypedDict):
    dataset_name: str
    embodiment: dict[str, Any]
    episodes: list[EpisodeManifest]
    extra: dict[str, Any]

class CanonicalBatch(TypedDict):
    inputs: dict[str, Any]
    targets: dict[str, Any]
    masks: dict[str, Any]
    meta: dict[str, Any]

def make_dataset_manifest(*, dataset_name: str, embodiment: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]], extra: Optional[Mapping[str, Any]]=None) -> DatasetManifest:
    return DatasetManifest(dataset_name=dataset_name, embodiment=deepcopy(dict(embodiment)), episodes=[deepcopy(dict(episode)) for episode in episodes], extra=deepcopy(dict(extra or {})))

def make_episode_manifest(*, episode_id: str, storage: Mapping[str, Any], task: Mapping[str, Any], timeline: Mapping[str, Any], sub_tasks: Sequence[Mapping[str, Any]], meta_data: Mapping[str, Any], source_info: Mapping[str, Any], steps: Optional[Sequence[Mapping[str, Any]]]=None, step_index: Optional[Mapping[str, Any]]=None, step_overrides: Optional[Sequence[Mapping[str, Any]]]=None, extra: Optional[Mapping[str, Any]]=None) -> EpisodeManifest:
    return EpisodeManifest(episode_id=episode_id, storage=deepcopy(dict(storage)), task=deepcopy(dict(task)), timeline=deepcopy(dict(timeline)), sub_tasks=[deepcopy(dict(sub_task)) for sub_task in sub_tasks], step_index=deepcopy(dict(step_index or {})), step_overrides=[deepcopy(dict(item)) for item in step_overrides or []], steps=[deepcopy(dict(step)) for step in steps or []], meta_data=deepcopy(dict(meta_data)), source_info=deepcopy(dict(source_info)), extra=deepcopy(dict(extra or {})))

def make_storage_manifest(*, backend: str, root: str, streams: Mapping[str, Mapping[str, Any]], extra: Optional[Mapping[str, Any]]=None) -> StorageManifest:
    return StorageManifest(backend=backend, root=root, streams=deepcopy(dict(streams)), extra=deepcopy(dict(extra or {})))

def make_step(*, timestamp: float, cameras: Mapping[str, Mapping[str, Any]], tactile: Optional[Mapping[str, Mapping[str, Any]]]=None, state: Optional[Mapping[str, Any]]=None, action: Optional[Mapping[str, Any]]=None, perception_gt: Optional[Mapping[str, Any]]=None, sub_task_id: Optional[str]=None, meta_data: Optional[Mapping[str, Any]]=None, extra: Optional[Mapping[str, Any]]=None) -> Step:
    return Step(timestamp=timestamp, observation=Observation(cameras=deepcopy(dict(cameras)), tactile=deepcopy(dict(tactile or {})), state=deepcopy(dict(state)) if state is not None else None), action=deepcopy(dict(action)) if action is not None else None, perception_gt=deepcopy(dict(perception_gt or make_empty_perception_gt())), sub_task_id=sub_task_id, meta_data=deepcopy(dict(meta_data or {})), extra=deepcopy(dict(extra or {})))

def make_empty_perception_gt() -> PerceptionGT:
    return PerceptionGT(depth={}, segmentation={}, objects=[], keypoints=[], scene_graph=None, occupancy=None, source=None, confidence=None, extra={})
