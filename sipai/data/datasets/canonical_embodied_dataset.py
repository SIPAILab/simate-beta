from __future__ import annotations
import copy
import hashlib
import json
import logging
import math
import os
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
from torch.utils.data import Dataset
from sipai.data.readers import BaseStorageReader, build_storage_reader
from sipai.data.transforms import CanonicalSampleTransform
from sipai.data.transforms.state_action.fields import read_dotted_source
from sipai.registry import DATASETS
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SampleIndex:
    episode_index: int
    step_index: int
    window_size: int = 1

@dataclass(frozen=True)
class EpisodeIndex:
    episode_index: int
    num_steps: int
    num_samples: int
    sample_local_indices: tuple[int, ...] | None = None

class StatisticsMaterializedBlock(list):

    def __init__(self, samples: List[Dict[str, Any]], *, trace: Mapping[str, Any]):
        super().__init__(samples)
        self.trace = dict(trace)

class CanonicalSampleLoadError(RuntimeError):

    def __init__(self, message: str, *, context: Mapping[str, Any], cause: BaseException):
        super().__init__(message)
        self.context = dict(context)
        self.__cause__ = cause

@DATASETS.register_module(force=True)
class CanonicalEmbodiedDataset(Dataset):

    def __init__(self, dataset_manifest: Optional[Mapping[str, Any]]=None, data_root: str | Path='.', *, reader_backend: str='reference', reader_options: Optional[Mapping[str, Any]]=None, storage_reader: Optional[BaseStorageReader]=None, transform: Optional[CanonicalSampleTransform]=None, representation_transform: Any=None, state_action_config: Optional[Mapping[str, Any]]=None, language_source: str='sub_task', include_raw: bool=True, action_config: Optional[Mapping[str, Any]]=None, action_supervision: Optional[Mapping[str, Any]]=None, counterfactual_action_supervision: Optional[Mapping[str, Any]]=None, state_recovery_supervision: Optional[Mapping[str, Any]]=None, episode_filter: Optional[Mapping[str, Any]]=None, sample_window_size: int=1, sample_stride: int=1, max_samples_per_episode: Optional[int]=None, camera_window_frame_offsets: Optional[List[int]]=None, camera_history_name: Optional[str]=None, camera_history_frame_offsets: Optional[List[int]]=None, camera_payload_image_size: Optional[List[int]]=None, max_cached_steps: int=0, batch_state_window: bool=True, batch_prefetch: bool=True, batch_prefetch_cameras: bool=True, statistics_state_action_only: bool=False, skip_bad_samples: bool=False, max_skip_attempts: int=32, strict: bool=True):
        self.data_root = Path(data_root)
        self.strict = strict
        self.sample_window_size = max(1, int(sample_window_size))
        self.sample_stride = max(1, int(sample_stride))
        self.max_samples_per_episode = max(1, int(max_samples_per_episode)) if max_samples_per_episode is not None else None
        self.action_config = dict(action_config or {})
        self.action_supervision_config = dict(action_supervision or {})
        self.counterfactual_action_supervision_config = dict(counterfactual_action_supervision or {})
        self._stored_transform = self._resolve_legacy_stored_transform(self.action_config)
        self.representation_transform = representation_transform
        self.state_action_config = dict(state_action_config or {})
        self.state_recovery_supervision_config = self._validate_state_recovery_supervision(state_recovery_supervision)
        self.camera_window_frame_offsets = tuple((int(offset) for offset in camera_window_frame_offsets)) if camera_window_frame_offsets is not None else None
        self.camera_history_name = str(camera_history_name) if camera_history_name is not None else None
        self.camera_history_frame_offsets = tuple((int(offset) for offset in camera_history_frame_offsets)) if camera_history_frame_offsets is not None else None
        if (self.camera_history_name is None) != (self.camera_history_frame_offsets is None):
            raise ValueError('camera_history_name and camera_history_frame_offsets must be configured together')
        if self.camera_history_frame_offsets is not None:
            if not self.camera_history_name:
                raise ValueError('camera_history_name must be non-empty')
            if not self.camera_history_frame_offsets:
                raise ValueError('camera_history_frame_offsets must be non-empty')
            if any((offset >= 0 for offset in self.camera_history_frame_offsets)):
                raise ValueError('camera_history_frame_offsets must contain only negative offsets')
            if tuple(sorted(set(self.camera_history_frame_offsets))) != self.camera_history_frame_offsets:
                raise ValueError('camera_history_frame_offsets must be unique and strictly increasing')
        self.camera_payload_image_size = self._normalize_image_size(camera_payload_image_size)
        self.max_cached_steps = max(0, int(max_cached_steps))
        self._step_cache: OrderedDict[tuple[int, int], Dict[str, Any]] = OrderedDict()
        self._step_cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self.batch_state_window = bool(batch_state_window)
        self.batch_prefetch = bool(batch_prefetch)
        self.statistics_state_action_only = bool(statistics_state_action_only)
        self.batch_prefetch_cameras = bool(batch_prefetch_cameras) and (not self.statistics_state_action_only)
        self._batch_prefetch_stats = {'batches': 0, 'step_refs': 0, 'action_stream_reads': 0, 'action_rows': 0, 'camera_stream_reads': 0, 'camera_frames': 0, 'state_stream_reads': 0, 'state_rows': 0, 'errors': 0}
        self.skip_bad_samples = bool(skip_bad_samples)
        self.max_skip_attempts = max(0, int(max_skip_attempts))
        self.reader: BaseStorageReader = storage_reader or build_storage_reader(self.data_root, reader_backend, **dict(reader_options or {}))
        self.transform = transform or CanonicalSampleTransform(language_source=language_source, include_raw=include_raw, action_config=action_config, action_already_processed=representation_transform is not None)
        if dataset_manifest is not None:
            self.manifest = dict(dataset_manifest)
        else:
            raise ValueError('CanonicalEmbodiedDataset requires `dataset_manifest`')
        self.dataset_name = self.manifest.get('dataset_name', 'canonical_embodied_dataset')
        self.embodiment = self.manifest.get('embodiment', {})
        self.episodes: List[Dict[str, Any]] = list(self.manifest.get('episodes', []))
        self.action_source_overrides = self._validate_action_source_overrides(self.action_supervision_config.get('source_overrides'), episodes=self.episodes)
        self.episode_filter_config = self._validate_episode_filter(episode_filter)
        self.action_supervision = None
        gripper_supervision = self.action_supervision_config.get('gripper')
        if gripper_supervision:
            from sipai.data.transforms.state_action.gripper_supervision import GripperActionSupervision, GripperSupervisionSpec
            from sipai.data.transforms.state_action.spec import RepresentationSpec
            supervision_spec = GripperSupervisionSpec.from_config(gripper_supervision)
            action_spec = getattr(self.representation_transform, 'action_spec', None)
            if supervision_spec is not None:
                action_source = str((self.state_action_config.get('action') or {}).get('source', 'stored')).lower()
                if action_source != 'state_derivation':
                    raise ValueError('Gripper action supervision requires state-derived actions')
                if not isinstance(action_spec, RepresentationSpec):
                    raise ValueError('Gripper action supervision requires a typed action representation')
                self.action_supervision = GripperActionSupervision(supervision_spec, action_spec=action_spec, dataset_name=self.dataset_name, episodes=self.episodes)
        self.target_overrides = None
        target_override_config = self.action_supervision_config.get('target_overrides')
        if target_override_config is not None:
            from sipai.data.transforms.state_action.absolute_target_overrides import AbsoluteTargetOverrides
            action_config = self.state_action_config.get('action') or {}
            if action_config.get('source') != 'state_derivation':
                raise ValueError('Absolute target overrides require state-derived actions')
            action_spec = getattr(self.representation_transform, 'action_spec', None)
            if action_spec is None or any((field.source != 'state.values' for field in action_spec.fields)):
                raise ValueError('Absolute target overrides require state.values action fields')
            self.target_overrides = AbsoluteTargetOverrides(target_override_config, episodes=self.episodes)
        self.hold_current_state_supervision = None
        hold_current_state_config = self.action_supervision_config.get('hold_current_state')
        if hold_current_state_config is not None:
            raise ValueError('Standalone hold_current_state supervision is not part of this training recipe.')
        (self.episodes, self.episode_filter_summary) = self._apply_episode_filter(self.episodes)
        self._index_build_stats = {'compact_metadata_episodes': 0, 'batched_step_table_episodes': 0, 'fallback_scan_episodes': 0, 'stored_action_batched_episodes': 0, 'stored_action_fallback_episodes': 0}
        (self.episode_index, self.sample_cumsum) = self._build_compact_sample_index()
        self._counterfactual_action_samples: dict[int, dict[str, Any]] = {}
        self.counterfactual_action_pools: dict[str, dict[str, list[int]]] = {}
        self.counterfactual_action_pair_pools: dict[str, dict[str, dict[str, list[int]]]] = {}
        if self.counterfactual_action_supervision_config.get('enabled', False):
            self._build_counterfactual_action_index()
        if self.strict:
            self._validate_minimal_schema()

    @staticmethod
    def _episode_source_name(episode: Mapping[str, Any]) -> str:
        source_info = episode.get('source_info') or {}
        storage_extra = (episode.get('storage') or {}).get('extra') or {}
        return str(source_info.get('sampling_source') or storage_extra.get('source_name') or source_info.get('source_dataset') or source_info.get('raw_dataset') or '')

    def _validate_action_source_overrides(self, config: Any, *, episodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        if config is None:
            return {}
        if not isinstance(config, (list, tuple)) or not config:
            raise ValueError('action_supervision.source_overrides must be a non-empty sequence')
        action_config = self.state_action_config.get('action') or {}
        if str(action_config.get('source', 'stored')).lower() != 'state_derivation':
            raise ValueError('Action source overrides require state-derived default actions')
        action_spec = getattr(self.representation_transform, 'action_spec', None)
        fields = tuple(getattr(action_spec, 'fields', ()))
        raw_sources = tuple(dict.fromkeys((str(field.source) for field in fields)))
        if len(raw_sources) != 1:
            raise ValueError('Stored-command overrides require one shared raw action field source')
        raw_source = raw_sources[0]
        source_shapes = {tuple((int(item) for item in field.source_shape)) for field in fields if str(field.source) == raw_source}
        if len(source_shapes) != 1 or len(next(iter(source_shapes), ())) != 1:
            raise ValueError('Stored-command overrides require one-dimensional raw action rows')
        raw_dim = next(iter(source_shapes))[0]
        episode_counts: dict[str, int] = {}
        for episode in episodes:
            source_name = self._episode_source_name(episode)
            episode_counts[source_name] = episode_counts.get(source_name, 0) + 1
        normalized: dict[str, dict[str, Any]] = {}
        for (index, raw_override) in enumerate(config):
            if not isinstance(raw_override, Mapping):
                raise ValueError(f'action_supervision.source_overrides[{index}] must be a mapping')
            unknown = set(raw_override) - {'source_names', 'action_source', 'disable_gripper_compensation', 'expected_episodes'}
            if unknown:
                raise ValueError(f'Unsupported action source override key(s): {sorted(unknown)}')
            source_names = tuple((str(item).strip() for item in raw_override.get('source_names', ())))
            if not source_names or any((not item for item in source_names)) or len(set(source_names)) != len(source_names):
                raise ValueError('Action source override source_names must be unique and non-empty')
            if str(raw_override.get('action_source', '')).lower() != 'stored':
                raise ValueError('Action source overrides currently only support stored actions')
            if raw_override.get('disable_gripper_compensation') is not True:
                raise ValueError('Stored-command overrides must explicitly disable gripper compensation')
            expected = raw_override.get('expected_episodes')
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
                raise ValueError('Action source override expected_episodes must be an integer >= 1')
            actual = sum((episode_counts.get(source_name, 0) for source_name in source_names))
            if actual != expected:
                raise ValueError(f'Action source override episode count mismatch: sources={list(source_names)!r}, expected={expected}, actual={actual}')
            for source_name in source_names:
                if source_name in normalized:
                    raise ValueError(f'Duplicate action source override: {source_name!r}')
                normalized[source_name] = {'action_source': 'stored', 'disable_gripper_compensation': True, 'raw_source': raw_source, 'raw_dim': raw_dim}
        for episode in episodes:
            source_name = self._episode_source_name(episode)
            if source_name not in normalized:
                continue
            action_stream = ((episode.get('storage') or {}).get('streams') or {}).get('action')
            action_spec = action_stream.get('action_spec') if isinstance(action_stream, Mapping) else None
            if not isinstance(action_spec, Mapping) or action_spec.get('available') != 'stored':
                raise ValueError(f'Stored-command source {source_name!r} has an episode without stored actions')
            shape = action_spec.get('shape')
            if shape != [raw_dim]:
                raise ValueError(f'Stored-command source {source_name!r} action shape must be [{raw_dim}], got {shape!r}')
        return normalized

    def _action_source_for_episode(self, episode: Mapping[str, Any]) -> str:
        override = self.action_source_overrides.get(self._episode_source_name(episode))
        if override is not None:
            return str(override['action_source'])
        return str((self.state_action_config.get('action') or {}).get('source', 'stored')).lower()

    @staticmethod
    def _validate_episode_filter(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        value = dict(config or {})
        unknown = set(value) - {'rules', 'exclusions'}
        if unknown:
            raise ValueError(f'Unsupported episode_filter key(s): {sorted(unknown)}')
        rules = value.get('rules') or ()
        if isinstance(rules, (str, bytes)) or not isinstance(rules, (list, tuple)):
            raise ValueError('episode_filter.rules must be a sequence')
        normalized: list[dict[str, Any]] = []
        for (index, rule) in enumerate(rules):
            if not isinstance(rule, Mapping):
                raise ValueError(f'episode_filter.rules[{index}] must be a mapping')
            unknown_rule_keys = set(rule) - {'task_instructions', 'task_instruction_contains', 'required_episode_flag', 'episode_audit_path'}
            if unknown_rule_keys:
                raise ValueError(f'Unsupported episode_filter.rules[{index}] key(s): {sorted(unknown_rule_keys)}')
            instructions = rule.get('task_instructions') or ()
            if isinstance(instructions, (str, bytes)) or not isinstance(instructions, (list, tuple)):
                raise ValueError(f'episode_filter.rules[{index}].task_instructions must be a sequence')
            instructions = tuple((str(item).strip() for item in instructions))
            if any((not item for item in instructions)) or len(set(instructions)) != len(instructions):
                raise ValueError(f'episode_filter.rules[{index}].task_instructions must be unique and non-empty')
            contains = rule.get('task_instruction_contains') or ()
            if isinstance(contains, (str, bytes)) or not isinstance(contains, (list, tuple)):
                raise ValueError(f'episode_filter.rules[{index}].task_instruction_contains must be a sequence')
            contains = tuple((str(item).strip().casefold() for item in contains))
            if any((not item for item in contains)) or len(set(contains)) != len(contains):
                raise ValueError(f'episode_filter.rules[{index}].task_instruction_contains must be unique and non-empty')
            if not instructions and (not contains):
                raise ValueError(f'episode_filter.rules[{index}] requires task_instructions or task_instruction_contains')
            required_flag = str(rule.get('required_episode_flag', '')).strip() or None
            audit_path_value = str(rule.get('episode_audit_path', '')).strip()
            audit_path = Path(audit_path_value).expanduser().resolve(strict=False) if audit_path_value else None
            if required_flag is None and audit_path is None:
                raise ValueError(f'episode_filter.rules[{index}] requires required_episode_flag or episode_audit_path')
            audited_episodes: dict[str, int] = {}
            if audit_path is not None:
                try:
                    audit = json.loads(audit_path.read_text(encoding='utf-8'))
                except FileNotFoundError as exc:
                    raise FileNotFoundError(f'Episode audit does not exist: {audit_path}') from exc
                if not isinstance(audit, Mapping) or audit.get('schema_version') != 'conveyor-eval-right-only-audit-v1':
                    raise ValueError(f'Unsupported episode audit artifact: {audit_path}')
                records = audit.get('episodes')
                if not isinstance(records, list) or not records:
                    raise ValueError(f'Episode audit contains no records: {audit_path}')
                for record in records:
                    if not isinstance(record, Mapping) or record.get('passed') is not True:
                        continue
                    episode_id = str(record.get('episode_id', '')).strip()
                    num_steps = int(record.get('num_steps', 0))
                    if not episode_id or num_steps <= 0 or episode_id in audited_episodes:
                        raise ValueError(f'Episode audit has invalid passed records: {audit_path}')
                    audited_episodes[episode_id] = num_steps
                expected_passed = int((audit.get('summary') or {}).get('passed', -1))
                if expected_passed != len(audited_episodes):
                    raise ValueError(f'Episode audit passed-count mismatch: {audit_path}')
            normalized.append({'task_instructions': instructions, 'task_instruction_contains': contains, 'required_episode_flag': required_flag, 'episode_audit_path': str(audit_path) if audit_path is not None else None, 'audited_episodes': audited_episodes})
        exclusions = value.get('exclusions') or ()
        if isinstance(exclusions, (str, bytes)) or not isinstance(exclusions, (list, tuple)):
            raise ValueError('episode_filter.exclusions must be a sequence')
        normalized_exclusions: list[dict[str, Any]] = []
        for (index, exclusion) in enumerate(exclusions):
            if not isinstance(exclusion, Mapping):
                raise ValueError(f'episode_filter.exclusions[{index}] must be a mapping')
            unknown_exclusion_keys = set(exclusion) - {'source_names', 'task_instructions', 'expected_episodes'}
            if unknown_exclusion_keys:
                raise ValueError(f'Unsupported episode_filter.exclusions[{index}] key(s): {sorted(unknown_exclusion_keys)}')
            source_names = exclusion.get('source_names') or ()
            if isinstance(source_names, (str, bytes)) or not isinstance(source_names, (list, tuple)):
                raise ValueError(f'episode_filter.exclusions[{index}].source_names must be a sequence')
            source_names = tuple((str(item).strip() for item in source_names))
            if not source_names or any((not item for item in source_names)) or len(set(source_names)) != len(source_names):
                raise ValueError(f'episode_filter.exclusions[{index}].source_names must be unique and non-empty')
            instructions = exclusion.get('task_instructions') or ()
            if isinstance(instructions, (str, bytes)) or not isinstance(instructions, (list, tuple)):
                raise ValueError(f'episode_filter.exclusions[{index}].task_instructions must be a sequence')
            instructions = tuple((str(item).strip() for item in instructions))
            if not instructions or any((not item for item in instructions)) or len(set(instructions)) != len(instructions):
                raise ValueError(f'episode_filter.exclusions[{index}].task_instructions must be unique and non-empty')
            expected_episodes = exclusion.get('expected_episodes')
            if isinstance(expected_episodes, bool) or not isinstance(expected_episodes, int) or expected_episodes < 0:
                raise ValueError(f'episode_filter.exclusions[{index}].expected_episodes must be an integer >= 0')
            normalized_exclusions.append({'source_names': source_names, 'task_instructions': instructions, 'expected_episodes': expected_episodes})
        return {'rules': tuple(normalized), 'exclusions': tuple(normalized_exclusions)}

    def _apply_episode_filter(self, episodes: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], dict[str, Any]]:
        rules = self.episode_filter_config['rules']
        exclusions = self.episode_filter_config['exclusions']
        if not rules and (not exclusions):
            return (episodes, {'original_episodes': len(episodes), 'retained_episodes': len(episodes), 'excluded_episodes': 0, 'excluded_by_rule': {}})
        exclusion_matches: dict[int, int] = {}
        exclusion_counts = [0] * len(exclusions)
        for (episode_position, episode) in enumerate(episodes):
            instruction = str((episode.get('task') or {}).get('instruction', '')).strip()
            source_name = self._episode_source_name(episode)
            matching_exclusions = [index for (index, exclusion) in enumerate(exclusions) if source_name in exclusion['source_names'] and instruction in exclusion['task_instructions']]
            if len(matching_exclusions) > 1:
                raise ValueError(f"Episode matches multiple episode_filter exclusions: episode={episode.get('episode_id')!r}, rules={matching_exclusions!r}")
            if matching_exclusions:
                exclusion_index = matching_exclusions[0]
                exclusion_matches[episode_position] = exclusion_index
                exclusion_counts[exclusion_index] += 1
        for (index, (exclusion, actual)) in enumerate(zip(exclusions, exclusion_counts, strict=True)):
            expected = exclusion['expected_episodes']
            if actual != expected:
                raise ValueError(f"episode_filter exclusion count mismatch: rule={index}, sources={list(exclusion['source_names'])!r}, instructions={list(exclusion['task_instructions'])!r}, expected={expected}, actual={actual}")
        retained: list[Dict[str, Any]] = []
        excluded_by_rule: dict[str, int] = {}
        excluded_ids: list[str] = []
        for (episode_position, episode) in enumerate(episodes):
            instruction = str((episode.get('task') or {}).get('instruction', '')).strip()
            normalized_instruction = instruction.casefold()
            rejected_flag: str | None = None
            for rule in rules:
                matches_exact = instruction in rule['task_instructions']
                matches_fragment = any((fragment in normalized_instruction for fragment in rule['task_instruction_contains']))
                if not matches_exact and (not matches_fragment):
                    continue
                required_flag = rule['required_episode_flag']
                passed = False
                if required_flag is not None:
                    eligibility = (episode.get('extra') or {}).get(str(required_flag))
                    if isinstance(eligibility, Mapping):
                        passed = eligibility.get('passed') is True
                    else:
                        passed = eligibility is True
                audited_steps = rule['audited_episodes'].get(str(episode.get('episode_id', '')))
                if audited_steps is not None:
                    actual_steps = self._num_steps(episode)
                    if actual_steps != audited_steps:
                        raise ValueError(f"Episode audit length mismatch for {episode.get('episode_id')!r}: expected={audited_steps}, actual={actual_steps}")
                    passed = True
                if not passed:
                    rejected_flag = str(required_flag or rule['episode_audit_path'] or 'episode_audit')
                    break
            exclusion_index = exclusion_matches.get(episode_position)
            if rejected_flag is None and exclusion_index is not None:
                exclusion = exclusions[exclusion_index]
                rejected_flag = f'explicit_exclusion[{exclusion_index}]'
            if rejected_flag is None:
                retained.append(episode)
                continue
            excluded_by_rule[rejected_flag] = excluded_by_rule.get(rejected_flag, 0) + 1
            excluded_ids.append(str(episode.get('episode_id', '')))
        if not retained:
            raise ValueError('episode_filter excluded every episode')
        summary = {'original_episodes': len(episodes), 'retained_episodes': len(retained), 'excluded_episodes': len(excluded_ids), 'excluded_by_rule': excluded_by_rule, 'excluded_episode_ids': tuple(excluded_ids)}
        logger.info('Applied episode filter: retained=%d excluded=%d rules=%s', len(retained), len(excluded_ids), excluded_by_rule)
        return (retained, summary)

    def _validate_state_recovery_supervision(self, config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        value = dict(config or {})
        defaults = {'enabled': False, 'tasks': (), 'required_episode_flag': None, 'start_step': 0, 'end_step': -1, 'probability': 0.0, 'value_source': 'state.values', 'value_indices': (), 'perturbation_std': 0.0, 'perturbation_clip': 0.0, 'seed': 0}
        unknown = set(value) - set(defaults)
        if unknown:
            raise ValueError(f'Unsupported state_recovery_supervision key(s): {sorted(unknown)}')
        result = {**defaults, **value}
        result['enabled'] = bool(result['enabled'])
        if not result['enabled']:
            return result
        tasks = tuple((str(task) for task in result['tasks']))
        if not tasks or len(set(tasks)) != len(tasks) or any((not task for task in tasks)):
            raise ValueError('state_recovery_supervision.tasks must be a non-empty unique sequence')
        result['tasks'] = tasks
        required_episode_flag = result['required_episode_flag']
        if required_episode_flag is not None:
            required_episode_flag = str(required_episode_flag).strip()
            if not required_episode_flag:
                raise ValueError('state_recovery_supervision.required_episode_flag must be non-empty when set')
        result['required_episode_flag'] = required_episode_flag
        result['start_step'] = int(result['start_step'])
        result['end_step'] = int(result['end_step'])
        if result['start_step'] < 0 or result['end_step'] < result['start_step']:
            raise ValueError('state_recovery_supervision requires 0 <= start_step <= end_step')
        probability = float(result['probability'])
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError('state_recovery_supervision.probability must lie in (0, 1]')
        result['probability'] = probability
        result['value_source'] = str(result['value_source'])
        if not result['value_source']:
            raise ValueError('state_recovery_supervision.value_source must be non-empty')
        indices = tuple((int(index) for index in result['value_indices']))
        if not indices or len(set(indices)) != len(indices) or any((index < 0 for index in indices)):
            raise ValueError('state_recovery_supervision.value_indices must be non-empty, unique, and non-negative')
        result['value_indices'] = indices
        result['perturbation_std'] = float(result['perturbation_std'])
        result['perturbation_clip'] = float(result['perturbation_clip'])
        if not math.isfinite(result['perturbation_std']) or not math.isfinite(result['perturbation_clip']) or result['perturbation_std'] <= 0.0 or (result['perturbation_clip'] <= 0.0):
            raise ValueError('state_recovery_supervision perturbation_std/perturbation_clip must be positive and finite')
        result['seed'] = int(result['seed'])
        state_representation = str((self.state_action_config.get('state') or {}).get('representation', 'absolute')).lower()
        action_representation = str((self.state_action_config.get('action') or {}).get('representation', 'absolute')).lower()
        action_source = str((self.state_action_config.get('action') or {}).get('source', 'stored')).lower()
        if state_representation != 'absolute' or action_representation != 'relative':
            raise ValueError('state_recovery_supervision requires absolute state and relative action representations')
        if action_source != 'state_derivation':
            raise ValueError('state_recovery_supervision requires state-derived actions')
        return result

    def _apply_state_recovery_supervision(self, episode: Mapping[str, Any], sample_ref: SampleIndex, raw_anchor: Mapping[str, Any], *, counterfactual_action: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
        config = self.state_recovery_supervision_config
        episode_extra = episode.get('extra') or {}
        required_episode_flag = config.get('required_episode_flag')
        eligibility = episode_extra.get(str(required_episode_flag)) if required_episode_flag else True
        if isinstance(eligibility, Mapping):
            eligibility = eligibility.get('passed') is True
        elif required_episode_flag:
            eligibility = eligibility is True
        if not bool(config['enabled']) or self.statistics_state_action_only or counterfactual_action is not None or (not int(config['start_step']) <= sample_ref.step_index <= int(config['end_step'])) or (str(episode_extra.get('memory_task', '')) not in config['tasks']) or (not bool(eligibility)):
            return (raw_anchor, None)
        episode_id = str(episode.get('episode_id', ''))
        digest = hashlib.sha256(f"{int(config['seed'])}:{episode_id}:{sample_ref.step_index}".encode('utf-8')).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], byteorder='little', signed=False))
        if float(rng.random()) >= float(config['probability']):
            return (raw_anchor, None)
        perturbed_anchor = copy.deepcopy(raw_anchor)
        raw_values = read_dotted_source(perturbed_anchor, str(config['value_source']))
        values = np.asarray(raw_values)
        if values.ndim != 1:
            raise ValueError('state_recovery_supervision expects value_source to resolve to a one-dimensional vector')
        indices = tuple((int(index) for index in config['value_indices']))
        if max(indices) >= values.shape[0]:
            raise ValueError('state_recovery_supervision.value_indices exceed raw_anchor.values dimension')
        delta = rng.normal(0.0, float(config['perturbation_std']), size=len(indices))
        delta = np.clip(delta, -float(config['perturbation_clip']), float(config['perturbation_clip']))
        perturbed_values = np.array(values, copy=True)
        perturbed_values[np.asarray(indices, dtype=np.int64)] += delta
        if isinstance(raw_values, np.ndarray):
            raw_values[...] = perturbed_values.astype(raw_values.dtype, copy=False)
        elif isinstance(raw_values, list):
            raw_values[:] = perturbed_values.tolist()
        else:
            raise ValueError('state_recovery_supervision value_source must resolve to a mutable list or ndarray')
        return (perturbed_anchor, {'state_recovery_supervision': True, 'state_recovery_value_indices': list(indices), 'state_recovery_value_source': str(config['value_source']), 'state_recovery_required_episode_flag': required_episode_flag, 'state_recovery_delta': delta.tolist(), 'state_recovery_target': 'recorded_expert_trajectory'})

    def __len__(self) -> int:
        return self.sample_cumsum[-1] if self.sample_cumsum else 0

    def _build_counterfactual_action_index(self) -> None:
        config = self.counterfactual_action_supervision_config
        unknown = set(config) - {'enabled', 'tasks', 'history_camera', 'supervised_action_steps'}
        if unknown:
            raise ValueError(f'Unsupported counterfactual_action_supervision key(s): {sorted(unknown)}')
        tasks = tuple((str(task) for task in config.get('tasks') or ()))
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError('counterfactual_action_supervision.tasks must be a non-empty unique sequence')
        if self.representation_transform is None or str((self.state_action_config.get('action') or {}).get('source', 'stored')).lower() != 'state_derivation':
            raise ValueError('Counterfactual hold actions require state-derived action representation')
        history_camera = str(config.get('history_camera', self.camera_history_name or ''))
        if history_camera != self.camera_history_name:
            raise ValueError('Counterfactual history camera must match camera_history_name')
        supervised_steps = int(config.get('supervised_action_steps', 0))
        horizon = int(self.state_action_config.get('horizon', 0))
        if supervised_steps < 1 or supervised_steps > horizon:
            raise ValueError('Counterfactual supervised_action_steps must lie inside the action horizon')
        episode_positions = {str(episode.get('episode_id', '')): index for (index, episode) in enumerate(self.episodes)}
        if len(episode_positions) != len(self.episodes):
            raise ValueError('Counterfactual actions require unique non-empty episode IDs')
        base_length = len(self)
        next_virtual_index = base_length
        for (compact_pos, compact) in enumerate(self.episode_index):
            episode = self.episodes[int(compact.episode_index)]
            task = str((episode.get('extra') or {}).get('memory_task', ''))
            events = (episode.get('extra') or {}).get('memory_events') or []
            specs = [event for event in events if isinstance(event.get('counterfactual_action'), Mapping)]
            if task not in tasks:
                if specs:
                    raise ValueError(f'Counterfactual action unexpectedly declared for disabled task {task!r}')
                continue
            if not specs:
                raise ValueError(f'Enabled counterfactual-action task {task!r} has an episode without a contract')
            local_indices = compact.sample_local_indices if compact.sample_local_indices is not None else tuple(range(compact.num_samples))
            anchor_stride = int((self.state_action_config.get('sampling') or {}).get('anchor_stride', 1))
            step_to_global = {int(local_index) * anchor_stride: int(self.sample_cumsum[compact_pos]) + position for (position, local_index) in enumerate(local_indices)}
            task_pool = self.counterfactual_action_pools.setdefault(task, {})
            task_pair_pool = self.counterfactual_action_pair_pools.setdefault(task, {})
            for event in specs:
                contract = dict(event['counterfactual_action'])
                schema_version = int(contract.get('schema_version', -1))
                if schema_version not in {1, 2, 4}:
                    raise ValueError('Unsupported counterfactual-action schema')
                if contract.get('task') != task:
                    raise ValueError(f'Invalid counterfactual-action contract for task {task!r}')
                if contract.get('history_camera') != history_camera:
                    raise ValueError('Counterfactual-action history camera mismatch')
                if int(contract.get('supervised_action_steps', -1)) != supervised_steps:
                    raise ValueError('Counterfactual-action supervised prefix mismatch')
                recipient_target = str(contract.get('recipient_target', ''))
                recipient_episode_id = str(episode.get('episode_id', ''))

                def replacements_for(anchor_step: int, *, opening_history_slots: int | None=None) -> list[dict[str, int]]:
                    if schema_version == 1:
                        raw = contract.get('history_replacements')
                        if not isinstance(raw, list) or not raw:
                            raise ValueError('Counterfactual-action contract has no history replacements')
                        replacements = [dict(item) for item in raw]
                    else:
                        context = contract.get('opening_context_steps')
                        if not isinstance(context, list) or len(context) != 2:
                            raise ValueError('Counterfactual-action v2 has no opening context')
                        (context_start, context_end) = map(int, context)
                        replacements = [{'slot_index': slot, 'history_offset': int(offset), 'recipient_step': int(anchor_step) + int(offset), 'donor_step': int(anchor_step) + int(offset)} for (slot, offset) in enumerate(self.camera_history_frame_offsets or ()) if context_start <= int(anchor_step) + int(offset) <= context_end]
                        expected_slots = int(contract.get('opening_history_slots', -1)) if opening_history_slots is None else int(opening_history_slots)
                        if len(replacements) != expected_slots:
                            raise ValueError('Counterfactual-action anchor lacks its declared opening slots')
                    for replacement in replacements:
                        slot = int(replacement['slot_index'])
                        if slot < 0 or slot >= len(self.camera_history_frame_offsets or ()):
                            raise ValueError('Counterfactual history replacement slot is outside configured history')
                        expected_offset = int((self.camera_history_frame_offsets or ())[slot])
                        if int(replacement['history_offset']) != expected_offset or int(replacement['recipient_step']) != anchor_step + expected_offset or int(replacement['donor_step']) < 0:
                            raise ValueError('Counterfactual history replacement does not match configured offsets')
                    return replacements

                def add_virtual(*, anchor_step: int, donor: Mapping[str, Any], sample_kind: str, action_mode: str, stage: str | None=None, pair_id: str | None=None, common_suffix_action_mode: str | None=None, opening_history_slots: int | None=None) -> int:
                    nonlocal next_virtual_index
                    if anchor_step not in step_to_global:
                        raise ValueError(f'Counterfactual-action anchor {anchor_step} is not an indexed sample in {recipient_episode_id!r}')
                    donor_id = str(donor.get('episode_id', ''))
                    donor_target = str(donor.get('target', ''))
                    use_recipient_history = bool(donor.get('use_recipient_history', False))
                    if donor_id not in episode_positions or not donor_target or (donor_id == recipient_episode_id and (not use_recipient_history)) or (donor_id != recipient_episode_id and use_recipient_history):
                        raise ValueError('Counterfactual donor must be distinct unless it explicitly uses recipient history')
                    donor_episode_index = episode_positions[donor_id]
                    donor_episode = self.episodes[donor_episode_index]
                    if str((donor_episode.get('extra') or {}).get('memory_task', '')) != task:
                        raise ValueError('Counterfactual donor crosses memory tasks')
                    virtual_index = next_virtual_index
                    next_virtual_index += 1
                    self._counterfactual_action_samples[virtual_index] = {'recipient_index': step_to_global[anchor_step], 'recipient_episode_index': int(compact.episode_index), 'recipient_episode_id': recipient_episode_id, 'recipient_target': recipient_target, 'anchor_step': int(anchor_step), 'donor_episode_index': donor_episode_index, 'donor_episode_id': donor_id, 'donor_target': donor_target, 'history_camera': history_camera, 'history_replacements': replacements_for(anchor_step, opening_history_slots=opening_history_slots), 'sample_kind': sample_kind, 'action_mode': action_mode, 'supervised_action_steps': supervised_steps, 'stage': stage, 'pair_id': pair_id, 'common_suffix_action_mode': common_suffix_action_mode, 'pairing_mode': contract.get('pairing_mode'), 'sampling_hierarchy': (recipient_target, recipient_episode_id, donor_target, donor_id, str(anchor_step))}
                    task_pool.setdefault(sample_kind, []).append(virtual_index)
                    return virtual_index
                if schema_version == 1:
                    if contract.get('action_mode') != 'hold_current_state':
                        raise ValueError(f'Invalid v1 counterfactual-action contract for task {task!r}')
                    anchor_step = int(contract['anchor_step'])
                    if anchor_step not in step_to_global:
                        raise ValueError(f'Counterfactual-action anchor {anchor_step} is not indexed')
                    task_pool.setdefault('positive', []).append(step_to_global[anchor_step])
                    seen_donor_targets = set()
                    for donor in contract.get('donors') or ():
                        donor_target = str(donor.get('target', ''))
                        if donor_target == recipient_target or donor_target in seen_donor_targets:
                            raise ValueError('Counterfactual v1 donor must have a unique alternate target')
                        add_virtual(anchor_step=anchor_step, donor=donor, sample_kind='synthetic_hold', action_mode='hold_current_state')
                        seen_donor_targets.add(donor_target)
                    continue
                positive_donors = contract.get('positive_donors') or ()
                negative_donors = contract.get('negative_donors') or ()
                decision_anchors = tuple(map(int, contract.get('decision_anchor_steps') or ()))
                if not positive_donors or bool(decision_anchors) != bool(negative_donors):
                    raise ValueError('Counterfactual-action decisions need positive and negative donors together')
                decision_common_suffix_action_mode = str(contract.get('decision_common_suffix_action_mode', ''))
                if decision_common_suffix_action_mode != 'hold_current_state':
                    raise ValueError('Counterfactual-action v2 decision pairs require a common hold suffix')
                for anchor_step in decision_anchors:
                    pair_id = f'{recipient_episode_id}:decision:{anchor_step}'
                    pair_group = task_pair_pool.setdefault(pair_id, {'decision_positive': [], 'decision_negative': []})
                    for donor in positive_donors:
                        pair_group['decision_positive'].append(add_virtual(anchor_step=anchor_step, donor=donor, sample_kind='decision_positive', action_mode=str(contract['decision_positive_action_mode']), pair_id=pair_id, common_suffix_action_mode=decision_common_suffix_action_mode))
                    for donor in negative_donors:
                        pair_group['decision_negative'].append(add_virtual(anchor_step=anchor_step, donor=donor, sample_kind='decision_negative', action_mode=str(contract['decision_negative_action_mode']), pair_id=pair_id, common_suffix_action_mode=decision_common_suffix_action_mode))
                for rollout in contract.get('positive_rollout_anchors') or ():
                    anchor_step = int(rollout['anchor_step'])
                    stage = str(rollout['stage'])
                    sample_kind = f'positive_rollout_{stage}'
                    for donor in positive_donors:
                        add_virtual(anchor_step=anchor_step, donor=donor, sample_kind=sample_kind, action_mode=str(contract['positive_rollout_action_mode']), stage=stage, opening_history_slots=int(rollout['opening_history_slots']))
                for (field, sample_kind) in (('positive_timing_wait_anchors', 'positive_timing_wait'), ('negative_distractor_wait_anchors', 'negative_distractor_wait')):
                    for wait_anchor in contract.get(field) or ():
                        anchor_step = int(wait_anchor['anchor_step'])
                        for donor in positive_donors:
                            add_virtual(anchor_step=anchor_step, donor=donor, sample_kind=sample_kind, action_mode=str(contract['positive_timing_wait_action_mode' if sample_kind == 'positive_timing_wait' else 'negative_distractor_wait_action_mode']), stage='timing_wait' if sample_kind == 'positive_timing_wait' else 'distractor_wait', common_suffix_action_mode=decision_common_suffix_action_mode, opening_history_slots=int(wait_anchor['opening_history_slots']))
        missing = set(tasks) - set(self.counterfactual_action_pools)
        if missing:
            raise ValueError(f'Counterfactual-action tasks have no virtual samples: {sorted(missing)}')
        for (task, pool) in self.counterfactual_action_pools.items():
            if not pool or any((not indices for indices in pool.values())):
                raise ValueError(f'Counterfactual-action pool {task!r} is incomplete')
            v2_variants = {'decision_positive', 'decision_negative', 'positive_rollout_approach', 'positive_rollout_closing', 'positive_rollout_secure_lift'}
            v4_wait_variants = {'positive_timing_wait', 'negative_distractor_wait'}
            expected_variants = v2_variants | v4_wait_variants
            if v4_wait_variants.intersection(pool) and set(pool) != expected_variants:
                raise ValueError(f'Counterfactual-action v4 pool {task!r} lacks required variants')
            if not v4_wait_variants.intersection(pool) and v2_variants.intersection(pool) and (set(pool) != v2_variants):
                raise ValueError(f'Counterfactual-action v2 pool {task!r} lacks required variants')
            for (pair_id, members) in self.counterfactual_action_pair_pools.get(task, {}).items():
                if any((not members.get(kind) for kind in ('decision_positive', 'decision_negative'))):
                    raise ValueError(f'Counterfactual action pair {pair_id!r} is incomplete')

    def is_counterfactual_action_index(self, idx: int) -> bool:
        return int(idx) in self._counterfactual_action_samples

    def counterfactual_action_spec(self, idx: int) -> Mapping[str, Any] | None:
        return self._counterfactual_action_samples.get(int(idx))

    def base_sample_index(self, idx: int) -> int:
        spec = self.counterfactual_action_spec(idx)
        return int(spec['recipient_index']) if spec is not None else int(idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._getitem_single(int(idx))

    def __getitems__(self, indices: List[int]) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        prefetch_before = dict(self._batch_prefetch_stats)
        if self.batch_prefetch and len(indices) > 1:
            try:
                self._prefetch_batch(indices)
            except Exception:
                self._batch_prefetch_stats['errors'] += 1
                if not self.skip_bad_samples:
                    raise
                logger.warning('Batch prefetch failed; falling back to per-sample load', exc_info=False)
        items = [self._getitem_single(int(idx)) for idx in indices]
        flush = getattr(self.reader, '_write_metrics_file', None)
        if callable(flush):
            flush()
        if self.statistics_state_action_only:
            try:
                from torch.utils.data import get_worker_info
                worker_info = get_worker_info()
                worker_id = worker_info.id if worker_info is not None else None
            except Exception:
                worker_id = None
            return StatisticsMaterializedBlock(items, trace={'pid': os.getpid(), 'worker_id': worker_id, 'indices': tuple((int(index) for index in indices)), 'block_size': len(indices), 'materialize_seconds': time.perf_counter() - started, 'used_getitems': True, 'prefetch': {key: int(value) - int(prefetch_before.get(key, 0)) for (key, value) in self._batch_prefetch_stats.items()}})
        return items

    def _prefetch_batch(self, indices: List[int]) -> None:
        self._batch_prefetch_stats['batches'] += 1
        refs_by_episode: Dict[int, List[SampleIndex]] = {}
        for idx in indices:
            sample_ref = self._sample_ref_from_index(int(idx))
            refs_by_episode.setdefault(sample_ref.episode_index, []).append(sample_ref)
        for (episode_index, sample_refs) in refs_by_episode.items():
            episode = self.episodes[int(episode_index)]
            self._prefetch_episode_batch(episode, sample_refs)
        if self.batch_prefetch_cameras:
            donor_frames: dict[tuple[int, str], list[int]] = {}
            for idx in indices:
                spec = self.counterfactual_action_spec(int(idx))
                if spec is None:
                    continue
                donor_episode_index = int(spec['donor_episode_index'])
                camera = str(spec['history_camera'])
                donor_episode = self.episodes[donor_episode_index]
                key = (donor_episode_index, camera)
                for replacement in spec['history_replacements']:
                    donor_step = self._materialize_step(donor_episode, int(replacement['donor_step']))
                    donor_ref = donor_step.get('observation', {}).get('cameras', {}).get(camera)
                    if self._is_simple_index_ref(donor_ref):
                        donor_frames.setdefault(key, []).append(int(donor_ref['index']))
            for ((episode_index, camera), frame_indices) in donor_frames.items():
                unique_indices = list(dict.fromkeys(frame_indices))
                self.reader.read_many(self.episodes[episode_index], f'camera.{camera}', indices=unique_indices)
                self._batch_prefetch_stats['camera_stream_reads'] += 1
                self._batch_prefetch_stats['camera_frames'] += len(unique_indices)

    def _prefetch_episode_batch(self, episode: Mapping[str, Any], sample_refs: List[SampleIndex]) -> None:
        step_indices: list[int] = []
        for sample_ref in sample_refs:
            step_indices.extend(range(sample_ref.step_index, sample_ref.step_index + sample_ref.window_size))
        if not step_indices:
            return
        seen_steps = sorted(set(step_indices))
        steps = []
        for step_index in seen_steps:
            steps.append((step_index, self._materialize_step(episode, step_index)))
        self._batch_prefetch_stats['step_refs'] += len(seen_steps)
        steps_by_index = {step_index: step for (step_index, step) in steps}
        self._prefetch_episode_batch_actions(episode, sample_refs, steps_by_index)
        if self.batch_prefetch_cameras:
            self._prefetch_episode_batch_cameras(episode, sample_refs, steps_by_index)
        if not self.batch_state_window:
            return
        allowed_state_streams = None
        if self.representation_transform is not None:
            allowed_state_streams = {self._source_stream_key(episode, source) for source in self._representation_anchor_sources()}
        grouped: Dict[str, list[int]] = {}
        for (_, step) in steps:
            state_ref = step.get('observation', {}).get('state')
            if not isinstance(state_ref, Mapping) or 'streams' not in state_ref:
                continue
            for (stream_key, ref) in (state_ref.get('streams') or {}).items():
                if allowed_state_streams is not None and stream_key not in allowed_state_streams:
                    continue
                if self._is_simple_index_ref(ref):
                    grouped.setdefault(str(stream_key), []).append(int(ref['index']))
        for (stream_key, stream_indices) in grouped.items():
            unique_indices = list(dict.fromkeys(stream_indices))
            if not unique_indices:
                continue
            self.reader.read_many(episode, stream_key, indices=unique_indices)
            self._batch_prefetch_stats['state_stream_reads'] += 1
            self._batch_prefetch_stats['state_rows'] += len(unique_indices)

    def _prefetch_episode_batch_actions(self, episode: Mapping[str, Any], sample_refs: List[SampleIndex], steps_by_index: Mapping[int, Mapping[str, Any]]) -> None:
        source = self._action_source_for_episode(episode)
        if self.representation_transform is not None and source != 'stored':
            return
        if self.representation_transform is None and str(self.action_config.get('source', 'stored')).lower() != 'stored':
            return
        horizon = self._stored_action_horizon()
        interval = self._stored_action_step_interval()
        read_horizon = horizon + 1 if self._stored_transform is not None else horizon
        requested: list[int] = []
        for sample_ref in sample_refs:
            step = steps_by_index.get(sample_ref.step_index)
            action_ref = step.get('action') if isinstance(step, Mapping) else None
            if not isinstance(action_ref, Mapping) or action_ref.get('index') is None:
                continue
            start = int(action_ref['index'])
            requested.extend((start + offset * interval for offset in range(read_horizon)))
        unique_indices = list(dict.fromkeys(requested))
        if not unique_indices:
            return
        self._read_many_with_context(episode, 'action', sample_ref=sample_refs[0] if sample_refs else None, indices=unique_indices)
        self._batch_prefetch_stats['action_stream_reads'] += 1
        self._batch_prefetch_stats['action_rows'] += len(unique_indices)

    def _prefetch_episode_batch_cameras(self, episode: Mapping[str, Any], sample_refs: List[SampleIndex], steps_by_index: Mapping[int, Mapping[str, Any]]) -> None:
        grouped: Dict[str, list[int]] = {}
        for sample_ref in sample_refs:
            offsets = self._camera_window_offsets(sample_ref.window_size)
            for offset in offsets:
                step = steps_by_index.get(sample_ref.step_index + offset)
                if not isinstance(step, Mapping):
                    continue
                cameras = step.get('observation', {}).get('cameras', {})
                if not isinstance(cameras, Mapping):
                    continue
                for (camera_name, ref) in cameras.items():
                    if self._is_simple_index_ref(ref):
                        grouped.setdefault(f'camera.{camera_name}', []).append(int(ref['index']))
            if self.camera_history_name is not None:
                for history_offset in self.camera_history_frame_offsets or ():
                    history_step_index = sample_ref.step_index + history_offset
                    if history_step_index < 0:
                        continue
                    history_step = steps_by_index.get(history_step_index)
                    if history_step is None:
                        history_step = self._materialize_step(episode, history_step_index)
                    history_ref = history_step.get('observation', {}).get('cameras', {}).get(self.camera_history_name)
                    if self._is_simple_index_ref(history_ref):
                        grouped.setdefault(f'camera.{self.camera_history_name}', []).append(int(history_ref['index']))
        for (stream_key, frame_indices) in grouped.items():
            unique_indices = list(dict.fromkeys(frame_indices))
            if not unique_indices:
                continue
            self.reader.read_many(episode, stream_key, indices=unique_indices)
            self._batch_prefetch_stats['camera_stream_reads'] += 1
            self._batch_prefetch_stats['camera_frames'] += len(unique_indices)

    def _is_simple_index_ref(self, ref: Any) -> bool:
        if not isinstance(ref, Mapping) or ref.get('index') is None:
            return False
        if any((ref.get(key) is not None for key in ('timestamp', 'timestamp_from', 'index_range', 'time_range'))):
            return False
        if bool(ref.get('interpolate', False)):
            return False
        return True

    def _getitem_single(self, idx: int) -> Dict[str, Any]:
        if not self.skip_bad_samples:
            return self._load_sample(idx)
        last_error: BaseException | None = None
        dataset_len = len(self)
        for attempt in range(self.max_skip_attempts + 1):
            candidate_idx = (idx + attempt) % dataset_len
            try:
                sample = self._load_sample(candidate_idx)
            except Exception as exc:
                last_error = exc
                context = self._error_context(candidate_idx, exc)
                logger.warning('Skipping bad canonical sample and trying next sample: %s', self._format_error_context(context), exc_info=False)
                continue
            if attempt > 0:
                logger.warning('Substituted canonical sample global_idx=%s with global_idx=%s after %s skipped sample(s)', idx, candidate_idx, attempt)
            return sample
        raise RuntimeError(f'Failed to load canonical sample {idx} after {self.max_skip_attempts + 1} attempt(s)') from last_error

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        sample_ref = self._sample_ref_from_index(idx)
        episode = self.episodes[sample_ref.episode_index]
        if self.representation_transform is not None:
            return self._load_representation_sample(idx, sample_ref, episode)
        try:
            step = self._materialize_step(episode, sample_ref.step_index)
            sub_task = self._resolve_sub_task(episode, step, sample_ref.step_index)
            observation = self._load_observation_window(episode, sample_ref) if sample_ref.window_size > 1 else self._load_observation(episode, step)
            if self.camera_history_name is not None:
                observation = self._attach_camera_history(episode, sample_ref, observation)
            sample = {'dataset_name': self.dataset_name, 'sample_index': idx, 'episode_id': episode.get('episode_id'), 'step_index': sample_ref.step_index, 'window': {'start_step': sample_ref.step_index, 'end_step': sample_ref.step_index + sample_ref.window_size, 'size': sample_ref.window_size}, 'timestamp': self._timestamp_from_observation(episode, observation, step), 'observation': observation, 'action': self._load_action(episode, step), 'task': episode.get('task', {}), 'sub_task': sub_task, 'sub_goal': sub_task.get('sub_goal') if sub_task else None, 'perception_gt': step.get('perception_gt'), 'meta_data': self._merge_meta_data(episode.get('meta_data', {}), step.get('meta_data', {})), 'source_info': episode.get('source_info', {}), 'raw_step': step}
            sample['value_weight'] = self._coerce_value_weight(sample['meta_data'].get('value_weight', 1.0), context=self._sample_context(episode, sample_ref))
        except CanonicalSampleLoadError:
            raise
        except Exception as exc:
            raise CanonicalSampleLoadError('Failed to load canonical sample', context=self._sample_context(episode, sample_ref), cause=exc) from exc
        return self.transform(sample) if self.transform is not None else sample

    def _load_representation_sample(self, idx: int, sample_ref: SampleIndex, episode: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            counterfactual_action = self.counterfactual_action_spec(idx)
            step = self._materialize_step(episode, sample_ref.step_index)
            sub_task = self._resolve_sub_task(episode, step, sample_ref.step_index)
            state_cfg = self.state_action_config.get('state') or {}
            action_cfg = self.state_action_config.get('action') or {}
            source = self._action_source_for_episode(episode)
            typed_stored_needs_anchor = source == 'stored' and any((isinstance(field, Mapping) and bool(field.get('anchor_source')) for field in action_cfg.get('fields') or []))
            needs_anchor = bool(state_cfg) or source == 'state_derivation' or typed_stored_needs_anchor
            observation = {'cameras': {}, 'tactile': {}} if self.statistics_state_action_only else self._load_non_state_observation(episode, step)
            if not self.statistics_state_action_only and self.camera_history_name is not None:
                observation = self._attach_camera_history(episode, sample_ref, observation)
                if counterfactual_action is not None:
                    observation = self._apply_counterfactual_camera_history(sample_ref, observation, counterfactual_action)
            raw_anchor = None
            if needs_anchor:
                raw_anchor = self._load_required_state_at_step(episode, sample_ref.step_index, self._representation_anchor_sources(), sample_ref=sample_ref)
            observation['state'] = raw_anchor
            timestamp = self._timestamp_from_observation(episode, observation, step)
            if needs_anchor and (not isinstance(raw_anchor, Mapping)):
                raise ValueError('Representation sample is missing its raw absolute anchor state.')
            state_recovery_meta = None
            if needs_anchor:
                (raw_anchor, state_recovery_meta) = self._apply_state_recovery_supervision(episode, sample_ref, raw_anchor, counterfactual_action=counterfactual_action)
            if state_cfg:
                history = None
                if str(state_cfg.get('representation', 'absolute')).lower() == 'relative':
                    history_offset = int(state_cfg['history_offsets'][0])
                    history_step = sample_ref.step_index + history_offset
                    if history_step >= 0:
                        history = self._load_required_state_at_step(episode, history_step, self._state_representation_sources(), sample_ref=sample_ref)
                encoded_state = self.representation_transform.encode_state(raw_anchor, history)
                observation = dict(observation)
                observation['state'] = encoded_state
            if source == 'state_derivation':
                start = int(action_cfg.get('target_start_offset', 1))
                interval = int(action_cfg.get('target_step_interval', 1))
                horizon = int(self.state_action_config['horizon'])
                tail_padding = str(action_cfg.get('tail_padding', 'none')).lower()
                last_state_index = None
                if tail_padding == 'repeat_last':
                    common_length = self._required_state_common_length(episode, self._num_steps(episode))
                    if common_length < 1:
                        raise ValueError('Cannot pad state-derived actions from an empty state stream')
                    last_state_index = common_length - 1
                target_step_indices = [min(sample_ref.step_index + start + row * interval, last_state_index) if last_state_index is not None else sample_ref.step_index + start + row * interval for row in range(horizon)]
                episode_id = str(episode.get('episode_id', ''))
                hold_current_state_supervision = self.hold_current_state_supervision is not None and self.hold_current_state_supervision.matches(episode_id)
                if counterfactual_action is not None:
                    action_mode = str(counterfactual_action.get('action_mode', ''))
                    if action_mode == 'hold_current_state':
                        targets = [raw_anchor for _ in range(horizon)]
                        action = self.representation_transform.encode_action(targets, raw_anchor)
                    elif action_mode != 'recipient_action':
                        raise ValueError('Unsupported counterfactual action mode')
                if counterfactual_action is None and hold_current_state_supervision:
                    action = self.representation_transform.encode_action([raw_anchor for _ in range(horizon)], raw_anchor)
                elif counterfactual_action is None or counterfactual_action.get('action_mode') == 'recipient_action':
                    targets = [self._load_required_state_at_step(episode, target_step_index, self._action_representation_sources(), sample_ref=sample_ref) for target_step_index in target_step_indices]
                    skip_gripper_compensation = False
                    if self.target_overrides is not None:
                        targets = self.target_overrides.apply(targets, episode_id=episode_id, target_step_indices=target_step_indices)
                        skip_gripper_compensation = self.target_overrides.disables_gripper(episode_id)
                    if self.action_supervision is None or skip_gripper_compensation:
                        action = self.representation_transform.encode_action(targets, raw_anchor)
                    else:
                        semantic_action = self.representation_transform.encode_action_semantic(targets, raw_anchor)
                        semantic_action = self.action_supervision.apply(semantic_action, episode_id=str(episode.get('episode_id', '')), target_step_indices=target_step_indices)
                        action = self.representation_transform.encode_action_semantic_values(semantic_action)
            elif source == 'stored':
                action = self._load_action(episode, step)
                override = self.action_source_overrides.get(self._episode_source_name(episode))
                if override is not None:
                    values = action.get('values') if isinstance(action, Mapping) else action
                    rows = np.asarray(values)
                    expected = (int(self.state_action_config['horizon']), int(override['raw_dim']))
                    if rows.shape != expected or not np.all(np.isfinite(rows)):
                        raise ValueError(f'Stored-command action must have finite shape {expected}, got {rows.shape}')
                    targets = tuple(({str(override['raw_source']): row} for row in rows))
                    action = self.representation_transform.encode_action(targets, raw_anchor)
                else:
                    encode_stored = getattr(self.representation_transform, 'encode_stored_action', None)
                    if callable(encode_stored):
                        action = encode_stored(action, raw_anchor)
            elif source == 'none':
                action = None
            else:
                raise ValueError(f'Unsupported representation action source: {source}')
            if counterfactual_action is not None:
                common_suffix_action_mode = counterfactual_action.get('common_suffix_action_mode')
                if common_suffix_action_mode is not None:
                    if source != 'state_derivation' or common_suffix_action_mode != 'hold_current_state':
                        raise ValueError('Unsupported counterfactual common action suffix')
                    supervised_steps = int(counterfactual_action['supervised_action_steps'])
                    hold_action = self.representation_transform.encode_action([raw_anchor for _ in range(horizon)], raw_anchor)
                    action_values = np.asarray(action['values'])
                    hold_values = np.asarray(hold_action['values'])
                    if action_values.shape != hold_values.shape or action_values.shape[0] != horizon:
                        raise ValueError('Counterfactual action/hold suffix shapes differ')
                    common_values = np.array(action_values, copy=True)
                    common_values[supervised_steps:] = hold_values[supervised_steps:]
                    action = dict(action)
                    action['values'] = common_values
            meta_data = self._merge_meta_data(episode.get('meta_data', {}), step.get('meta_data', {}))
            if state_recovery_meta is not None:
                meta_data = dict(meta_data)
                meta_data.update(state_recovery_meta)
            if source == 'state_derivation' and hold_current_state_supervision:
                meta_data = dict(meta_data)
                meta_data.update({'hold_current_state_supervision': True, 'hold_current_state_horizon': int(self.state_action_config['horizon'])})
            if source == 'stored' and self._episode_source_name(episode) in self.action_source_overrides:
                meta_data = dict(meta_data)
                meta_data.update({'action_source_override': 'stored', 'gripper_compensation_disabled': True})
            if counterfactual_action is not None:
                supervised_steps = int(counterfactual_action['supervised_action_steps'])
                meta_data = dict(meta_data)
                for key in tuple(meta_data):
                    if str(key).startswith('counterfactual_'):
                        meta_data.pop(key)
                meta_data.update({'action_value_weight': [1.0] * supervised_steps + [0.0] * (horizon - supervised_steps), 'counterfactual_action_supervision': True, 'counterfactual_action_mode': counterfactual_action['action_mode'], 'counterfactual_action_sample_kind': counterfactual_action.get('sample_kind'), 'counterfactual_pair_id': counterfactual_action.get('pair_id'), 'counterfactual_action_pair_id': counterfactual_action.get('pair_id'), 'counterfactual_common_suffix_action_mode': counterfactual_action.get('common_suffix_action_mode'), 'counterfactual_rollout_stage': counterfactual_action.get('stage'), 'counterfactual_source_event_type': 'positive_rollout' if str(counterfactual_action.get('sample_kind', '')).startswith('positive_rollout_') else 'target_not_ready_wait' if counterfactual_action.get('sample_kind') == 'positive_timing_wait' else 'distractor_wait' if counterfactual_action.get('sample_kind') == 'negative_distractor_wait' else 'matching_pick', 'counterfactual_recipient_target': counterfactual_action['recipient_target'], 'counterfactual_donor_episode_id': counterfactual_action['donor_episode_id'], 'counterfactual_donor_target': counterfactual_action['donor_target'], 'counterfactual_pairing_mode': counterfactual_action.get('pairing_mode')})
            sample = {'dataset_name': self.dataset_name, 'sample_index': idx, 'episode_id': episode.get('episode_id'), 'step_index': sample_ref.step_index, 'window': {'start_step': sample_ref.step_index, 'end_step': sample_ref.step_index + 1, 'size': 1}, 'timestamp': timestamp, 'observation': observation, 'action': action, 'task': episode.get('task', {}), 'sub_task': sub_task, 'sub_goal': sub_task.get('sub_goal') if sub_task else None, 'perception_gt': step.get('perception_gt'), 'meta_data': meta_data, 'source_info': episode.get('source_info', {}), 'raw_state': raw_anchor, 'raw_step': step}
            sample['value_weight'] = self._coerce_value_weight(sample['meta_data'].get('value_weight', 1.0), context=self._sample_context(episode, sample_ref))
        except CanonicalSampleLoadError:
            raise
        except Exception as exc:
            raise CanonicalSampleLoadError('Failed to load state/action representation sample', context=self._sample_context(episode, sample_ref), cause=exc) from exc
        return self.transform(sample) if self.transform is not None else sample

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def _build_compact_sample_index(self) -> tuple[List[EpisodeIndex], List[int]]:
        index: List[EpisodeIndex] = []
        cumsum: List[int] = [0]
        for (episode_index, episode) in enumerate(self.episodes):
            num_steps = self._num_steps(episode)
            if self.representation_transform is not None:
                raw_num_samples = self._representation_num_samples(episode, num_steps)
            elif self._stored_transform is not None and self.sample_window_size <= 1:
                raw_num_samples = self._legacy_stored_transform_num_samples(episode, num_steps)
            elif self.sample_window_size <= 1:
                raw_num_samples = num_steps
            else:
                max_start = self._max_window_start(episode, num_steps)
                raw_num_samples = 0 if max_start < 0 else max_start // self.sample_stride + 1
            sample_local_indices = self._capped_local_indices(raw_num_samples)
            if self.representation_transform is not None and self._stored_action_validity_column(episode) is not None:
                candidates = sample_local_indices if sample_local_indices is not None else tuple(range(raw_num_samples))
                anchor_stride = int((self.state_action_config.get('sampling') or {}).get('anchor_stride', 1))
                sample_local_indices = self._stored_action_valid_local_indices(episode, candidates, anchor_stride=anchor_stride)
            num_samples = len(sample_local_indices) if sample_local_indices is not None else raw_num_samples
            index.append(EpisodeIndex(episode_index=episode_index, num_steps=num_steps, num_samples=num_samples, sample_local_indices=sample_local_indices))
            cumsum.append(cumsum[-1] + num_samples)
        return (index, cumsum)

    def _representation_num_samples(self, episode: Mapping[str, Any], num_steps: int) -> int:
        common_length = self._required_state_common_length(episode, num_steps)
        action_cfg = self.state_action_config.get('action') or {}
        source = self._action_source_for_episode(episode)
        if source == 'state_derivation':
            if str(action_cfg.get('tail_padding', 'none')).lower() == 'repeat_last':
                required_last = 0
            else:
                horizon = int(self.state_action_config['horizon'])
                required_last = int(action_cfg.get('target_start_offset', 1)) + (horizon - 1) * int(action_cfg.get('target_step_interval', 1))
        elif source == 'stored':
            required_last = 0
        else:
            required_last = 0
        if common_length < required_last + 1:
            return 0
        anchor_stride = int((self.state_action_config.get('sampling') or {}).get('anchor_stride', 1))
        raw_num_samples = (common_length - 1 - required_last) // anchor_stride + 1
        if source == 'stored':
            raw_num_samples = self._stored_representation_prefix_samples(episode, raw_num_samples, anchor_stride=anchor_stride)
        return raw_num_samples

    def _legacy_stored_transform_num_samples(self, episode: Mapping[str, Any], num_steps: int) -> int:
        horizon = self._stored_action_horizon()
        interval = self._stored_action_step_interval()
        action_rows = num_steps
        rule = ((episode.get('step_index') or {}).get('stream_rules') or {}).get('action') or {}
        if isinstance(rule, Mapping) and rule.get('max_index') is not None:
            action_rows = min(action_rows, int(rule['max_index']) + 1)
        return max(0, action_rows - horizon * interval)

    def _required_state_common_length(self, episode: Mapping[str, Any], num_steps: int) -> int:
        sources = self._state_representation_sources()
        action_cfg = self.state_action_config.get('action') or {}
        if str(action_cfg.get('source', 'stored')).lower() == 'state_derivation':
            sources = tuple(dict.fromkeys((*sources, *self._action_representation_sources())))
        elif str(action_cfg.get('source', 'stored')).lower() == 'stored':
            sources = tuple(dict.fromkeys((*sources, *(str(field['anchor_source']) for field in action_cfg.get('fields') or [] if isinstance(field, Mapping) and field.get('anchor_source')))))
        compact_length = self._compact_required_state_common_length(episode, sources, num_steps)
        if compact_length is not None:
            self._index_build_stats['compact_metadata_episodes'] += 1
            return compact_length
        batched_length = self._batched_required_state_common_length(episode, sources, num_steps)
        if batched_length is not None:
            self._index_build_stats['batched_step_table_episodes'] += 1
            return batched_length
        self._index_build_stats['fallback_scan_episodes'] += 1
        for step_index in range(num_steps):
            if not all((self._state_source_available(episode, source, step_index) for source in sources)):
                return step_index
        return num_steps

    def _batched_required_state_common_length(self, episode: Mapping[str, Any], sources: tuple[str, ...], num_steps: int) -> int | None:
        if episode.get('steps') or not sources:
            return None
        if type(self.reader).step_table_values is BaseStorageReader.step_table_values:
            return None
        step_index_cfg = episode.get('step_index') or {}
        stream_rules = step_index_cfg.get('stream_rules') or {}
        if not isinstance(stream_rules, Mapping):
            return None
        default_rule = step_index_cfg.get('default_index_rule', 'same_as_step')
        step_indices = list(range(num_steps))
        common_length = num_steps
        for source in sources:
            try:
                stream_key = self._source_stream_key(episode, source)
            except KeyError:
                return 0
            rule = stream_rules.get(stream_key, default_rule)
            if not isinstance(rule, Mapping):
                return None
            rule_type = str(rule.get('index_rule') or rule.get('type') or 'same_as_step')
            if rule_type != 'step_table_column':
                return None
            column = str(rule.get('column') or f'{stream_key}.index')
            offset = int(rule.get('offset', 0))
            max_index = rule.get('max_index')
            values = self.reader.step_table_values(episode, column, step_indices)
            if len(values) != num_steps:
                return None
            stream_length = self.reader.stream_length(episode, stream_key)
            for (position, value) in enumerate(values):
                if value is None:
                    common_length = min(common_length, position)
                    break
                stream_index = int(value)
                if stream_index < 0:
                    common_length = min(common_length, position)
                    break
                if max_index is not None and stream_index > int(max_index):
                    common_length = min(common_length, position)
                    break
                if stream_length is not None and stream_index >= int(stream_length):
                    common_length = min(common_length, position)
                    break
            del offset
        return common_length

    def _compact_required_state_common_length(self, episode: Mapping[str, Any], sources: tuple[str, ...], num_steps: int) -> int | None:
        if episode.get('steps'):
            return None
        common_length = num_steps
        streams = (episode.get('storage') or {}).get('streams') or {}
        if not isinstance(streams, Mapping):
            return 0
        step_index = episode.get('step_index') or {}
        stream_rules = step_index.get('stream_rules') or {}
        if not isinstance(stream_rules, Mapping):
            return 0
        default_rule = step_index.get('default_index_rule', 'same_as_step')
        deterministic_rules = {'same_as_step', 'clock_index'}
        for source in sources:
            try:
                stream_key = self._source_stream_key(episode, source)
            except KeyError:
                return 0
            if not isinstance(streams.get(stream_key), Mapping):
                return 0
            rule = stream_rules.get(stream_key, default_rule)
            if isinstance(rule, Mapping):
                rule_type = str(rule.get('index_rule') or rule.get('type') or 'same_as_step')
                offset = int(rule.get('offset', 0))
                max_index = rule.get('max_index')
            else:
                rule_type = str(rule or 'same_as_step')
                offset = 0
                max_index = None
            if rule_type not in deterministic_rules:
                return None
            if offset < 0:
                return 0
            bounds: list[int] = []
            stream_length = self.reader.stream_length(episode, stream_key)
            if stream_length is not None:
                exact_length = int(stream_length)
                if exact_length < 0:
                    return 0
                bounds.append(exact_length - offset)
            if max_index is not None:
                bounds.append(int(max_index) + 1 - offset)
            if not bounds:
                return None
            common_length = min(common_length, max(0, min(bounds)))
            if common_length == 0:
                return 0
        return common_length

    def _stored_representation_prefix_samples(self, episode: Mapping[str, Any], raw_num_samples: int, *, anchor_stride: int) -> int:
        horizon = int(self.state_action_config['horizon'])
        local_indices = tuple(range(raw_num_samples))
        starts = self._stored_action_start_indices(episode, local_indices, anchor_stride=anchor_stride)
        if starts is not None:
            self._index_build_stats['stored_action_batched_episodes'] += 1
            upper_bound = self._stored_action_available_upper_bound(episode)
            if upper_bound is False:
                return 0
            valid = 0
            for start in starts:
                if start is None:
                    break
                if start < 0:
                    break
                if upper_bound is not None and start + horizon - 1 > upper_bound:
                    break
                valid += 1
            return valid
        self._index_build_stats['stored_action_fallback_episodes'] += 1
        valid = 0
        for local_index in range(raw_num_samples):
            step_index = local_index * anchor_stride
            step = self._materialize_step(episode, step_index)
            ref = step.get('action')
            if not isinstance(ref, Mapping) or ref.get('index') is None:
                break
            start = int(ref['index'])
            if not self._stored_action_indices_available(episode, [start + row for row in range(horizon)]):
                break
            valid += 1
        return valid

    def _stored_action_start_indices(self, episode: Mapping[str, Any], local_indices: tuple[int, ...], *, anchor_stride: int) -> list[int | None] | None:
        if episode.get('steps') or episode.get('step_overrides'):
            return None
        if 'action' not in self._step_stream_keys(episode):
            return None
        step_index_cfg = episode.get('step_index') or {}
        stream_rules = step_index_cfg.get('stream_rules') or {}
        rule = stream_rules.get('action', step_index_cfg.get('default_index_rule', 'same_as_step'))
        if isinstance(rule, Mapping):
            rule_type = str(rule.get('index_rule') or rule.get('type') or 'same_as_step')
            offset = int(rule.get('offset', 0))
            max_index = rule.get('max_index')
        else:
            rule_type = str(rule or 'same_as_step')
            offset = 0
            max_index = None
        step_indices = [int(local_index) * anchor_stride for local_index in local_indices]
        if rule_type in {'same_as_step', 'clock_index'}:
            values: list[Any] = [step_index + offset for step_index in step_indices]
        elif rule_type == 'step_table_column':
            if not isinstance(rule, Mapping):
                return None
            if type(self.reader).step_table_values is BaseStorageReader.step_table_values:
                return None
            column = str(rule.get('column') or 'action.index')
            values = self.reader.step_table_values(episode, column, step_indices)
            if len(values) != len(step_indices):
                return None
            values = [step_index + offset if value is None else value for (step_index, value) in zip(step_indices, values, strict=True)]
        else:
            return None
        starts = [int(value) if value is not None else None for value in values]
        if max_index is not None:
            starts = [min(start, int(max_index)) if start is not None else None for start in starts]
        return starts

    def _stored_action_available_upper_bound(self, episode: Mapping[str, Any]) -> int | None | bool:
        action_stream = ((episode.get('storage') or {}).get('streams') or {}).get('action')
        if not isinstance(action_stream, Mapping):
            return False
        exact_length = self.reader.stream_length(episode, 'action')
        if exact_length is not None:
            return int(exact_length) - 1
        action_spec = action_stream.get('action_spec') or {}
        row_count = action_spec.get('num_rows', action_spec.get('count')) if isinstance(action_spec, Mapping) else None
        if row_count is not None:
            return int(row_count) - 1
        rule = ((episode.get('step_index') or {}).get('stream_rules') or {}).get('action') or {}
        max_index = rule.get('max_index') if isinstance(rule, Mapping) else None
        return int(max_index) if max_index is not None else None

    def _stored_action_indices_available(self, episode: Mapping[str, Any], indices: List[int]) -> bool:
        action_stream = ((episode.get('storage') or {}).get('streams') or {}).get('action')
        if not isinstance(action_stream, Mapping):
            return False
        exact_length = self.reader.stream_length(episode, 'action')
        if exact_length is not None:
            return all((0 <= index < int(exact_length) for index in indices))
        action_spec = action_stream.get('action_spec') or {}
        row_count = action_spec.get('num_rows', action_spec.get('count')) if isinstance(action_spec, Mapping) else None
        if row_count is not None:
            return all((0 <= index < int(row_count) for index in indices))
        rule = ((episode.get('step_index') or {}).get('stream_rules') or {}).get('action') or {}
        max_index = rule.get('max_index') if isinstance(rule, Mapping) else None
        return max_index is None or all((0 <= index <= int(max_index) for index in indices))

    def _stored_action_valid_local_indices(self, episode: Mapping[str, Any], candidates: tuple[int, ...], *, anchor_stride: int) -> tuple[int, ...]:
        horizon = int(self.state_action_config['horizon'])
        starts = self._stored_action_start_indices(episode, candidates, anchor_stride=anchor_stride)
        windows: list[tuple[int, tuple[int, ...]]] = []
        required_indices: set[int] = set()
        for (position, local_index) in enumerate(candidates):
            if starts is None:
                step = self._materialize_step(episode, int(local_index) * anchor_stride)
                ref = step.get('action')
                if not isinstance(ref, Mapping) or ref.get('index') is None:
                    continue
                start = int(ref['index'])
            else:
                start = starts[position]
                if start is None:
                    continue
            indices = tuple((start + row for row in range(horizon)))
            windows.append((int(local_index), indices))
            required_indices.update(indices)
        if not required_indices:
            return ()
        ordered_indices = sorted(required_indices)
        payloads = self._read_many_with_context(episode, 'action', sample_ref=None, indices=ordered_indices)
        validity_column = self._stored_action_validity_column(episode)
        validity_by_index = {index: self._stored_action_payload_is_valid(payload, validity_column=validity_column) for (index, payload) in zip(ordered_indices, payloads, strict=True)}
        return tuple((local_index for (local_index, indices) in windows if all((validity_by_index[index] for index in indices))))

    def _stored_action_validity_column(self, episode: Mapping[str, Any] | None=None) -> str | None:
        if episode is not None:
            streams = (episode.get('storage') or {}).get('streams') or {}
            action_stream = streams.get('action') if isinstance(streams, Mapping) else None
            action_spec = action_stream.get('action_spec') if isinstance(action_stream, Mapping) else None
            validity = action_spec.get('validity') if isinstance(action_spec, Mapping) else None
            column = validity.get('column') if isinstance(validity, Mapping) else None
            if column:
                return str(column)
        schema = (self.manifest.get('extra') or {}).get('schema') or {}
        action_schema = schema.get('action') if isinstance(schema, Mapping) else None
        validity = action_schema.get('validity') if isinstance(action_schema, Mapping) else None
        column = validity.get('column') if isinstance(validity, Mapping) else None
        return str(column) if column else None

    @staticmethod
    def _stored_action_payload_is_valid(payload: Any, *, validity_column: str | None) -> bool:
        column = validity_column
        if column is None:
            return True
        if not isinstance(payload, Mapping):
            raise ValueError(f'Stored action payload is missing validity field {column!r}')
        value = payload.get(column)
        if value is None:
            message = payload.get('message')
            value = message.get(column) if isinstance(message, Mapping) else None
        if value is None:
            raise ValueError(f'Stored action payload is missing validity field {column!r}')
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f'Stored action validity field {column!r} must be scalar')
        scalar = float(array.reshape(-1)[0])
        if not np.isfinite(scalar):
            raise ValueError(f'Stored action validity field {column!r} must be finite')
        return scalar > 0.0

    def _max_window_start(self, episode: Mapping[str, Any], num_steps: int) -> int:
        max_start = num_steps - self.sample_window_size
        required_last_offset = self.sample_window_size - 1
        for stream_key in self._action_state_stream_keys():
            stream_max_index = self._stream_max_index(episode, stream_key)
            if stream_max_index is None:
                continue
            max_start = min(max_start, stream_max_index - required_last_offset)
        return max_start

    def _action_state_stream_keys(self) -> List[str]:
        cfg = self.action_config
        source = str(cfg.get('source', 'stored')).lower()
        if source != 'state_derivation':
            return []
        derivation_cfg = cfg.get('derivation') or cfg
        if not isinstance(derivation_cfg, Mapping):
            return []
        strategy = str(derivation_cfg.get('strategy', 'eef_pose_delta')).lower()
        if strategy != 'eef_pose_delta':
            return []
        keys = list(derivation_cfg.get('state_keys') or ['eef_pose.left', 'eef_pose.right'])
        keys.extend(list(derivation_cfg.get('gripper_state_keys') or []))
        return sorted(set(keys))

    def _stream_max_index(self, episode: Mapping[str, Any], stream_key: str) -> Optional[int]:
        step_index_cfg = episode.get('step_index') or {}
        stream_rules = step_index_cfg.get('stream_rules') or {}
        rule = stream_rules.get(stream_key)
        if not isinstance(rule, Mapping):
            return None
        max_index = rule.get('max_index')
        return int(max_index) if max_index is not None else None

    def _state_representation_sources(self) -> tuple[str, ...]:
        fields = (self.state_action_config.get('state') or {}).get('fields') or []
        return tuple(dict.fromkeys((str(field['source']) for field in fields)))

    def _action_representation_sources(self) -> tuple[str, ...]:
        fields = (self.state_action_config.get('action') or {}).get('fields') or []
        return tuple(dict.fromkeys((str(field['source']) for field in fields)))

    def _representation_anchor_sources(self) -> tuple[str, ...]:
        sources = self._state_representation_sources()
        action_cfg = self.state_action_config.get('action') or {}
        if str(action_cfg.get('source', 'stored')).lower() == 'state_derivation':
            sources = tuple(dict.fromkeys((*sources, *self._action_representation_sources())))
        elif str(action_cfg.get('source', 'stored')).lower() == 'stored':
            anchor_sources = tuple((str(field['anchor_source']) for field in action_cfg.get('fields') or [] if isinstance(field, Mapping) and field.get('anchor_source')))
            sources = tuple(dict.fromkeys((*sources, *anchor_sources)))
        return sources

    def _source_stream_key(self, episode: Mapping[str, Any], source: str) -> str:
        streams = (episode.get('storage') or {}).get('streams') or {}
        matches = [key for key in streams if source == key or source.startswith(f'{key}.')]
        if not matches:
            raise KeyError(f'Required source {source!r} does not match a canonical state stream.')
        return max(matches, key=len)

    def _state_source_available(self, episode: Mapping[str, Any], source: str, step_index: int) -> bool:
        try:
            stream_key = self._source_stream_key(episode, source)
        except KeyError:
            return False
        step_index_cfg = episode.get('step_index') or {}
        stream_rules = step_index_cfg.get('stream_rules') or {}
        rule = stream_rules.get(stream_key, step_index_cfg.get('default_index_rule', 'same_as_step'))
        if isinstance(rule, Mapping):
            rule_type = str(rule.get('index_rule') or rule.get('type') or 'same_as_step')
            offset = int(rule.get('offset', 0))
            max_index = rule.get('max_index')
        else:
            rule_type = str(rule or 'same_as_step')
            offset = 0
            max_index = None
        if rule_type == 'step_table_column':
            column = str(rule.get('column') or f'{stream_key}.index') if isinstance(rule, Mapping) else f'{stream_key}.index'
            try:
                value = self.reader.step_table_value(episode, column, step_index)
            except NotImplementedError:
                value = step_index + offset
            if value is None:
                return False
            stream_index = int(value)
        else:
            stream_index = step_index + offset
        if max_index is not None and stream_index > int(max_index):
            return False
        if stream_index < 0:
            return False
        stream_length = self.reader.stream_length(episode, stream_key)
        return stream_length is None or stream_index < int(stream_length)

    def _load_required_state_at_step(self, episode: Mapping[str, Any], step_index: int, sources: tuple[str, ...], *, sample_ref: SampleIndex) -> Dict[str, Any]:
        step = self._materialize_step(episode, step_index)
        state_ref = (step.get('observation') or {}).get('state')
        if not isinstance(state_ref, Mapping) or 'streams' not in state_ref:
            loaded = self._load_state(episode, state_ref)
            if not isinstance(loaded, Mapping):
                raise ValueError(f'Required state at episode step {step_index} is missing.')
            return dict(loaded)
        stream_refs = state_ref.get('streams') or {}
        stream_keys = tuple(dict.fromkeys((self._source_stream_key(episode, source) for source in sources)))
        clock_timestamp = None
        if any((isinstance(stream_refs.get(stream_key), Mapping) and stream_refs[stream_key].get('timestamp_from') is not None for stream_key in stream_keys)):
            clock_timestamp = self._clock_timestamp_for_step(episode, step, sample_ref=sample_ref, step_index=step_index)
        state: Dict[str, Any] = {}
        for stream_key in stream_keys:
            ref = stream_refs.get(stream_key)
            if not isinstance(ref, Mapping):
                raise KeyError(f'Required stream {stream_key!r} is missing at episode step {step_index}.')
            timestamp = ref.get('timestamp')
            index = ref.get('index')
            if ref.get('timestamp_from') is not None:
                if clock_timestamp is None:
                    raise ValueError(f'Unable to resolve timestamp_from for {stream_key!r} at step {step_index}.')
                timestamp = clock_timestamp
                index = None
            payload = self._read_with_context(episode, stream_key, sample_ref=sample_ref, step_index=step_index, index=index, index_range=ref.get('index_range'), timestamp=timestamp, time_range=ref.get('time_range'), interpolate=bool(ref.get('interpolate', False)))
            state[stream_key] = self._apply_state_ref_values(payload, ref)
        return state

    def _clock_timestamp_for_step(self, episode: Mapping[str, Any], step: Mapping[str, Any], *, sample_ref: SampleIndex, step_index: int) -> float | None:
        clock_key = self._step_clock(episode)
        obs = step.get('observation') or {}
        if clock_key.startswith('camera.'):
            ref = (obs.get('cameras') or {}).get(clock_key.split('.', 1)[1])
        elif clock_key.startswith('tactile.'):
            ref = (obs.get('tactile') or {}).get(clock_key.split('.', 1)[1])
        else:
            ref = ((obs.get('state') or {}).get('streams') or {}).get(clock_key)
        if not isinstance(ref, Mapping):
            return None
        payload = self._read_with_context(episode, clock_key, sample_ref=sample_ref, step_index=step_index, index=ref.get('index'), timestamp=ref.get('timestamp'))
        return self._payload_timestamp_sec(payload)

    def _sample_ref_from_index(self, idx: int) -> SampleIndex:
        idx = self.base_sample_index(idx)
        dataset_len = len(self)
        if idx < 0:
            idx += dataset_len
        if idx < 0 or idx >= dataset_len:
            raise IndexError(idx)
        compact_pos = bisect_right(self.sample_cumsum, idx) - 1
        compact = self.episode_index[compact_pos]
        local_index = idx - self.sample_cumsum[compact_pos]
        if compact.sample_local_indices is not None:
            local_index = compact.sample_local_indices[local_index]
        if self.representation_transform is not None:
            anchor_stride = int((self.state_action_config.get('sampling') or {}).get('anchor_stride', 1))
            step_index = local_index * anchor_stride
        else:
            step_index = local_index if self.sample_window_size <= 1 else local_index * self.sample_stride
        return SampleIndex(episode_index=compact.episode_index, step_index=step_index, window_size=1 if self.representation_transform is not None else self.sample_window_size)

    def _capped_local_indices(self, num_samples: int) -> tuple[int, ...] | None:
        cap = self.max_samples_per_episode
        if cap is None or num_samples <= cap:
            return None
        if cap == 1:
            return (0,)
        max_index = num_samples - 1
        return tuple((round(i * max_index / (cap - 1)) for i in range(cap)))

    def _num_steps(self, episode: Mapping[str, Any]) -> int:
        step_index = episode.get('step_index') or {}
        if 'num_steps' in step_index:
            return int(step_index['num_steps'])
        return len(episode.get('steps', []))

    def _materialize_step(self, episode: Mapping[str, Any], step_index: int) -> Dict[str, Any]:
        steps = episode.get('steps') or []
        if steps:
            return self._copy_materialized_step(dict(steps[step_index]))
        cache_key = (id(episode), int(step_index))
        cached = self._materialized_step_cache_get(cache_key)
        if cached is not None:
            self._step_cache_stats['hits'] += 1
            return self._copy_materialized_step(cached)
        self._step_cache_stats['misses'] += 1
        observation = {'cameras': {}, 'tactile': {}, 'state': {'streams': {}}}
        action_ref = None
        for stream_key in self._step_stream_keys(episode):
            ref = self._stream_ref_for_step(episode, stream_key, step_index)
            if stream_key == 'action':
                action_ref = ref
            elif stream_key.startswith('camera.'):
                observation['cameras'][stream_key.split('.', 1)[1]] = ref
            elif stream_key.startswith('tactile.'):
                observation['tactile'][stream_key.split('.', 1)[1]] = ref
            else:
                observation['state']['streams'][stream_key] = ref
        timestamp = self._step_table_value_or_none(episode, 'timestamp', step_index)
        sub_task_id = self._step_table_value_or_none(episode, 'sub_task_id', step_index)
        value_weight = self._step_table_value_or_none(episode, 'value_weight', step_index)
        if not observation['state']['streams']:
            observation['state'] = None
        step = {'timestamp': timestamp, 'observation': observation, 'action': action_ref, 'perception_gt': None, 'sub_task_id': sub_task_id if sub_task_id is not None else self._sub_task_id_for_step(episode, step_index), 'meta_data': {'value_weight': value_weight} if value_weight is not None else {}, 'extra': {}}
        for override in episode.get('step_overrides', []) or []:
            if int(override.get('step_index', -1)) == step_index:
                step = self._apply_step_override(step, override)
        self._materialized_step_cache_put(cache_key, step)
        return self._copy_materialized_step(step)

    def _materialized_step_cache_get(self, key: tuple[int, int]) -> Dict[str, Any] | None:
        if self.max_cached_steps <= 0:
            return None
        if key not in self._step_cache:
            return None
        self._step_cache.move_to_end(key)
        return self._step_cache[key]

    def _materialized_step_cache_put(self, key: tuple[int, int], step: Dict[str, Any]) -> None:
        if self.max_cached_steps <= 0:
            return
        self._step_cache[key] = step
        self._step_cache.move_to_end(key)
        while len(self._step_cache) > self.max_cached_steps:
            self._step_cache.popitem(last=False)
            self._step_cache_stats['evictions'] += 1

    def _copy_materialized_step(self, step: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(step)
        observation = out.get('observation')
        if isinstance(observation, Mapping):
            obs = dict(observation)
            if isinstance(obs.get('cameras'), Mapping):
                obs['cameras'] = {name: dict(ref) if isinstance(ref, Mapping) else ref for (name, ref) in obs['cameras'].items()}
            if isinstance(obs.get('tactile'), Mapping):
                obs['tactile'] = {name: dict(ref) if isinstance(ref, Mapping) else ref for (name, ref) in obs['tactile'].items()}
            state = obs.get('state')
            if isinstance(state, Mapping):
                copied_state = dict(state)
                if isinstance(copied_state.get('streams'), Mapping):
                    copied_state['streams'] = {name: self._copy_stream_ref(ref) if isinstance(ref, Mapping) else ref for (name, ref) in copied_state['streams'].items()}
                obs['state'] = copied_state
            out['observation'] = obs
        if isinstance(out.get('meta_data'), Mapping):
            out['meta_data'] = dict(out['meta_data'])
        if isinstance(out.get('extra'), Mapping):
            out['extra'] = dict(out['extra'])
        return out

    def _copy_stream_ref(self, ref: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(ref)
        if isinstance(out.get('values'), Mapping):
            out['values'] = dict(out['values'])
        return out

    def cache_stats(self) -> Dict[str, Any]:
        return {'index_build': dict(self._index_build_stats), 'materialized_step_cache_hits': self._step_cache_stats['hits'], 'materialized_step_cache_misses': self._step_cache_stats['misses'], 'materialized_step_cache_evictions': self._step_cache_stats['evictions'], 'materialized_step_cache_entries': len(self._step_cache), 'max_cached_steps': self.max_cached_steps, 'batch_state_window': self.batch_state_window, 'batch_prefetch': self.batch_prefetch, 'batch_prefetch_cameras': self.batch_prefetch_cameras, 'batch_prefetch_stats': dict(self._batch_prefetch_stats), 'reader': self.reader.cache_stats()}

    def _step_table_value_or_none(self, episode: Mapping[str, Any], column: str, step_index: int) -> Any:
        step_index_cfg = episode.get('step_index') or {}
        if not step_index_cfg.get('step_table'):
            return None
        try:
            return self.reader.step_table_value(episode, column, step_index)
        except Exception:
            return None

    def _step_clock(self, episode: Mapping[str, Any]) -> str:
        step_index = episode.get('step_index') or {}
        timeline = episode.get('timeline') or {}
        return str(step_index.get('clock') or timeline.get('step_clock') or 'camera.left_wrist')

    def _step_stream_keys(self, episode: Mapping[str, Any]) -> List[str]:
        step_index = episode.get('step_index') or {}
        stream_rules = step_index.get('stream_rules')
        if isinstance(stream_rules, Mapping) and stream_rules:
            return list(stream_rules.keys())
        return [key for key in episode.get('storage', {}).get('streams', {}) if key.startswith(('camera.', 'tactile.', 'state.', 'eef_pose.', 'robot_state.', 'head.', 'base.'))]

    def _stream_ref_for_step(self, episode: Mapping[str, Any], stream_key: str, step_index: int) -> Dict[str, Any]:
        step_index_cfg = episode.get('step_index') or {}
        stream_rules = step_index_cfg.get('stream_rules') or {}
        rule = stream_rules.get(stream_key, step_index_cfg.get('default_index_rule', 'same_as_step'))
        if isinstance(rule, Mapping):
            rule_type = str(rule.get('index_rule') or rule.get('type') or 'same_as_step')
            offset = int(rule.get('offset', 0))
            max_index = rule.get('max_index')
        else:
            rule_type = str(rule or 'same_as_step')
            offset = 0
            max_index = None
        if rule_type not in {'same_as_step', 'clock_index', 'nearest_clock_time', 'interpolate_clock_time', 'step_table_column'}:
            raise ValueError(f'Unsupported compact stream index rule `{rule_type}` for `{stream_key}`')
        ref_values: Dict[str, Any] = {}
        if rule_type == 'step_table_column':
            column = str(rule.get('column') or f'{stream_key}.index') if isinstance(rule, Mapping) else f'{stream_key}.index'
            value = self.reader.step_table_value(episode, column, step_index)
            index = int(value) if value is not None else step_index + offset
            value_columns = rule.get('value_columns') if isinstance(rule, Mapping) else None
            if isinstance(value_columns, Mapping):
                for (value_name, value_column) in value_columns.items():
                    ref_value = self._step_table_value_or_none(episode, str(value_column), step_index)
                    if ref_value is not None:
                        ref_values[str(value_name)] = ref_value
        else:
            index = step_index + offset
        if max_index is not None:
            index = min(index, int(max_index))
        if rule_type in {'nearest_clock_time', 'interpolate_clock_time'}:
            ref = {'index': index, 'timestamp_from': self._step_clock(episode), 'interpolate': rule_type == 'interpolate_clock_time'}
        else:
            ref = {'index': index}
        if ref_values:
            ref['values'] = ref_values
        return ref

    def _sub_task_id_for_step(self, episode: Mapping[str, Any], step_index: int) -> Optional[str]:
        for sub_task in episode.get('sub_tasks', []) or []:
            start = sub_task.get('start_step')
            end = sub_task.get('end_step')
            if start is not None and end is not None and (int(start) <= step_index <= int(end)):
                return sub_task.get('sub_task_id')
        return None

    def _apply_step_override(self, step: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(step)
        for (key, value) in override.items():
            if key == 'step_index':
                continue
            if key == 'missing_streams':
                self._remove_missing_streams(out, value or [])
                continue
            out[key] = value
        return out

    def _remove_missing_streams(self, step: Dict[str, Any], stream_keys: List[str]) -> None:
        obs = step.get('observation') or {}
        for stream_key in stream_keys:
            if stream_key.startswith('camera.'):
                obs.get('cameras', {}).pop(stream_key.split('.', 1)[1], None)
            elif stream_key.startswith('tactile.'):
                obs.get('tactile', {}).pop(stream_key.split('.', 1)[1], None)
            else:
                state = obs.get('state') or {}
                streams = state.get('streams') if isinstance(state, Mapping) else None
                if isinstance(streams, Mapping):
                    streams.pop(stream_key, None)

    def _timestamp_from_observation(self, episode: Mapping[str, Any], observation: Mapping[str, Any], step: Mapping[str, Any]) -> Any:
        payload = self._clock_payload_from_observation(episode, observation)
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if isinstance(payload, Mapping):
            return {'log_time_ns': payload.get('log_time_ns'), 'publish_time_ns': payload.get('publish_time_ns'), 'sensor_time_sec': payload.get('timestamp')}
        return step.get('timestamp')

    def _clock_payload_from_observation(self, episode: Mapping[str, Any], observation: Mapping[str, Any]) -> Any:
        step_clock = self._step_clock(episode)
        if step_clock.startswith('camera.'):
            return (observation.get('cameras') or {}).get(step_clock.split('.', 1)[1])
        if step_clock.startswith('tactile.'):
            return (observation.get('tactile') or {}).get(step_clock.split('.', 1)[1])
        state = observation.get('state')
        if isinstance(state, Mapping):
            return state.get(step_clock)
        if isinstance(state, list) and state and isinstance(state[0], Mapping):
            return state[0].get(step_clock)
        return None

    def _clock_timestamp_from_observation(self, episode: Mapping[str, Any], observation: Mapping[str, Any]) -> float | None:
        return self._payload_timestamp_sec(self._clock_payload_from_observation(episode, observation))

    def _payload_timestamp_sec(self, payload: Any) -> float | None:
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if not isinstance(payload, Mapping):
            return None
        timestamp = payload.get('timestamp')
        if timestamp is not None:
            return float(timestamp)
        log_time_ns = payload.get('log_time_ns')
        if log_time_ns is not None:
            return float(log_time_ns) / 1000000000.0
        publish_time_ns = payload.get('publish_time_ns')
        if publish_time_ns is not None:
            return float(publish_time_ns) / 1000000000.0
        return None

    def _load_observation(self, episode: Mapping[str, Any], step: Mapping[str, Any]) -> Dict[str, Any]:
        obs = step.get('observation', {})
        loaded = self._load_non_state_observation(episode, step)
        clock_timestamp = self._clock_timestamp_from_observation(episode, loaded)
        loaded['state'] = self._load_state(episode, obs.get('state'), clock_timestamp=clock_timestamp)
        return loaded

    def _load_non_state_observation(self, episode: Mapping[str, Any], step: Mapping[str, Any]) -> Dict[str, Any]:
        obs = step.get('observation', {})
        loaded: Dict[str, Any] = {}
        cameras = {}
        for (camera_name, ref) in obs.get('cameras', {}).items():
            cameras[camera_name] = self._resize_camera_payload(self._read_with_context(episode, f'camera.{camera_name}', sample_ref=None, step_index=None, index=ref.get('index'), timestamp=ref.get('timestamp')))
        loaded['cameras'] = cameras
        tactile = {}
        for (sensor_name, ref) in obs.get('tactile', {}).items():
            tactile[sensor_name] = self._read_with_context(episode, f'tactile.{sensor_name}', sample_ref=None, step_index=None, index=ref.get('index'), index_range=ref.get('index_range'), timestamp=ref.get('timestamp'), time_range=ref.get('time_range'))
        loaded['tactile'] = tactile
        return loaded

    def _attach_camera_history(self, episode: Mapping[str, Any], sample_ref: SampleIndex, observation: Mapping[str, Any]) -> Dict[str, Any]:
        if self.camera_history_name is None:
            return dict(observation)
        loaded = dict(observation)
        loaded['camera_history'] = {self.camera_history_name: self._load_camera_history(episode, sample_ref)}
        return loaded

    def _load_camera_history(self, episode: Mapping[str, Any], sample_ref: SampleIndex) -> List[Any | None]:
        if self.camera_history_name is None or self.camera_history_frame_offsets is None:
            return []
        history: List[Any | None] = []
        for offset in self.camera_history_frame_offsets:
            step_index = sample_ref.step_index + offset
            if step_index < 0:
                history.append(None)
                continue
            step = self._materialize_step(episode, step_index)
            ref = step.get('observation', {}).get('cameras', {}).get(self.camera_history_name)
            if not isinstance(ref, Mapping) or (ref.get('index') is None and ref.get('timestamp') is None):
                history.append(None)
                continue
            history.append(self._resize_camera_payload(self._read_with_context(episode, f'camera.{self.camera_history_name}', sample_ref=sample_ref, step_index=step_index, index=ref.get('index'), timestamp=ref.get('timestamp'))))
        return history

    def _apply_counterfactual_camera_history(self, recipient_ref: SampleIndex, observation: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
        camera = str(spec['history_camera'])
        donor_episode_index = int(spec['donor_episode_index'])
        donor_episode = self.episodes[donor_episode_index]
        histories = dict(observation.get('camera_history') or {})
        history = list(histories.get(camera) or [])
        if len(history) != len(self.camera_history_frame_offsets or ()):
            raise ValueError('Counterfactual recipient history does not match configured slot count')
        replaced_slots = []
        for replacement in spec['history_replacements']:
            slot = int(replacement['slot_index'])
            offset = int(replacement['history_offset'])
            recipient_step = int(replacement['recipient_step'])
            donor_step = int(replacement['donor_step'])
            if int((self.camera_history_frame_offsets or ())[slot]) != offset or recipient_ref.step_index + offset != recipient_step:
                raise ValueError('Counterfactual history replacement drifted from its recipient anchor')
            donor_step_record = self._materialize_step(donor_episode, donor_step)
            ref = donor_step_record.get('observation', {}).get('cameras', {}).get(camera)
            if not isinstance(ref, Mapping) or (ref.get('index') is None and ref.get('timestamp') is None):
                raise ValueError(f"Counterfactual donor {spec['donor_episode_id']!r} has no {camera} frame at {donor_step}")
            donor_ref = SampleIndex(episode_index=donor_episode_index, step_index=donor_step)
            history[slot] = self._resize_camera_payload(self._read_with_context(donor_episode, f'camera.{camera}', sample_ref=donor_ref, step_index=donor_step, index=ref.get('index'), timestamp=ref.get('timestamp')))
            replaced_slots.append(slot)
        if not replaced_slots:
            raise ValueError('Counterfactual history intervention replaced no slots')
        histories[camera] = history
        loaded = dict(observation)
        loaded['camera_history'] = histories
        return loaded

    def _load_observation_window(self, episode: Mapping[str, Any], sample_ref: SampleIndex) -> Dict[str, Any]:
        steps = [self._materialize_step(episode, step_index) for step_index in range(sample_ref.step_index, sample_ref.step_index + sample_ref.window_size)]
        first_obs = steps[0].get('observation', {})
        loaded: Dict[str, Any] = {}
        cameras = {}
        camera_offsets = self._camera_window_offsets(sample_ref.window_size)
        if len(camera_offsets) != sample_ref.window_size and self._state_refs_use_clock_timestamps(steps):
            raise ValueError('camera_window_frame_offsets cannot be used when state refs depend on camera timestamp_from')
        for camera_name in first_obs.get('cameras', {}):
            all_refs = [step.get('observation', {}).get('cameras', {}).get(camera_name, {}) for step in steps]
            refs = [all_refs[offset] for offset in camera_offsets]
            indices = [int(ref['index']) for ref in refs if ref.get('index') is not None]
            if len(indices) == len(refs) and indices == list(range(indices[0], indices[0] + len(indices))):
                cameras[camera_name] = self._resize_camera_payload(self._read_with_context(episode, f'camera.{camera_name}', sample_ref=sample_ref, index_range=[indices[0], indices[-1] + 1]))
            else:
                cameras[camera_name] = [self._resize_camera_payload(self._read_with_context(episode, f'camera.{camera_name}', sample_ref=sample_ref, step_index=sample_ref.step_index + offset, index=ref.get('index'), timestamp=ref.get('timestamp'))) for (offset, ref) in zip(camera_offsets, refs)]
        loaded['cameras'] = cameras
        loaded['tactile'] = [{sensor_name: self._read_with_context(episode, f'tactile.{sensor_name}', sample_ref=sample_ref, step_index=sample_ref.step_index + offset, index=ref.get('index'), index_range=ref.get('index_range'), timestamp=ref.get('timestamp'), time_range=ref.get('time_range'), interpolate=bool(ref.get('interpolate', False))) for (sensor_name, ref) in step.get('observation', {}).get('tactile', {}).items()} for (offset, step) in enumerate(steps)]
        clock_payloads = self._clock_payload_from_observation(episode, loaded)
        state_refs = [step.get('observation', {}).get('state') for step in steps]
        if self._can_batch_state_window(state_refs):
            loaded['state'] = self._load_state_window_batched(episode, state_refs, sample_ref=sample_ref)
        else:
            loaded['state'] = []
            for (offset, step) in enumerate(steps):
                payload = clock_payloads[min(offset, len(clock_payloads) - 1)] if isinstance(clock_payloads, list) and clock_payloads else clock_payloads
                loaded['state'].append(self._load_state(episode, step.get('observation', {}).get('state'), clock_timestamp=self._payload_timestamp_sec(payload)))
        return loaded

    def _can_batch_state_window(self, state_refs: List[Any]) -> bool:
        if not self.batch_state_window or not state_refs:
            return False
        for state_ref in state_refs:
            if not isinstance(state_ref, Mapping) or 'streams' not in state_ref:
                return False
            streams = state_ref.get('streams') or {}
            if not isinstance(streams, Mapping):
                return False
            for ref in streams.values():
                if not isinstance(ref, Mapping):
                    return False
                if ref.get('index') is None:
                    return False
                if any((ref.get(key) is not None for key in ('timestamp', 'timestamp_from', 'index_range', 'time_range'))):
                    return False
                if bool(ref.get('interpolate', False)):
                    return False
        return True

    def _load_state_window_batched(self, episode: Mapping[str, Any], state_refs: List[Any], *, sample_ref: SampleIndex) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = [dict() for _ in state_refs]
        grouped: Dict[str, List[tuple[int, int, Mapping[str, Any]]]] = {}
        for (offset, state_ref) in enumerate(state_refs):
            for (stream_key, ref) in (state_ref.get('streams') or {}).items():
                grouped.setdefault(stream_key, []).append((offset, int(ref['index']), ref))
        for (stream_key, items) in grouped.items():
            indices = [index for (_, index, _) in items]
            payloads = self._read_many_with_context(episode, stream_key, sample_ref=sample_ref, indices=indices)
            for ((offset, _, ref), payload) in zip(items, payloads):
                outputs[offset][stream_key] = self._apply_state_ref_values(payload, ref)
        return outputs

    def _camera_window_offsets(self, window_size: int) -> tuple[int, ...]:
        if self.camera_window_frame_offsets is None:
            return tuple(range(window_size))
        offsets = []
        for offset in self.camera_window_frame_offsets:
            idx = offset if offset >= 0 else window_size + offset
            if idx < 0 or idx >= window_size:
                raise IndexError(f'camera_window_frame_offsets contains {offset} outside window_size={window_size}')
            offsets.append(idx)
        return tuple(offsets)

    def _state_refs_use_clock_timestamps(self, steps: List[Mapping[str, Any]]) -> bool:
        for step in steps:
            state_ref = (step.get('observation') or {}).get('state')
            if isinstance(state_ref, Mapping) and 'streams' in state_ref:
                refs = state_ref.get('streams', {}).values()
            elif isinstance(state_ref, Mapping):
                refs = [state_ref]
            else:
                refs = []
            for ref in refs:
                if isinstance(ref, Mapping) and ref.get('timestamp_from') is not None:
                    return True
        return False

    def _normalize_image_size(self, image_size: Optional[List[int]]) -> tuple[int, int] | None:
        if image_size is None:
            return None
        values = list(image_size)
        if len(values) != 2:
            raise ValueError(f'camera_payload_image_size must be [height, width], got {image_size}')
        (height, width) = (int(values[0]), int(values[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f'camera_payload_image_size must be positive, got {image_size}')
        return (height, width)

    def _resize_camera_payload(self, payload: Any) -> Any:
        if self.camera_payload_image_size is None:
            return payload
        if isinstance(payload, list):
            return [self._resize_camera_payload(item) for item in payload]
        if not isinstance(payload, Mapping) or payload.get('array') is None:
            return payload
        import numpy as np
        from PIL import Image
        (height, width) = self.camera_payload_image_size
        array = self._as_uint8_array(payload['array'])
        if array.shape[:2] == (height, width):
            return payload
        image = Image.fromarray(array).convert('RGB').resize((width, height))
        out = dict(payload)
        out['array'] = np.asarray(image, dtype=np.uint8).copy()
        out['original_shape'] = list(array.shape)
        return out

    def _as_uint8_array(self, array: Any) -> Any:
        import numpy as np
        out = np.asarray(array)
        if out.dtype == np.uint8:
            return out
        return out.astype(np.uint8)

    def _load_state(self, episode: Mapping[str, Any], state_ref: Any, *, clock_timestamp: float | None=None) -> Optional[Any]:
        if state_ref is None:
            return None
        elif isinstance(state_ref, Mapping) and 'streams' in state_ref:
            state = {}
            for (stream_key, ref) in state_ref.get('streams', {}).items():
                timestamp = ref.get('timestamp')
                index = ref.get('index')
                if ref.get('timestamp_from') is not None and clock_timestamp is not None:
                    timestamp = clock_timestamp
                    index = None
                payload = self._read_with_context(episode, stream_key, sample_ref=None, step_index=None, index=index, index_range=ref.get('index_range'), timestamp=timestamp, time_range=ref.get('time_range'), interpolate=bool(ref.get('interpolate', False)))
                state[stream_key] = self._apply_state_ref_values(payload, ref)
            return state
        else:
            timestamp = state_ref.get('timestamp')
            index = state_ref.get('index')
            if state_ref.get('timestamp_from') is not None and clock_timestamp is not None:
                timestamp = clock_timestamp
                index = None
            payload = self._read_with_context(episode, 'robot_state', sample_ref=None, step_index=None, index=index, index_range=state_ref.get('index_range'), timestamp=timestamp, time_range=state_ref.get('time_range'), interpolate=bool(state_ref.get('interpolate', False)))
            return self._apply_state_ref_values(payload, state_ref)

    def _load_action(self, episode: Mapping[str, Any], step: Mapping[str, Any]) -> Optional[Any]:
        if str(self.action_config.get('source', 'stored')).lower() == 'none':
            return None
        action_ref = step.get('action')
        if action_ref is None:
            return None
        horizon = self._stored_action_horizon()
        action_index = action_ref.get('index')
        if horizon > 1 and action_index is not None:
            return self._load_stored_action_window(episode, action_ref, start_index=int(action_index), horizon=horizon)
        payload = self._read_with_context(episode, 'action', sample_ref=None, step_index=None, index=action_ref.get('index'), index_range=action_ref.get('index_range'), timestamp=action_ref.get('timestamp'), time_range=action_ref.get('time_range'))
        if not self._stored_action_payload_is_valid(payload, validity_column=self._stored_action_validity_column(episode)):
            raise ValueError('Stored action row is marked invalid by the canonical schema')
        return payload

    def _stored_action_horizon(self) -> int:
        if self.representation_transform is not None:
            return int(self.state_action_config['horizon'])
        cfg = self.action_config
        if str(cfg.get('source', 'stored')).lower() != 'stored':
            return 1
        value = cfg.get('stored_horizon', cfg.get('action_horizon', 1))
        return max(1, int(value))

    def _stored_action_step_interval(self) -> int:
        if self.representation_transform is not None:
            return 1
        value = self.action_config.get('stored_step_interval', 1)
        return max(1, int(value))

    def _load_stored_action_window(self, episode: Mapping[str, Any], action_ref: Mapping[str, Any], *, start_index: int, horizon: int) -> Dict[str, Any]:
        interval = self._stored_action_step_interval()
        read_horizon = horizon + 1 if self._stored_transform is not None else horizon
        indices = [start_index + offset * interval for offset in range(read_horizon)]
        payloads = self._read_many_with_context(episode, 'action', sample_ref=None, indices=indices)
        return self._merge_stored_action_payloads(payloads, episode=episode, action_ref=action_ref, requested_indices=indices)

    def _merge_stored_action_payloads(self, payloads: List[Any], *, episode: Mapping[str, Any] | None=None, action_ref: Mapping[str, Any], requested_indices: List[int]) -> Dict[str, Any]:
        rows = [dict(payload) if isinstance(payload, Mapping) else {'values': payload} for payload in payloads]
        validity_column = self._stored_action_validity_column(episode)
        if any((not self._stored_action_payload_is_valid(row, validity_column=validity_column) for row in rows)):
            raise ValueError('Stored action window contains a row marked invalid by the canonical schema')
        values = [self._stored_action_values(row) for row in rows]
        if self._stored_transform is not None:
            values = self._absolute_tcp_to_consecutive_relative_tcp(values)
            rows = rows[:len(values)]
            requested_indices = requested_indices[:len(values)]
        values = self._normalize_stored_action_values(values, episode=episode)
        mask = [[1] * len(row) if isinstance(row, list) else 1 for row in values]
        first = dict(rows[0]) if rows else {}
        return {**first, 'type': first.get('type') or 'action', 'index': int(action_ref.get('index', requested_indices[0] if requested_indices else 0)), 'indices': [int(row.get('index') if isinstance(row, Mapping) and row.get('index') is not None else idx) for (row, idx) in zip(rows, requested_indices, strict=True)], 'requested_indices': [int(index) for index in requested_indices], 'timestamp': [row.get('timestamp') for row in rows], 'valid': [row.get('valid') for row in rows], 'values': values, 'mask': mask, 'source': 'stored'}

    @staticmethod
    def _resolve_legacy_stored_transform(config: Mapping[str, Any]) -> dict[str, str] | None:
        value = config.get('stored_transform')
        if value is None:
            return None
        if isinstance(value, str):
            transform = {'type': value}
        elif isinstance(value, Mapping):
            transform = dict(value)
        else:
            raise ValueError('stored_transform must be a string or mapping')
        transform_type = str(transform.get('type', ''))
        if transform_type != 'absolute_tcp_to_consecutive_relative_tcp':
            raise ValueError(f'Unsupported stored_transform type: {transform_type!r}')
        quaternion_order = str(transform.get('quaternion_order', 'wxyz'))
        if quaternion_order != 'wxyz':
            raise ValueError("absolute TCP stored_transform requires quaternion_order='wxyz'")
        gripper = str(transform.get('gripper', 'absolute_next_frame_value'))
        if gripper != 'absolute_next_frame_value':
            raise ValueError("absolute TCP stored_transform requires gripper='absolute_next_frame_value'")
        return {'type': transform_type, 'quaternion_order': quaternion_order, 'gripper': gripper}

    @classmethod
    def _absolute_tcp_to_consecutive_relative_tcp(cls, rows: List[Any]) -> List[List[float]]:
        if len(rows) < 2:
            raise ValueError('absolute TCP stored_transform requires H+1 action rows')
        arrays = [np.asarray(row, dtype=np.float64) for row in rows]
        shape = arrays[0].shape
        if len(shape) != 1 or shape[0] == 0 or shape[0] % 8 != 0:
            raise ValueError('absolute TCP action rows must be flat repeated [xyz,wxyz,gripper] blocks')
        if any((row.shape != shape or not np.isfinite(row).all() for row in arrays)):
            raise ValueError('absolute TCP action rows must share one finite flat shape')
        output: List[List[float]] = []
        for (current, following) in pairwise(arrays):
            row: list[float] = []
            for start in range(0, shape[0], 8):
                position = following[start:start + 3] - current[start:start + 3]
                rotation = cls._legacy_wxyz_multiply(following[start + 3:start + 7], cls._legacy_wxyz_inverse(current[start + 3:start + 7]))
                row.extend(position.tolist())
                row.extend(rotation.tolist())
                row.append(float(following[start + 7]))
            output.append(row)
        return output

    @staticmethod
    def _legacy_wxyz_inverse(value: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError('absolute TCP stored_transform received an invalid quaternion')
        quaternion = quaternion / norm
        return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])

    @staticmethod
    def _legacy_wxyz_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        (w1, x1, y1, z1) = np.asarray(left, dtype=np.float64)
        (w2, x2, y2, z2) = np.asarray(right, dtype=np.float64)
        result = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        norm = float(np.linalg.norm(result))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError('absolute TCP stored_transform produced an invalid quaternion')
        result /= norm
        if result[0] < 0.0:
            result = -result
        return result

    def _stored_action_values(self, payload: Mapping[str, Any]) -> Any:
        values = payload.get('values')
        if values is None:
            message = payload.get('message')
            if isinstance(message, Mapping):
                values = message.get('values')
        if values is None:
            return []
        if hasattr(values, 'tolist'):
            values = values.tolist()
        if isinstance(values, tuple):
            values = list(values)
        return values

    def _normalize_stored_action_values(self, values: List[Any], *, episode: Mapping[str, Any] | None=None) -> List[Any]:
        if self.representation_transform is not None:
            return values
        norm_cfg = self.action_config.get('stored_normalization') or self.action_config.get('normalization')
        if not isinstance(norm_cfg, Mapping):
            return values
        mode = str(norm_cfg.get('mode', 'none')).lower()
        if mode in {'none', ''}:
            return values
        stats = self._stored_action_normalization_statistics(norm_cfg, episode=episode)
        if not isinstance(stats, Mapping):
            raise ValueError('stored action normalization requires action statistics')
        exclude_indices = {int(index) for index in norm_cfg.get('exclude_indices', [])}
        return [self._normalize_stored_action_row(row, mode=mode, stats=stats, exclude_indices=exclude_indices) for row in values]

    def _stored_action_normalization_statistics(self, norm_cfg: Mapping[str, Any], *, episode: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        source_name = self._episode_source_dataset_name(episode)
        for key in ('statistics_by_source', 'per_source_statistics', 'statistics_by_dataset'):
            source_stats = norm_cfg.get(key)
            if isinstance(source_stats, Mapping) and source_name:
                stats = source_stats.get(source_name)
                if isinstance(stats, Mapping):
                    return stats
        stats = norm_cfg.get('statistics')
        if isinstance(stats, Mapping):
            return stats
        if episode is not None:
            action_ref = ((episode.get('storage') or {}).get('streams') or {}).get('action')
            action_spec = action_ref.get('action_spec') if isinstance(action_ref, Mapping) else None
            stats = action_spec.get('normalization_statistics') if isinstance(action_spec, Mapping) else None
            if isinstance(stats, Mapping):
                return stats
        schema = (self.manifest.get('extra') or {}).get('schema') or {}
        action_schema = schema.get('action') if isinstance(schema, Mapping) else None
        stats = action_schema.get('normalization_statistics') if isinstance(action_schema, Mapping) else None
        if isinstance(stats, Mapping):
            return stats
        stats = ((self.manifest.get('extra') or {}).get('statistics') or {}).get('action')
        return stats if isinstance(stats, Mapping) else None

    def _episode_source_dataset_name(self, episode: Mapping[str, Any] | None) -> str | None:
        if not isinstance(episode, Mapping):
            return None
        source_info = episode.get('source_info') or {}
        if isinstance(source_info, Mapping):
            source_dataset = source_info.get('source_dataset') or source_info.get('raw_dataset')
            if source_dataset:
                return str(source_dataset)
        extra = episode.get('extra') or {}
        if isinstance(extra, Mapping):
            source = extra.get('source')
            if isinstance(source, Mapping):
                source_dataset = source.get('source_dataset') or source.get('raw_dataset')
                if source_dataset:
                    return str(source_dataset)
        return None

    def _normalize_stored_action_row(self, row: Any, *, mode: str, stats: Mapping[str, Any], exclude_indices: set[int] | None=None) -> Any:
        if not isinstance(row, list):
            return row
        exclude_indices = exclude_indices or set()
        if mode == 'min_max':
            mins = list(stats.get('min') or [])
            maxs = list(stats.get('max') or [])
            out = []
            for (idx, value) in enumerate(row):
                if idx in exclude_indices:
                    out.append(float(value))
                    continue
                min_value = float(mins[idx]) if idx < len(mins) else 0.0
                max_value = float(maxs[idx]) if idx < len(maxs) else 0.0
                if max_value == min_value:
                    out.append(0.0)
                else:
                    out.append(2.0 * (float(value) - min_value) / (max_value - min_value) - 1.0)
            return out
        if mode == 'mean_std':
            means = list(stats.get('mean') or [])
            stds = list(stats.get('std') or [])
            out = []
            for (idx, value) in enumerate(row):
                if idx in exclude_indices:
                    out.append(float(value))
                    continue
                mean_value = float(means[idx]) if idx < len(means) else 0.0
                std_value = float(stds[idx]) if idx < len(stds) else 0.0
                out.append(float(value) if std_value == 0.0 else (float(value) - mean_value) / std_value)
            return out
        raise ValueError(f'Unsupported stored action normalization mode: {mode}')

    def _apply_state_ref_values(self, payload: Any, ref: Mapping[str, Any]) -> Any:
        values = ref.get('values') if isinstance(ref, Mapping) else None
        if not isinstance(values, Mapping) or not isinstance(payload, Mapping):
            return payload
        out = dict(payload)
        message = dict(out.get('message') or {}) if isinstance(out.get('message'), Mapping) else None
        for (key, value) in values.items():
            if value is None:
                continue
            numeric = float(value) if isinstance(value, (int, float)) else value
            out[str(key)] = numeric
            if message is not None:
                message[str(key)] = numeric
        if message is not None:
            out['message'] = message
        return out

    def _read_many_with_context(self, episode: Mapping[str, Any], stream_key: str, *, sample_ref: SampleIndex | None, indices: List[int]) -> List[Any]:
        try:
            return self.reader.read_many(episode, stream_key, indices=indices)
        except Exception as exc:
            context = self._sample_context(episode, sample_ref)
            context.update({'stream_key': stream_key, 'indices': list(indices)})
            raise CanonicalSampleLoadError(f'Failed to read canonical stream `{stream_key}` batch', context=context, cause=exc) from exc

    def _read_with_context(self, episode: Mapping[str, Any], stream_key: str, *, sample_ref: SampleIndex | None, step_index: int | None=None, index: Any=None, index_range: Any=None, timestamp: Any=None, time_range: Any=None, interpolate: bool=False) -> Any:
        try:
            return self.reader.read(episode, stream_key, index=index, index_range=index_range, timestamp=timestamp, time_range=time_range, interpolate=interpolate)
        except Exception as exc:
            context = self._sample_context(episode, sample_ref, step_index=step_index)
            context.update({'stream_key': stream_key, 'index': index, 'index_range': index_range, 'timestamp': timestamp, 'time_range': time_range})
            raise CanonicalSampleLoadError(f'Failed to read canonical stream `{stream_key}`', context=context, cause=exc) from exc

    def _sample_context(self, episode: Mapping[str, Any], sample_ref: SampleIndex | None, *, step_index: int | None=None) -> Dict[str, Any]:
        source_info = episode.get('source_info') or {}
        extra = episode.get('extra') or {}
        context: Dict[str, Any] = {'episode_id': episode.get('episode_id'), 'raw_path': source_info.get('raw_path'), 'sidecar': extra.get('sidecar_manifest_path')}
        if sample_ref is not None:
            context.update({'episode_index': sample_ref.episode_index, 'step_index': sample_ref.step_index if step_index is None else step_index, 'window_start': sample_ref.step_index, 'window_size': sample_ref.window_size})
        elif step_index is not None:
            context['step_index'] = step_index
        return context

    def _error_context(self, idx: int, exc: BaseException) -> Dict[str, Any]:
        context = {}
        if isinstance(exc, CanonicalSampleLoadError):
            context.update(exc.context)
        try:
            sample_ref = self._sample_ref_from_index(idx)
            episode = self.episodes[sample_ref.episode_index]
            context = {**self._sample_context(episode, sample_ref), **context}
        except Exception:
            pass
        context['global_idx'] = idx
        context['error_type'] = type(exc.__cause__ if isinstance(exc, CanonicalSampleLoadError) else exc).__name__
        context['error'] = str(exc.__cause__ if isinstance(exc, CanonicalSampleLoadError) else exc)
        return context

    def _format_error_context(self, context: Mapping[str, Any]) -> str:
        keys = ['global_idx', 'episode_id', 'episode_index', 'step_index', 'window_start', 'window_size', 'stream_key', 'index', 'index_range', 'raw_path', 'sidecar', 'error_type', 'error']
        return ', '.join((f'{key}={context.get(key)!r}' for key in keys if context.get(key) is not None))

    def _resolve_sub_task(self, episode: Mapping[str, Any], step: Mapping[str, Any], step_index: int) -> Optional[Dict[str, Any]]:
        sub_tasks = episode.get('sub_tasks', [])
        sub_task_id = step.get('sub_task_id')
        if sub_task_id is not None:
            for sub_task in sub_tasks:
                if sub_task.get('sub_task_id') == sub_task_id:
                    return sub_task
        for sub_task in sub_tasks:
            start = sub_task.get('start_step')
            end = sub_task.get('end_step')
            if start is not None and end is not None and (int(start) <= step_index <= int(end)):
                return sub_task
        return None

    def _merge_meta_data(self, episode_meta: Mapping[str, Any], step_meta: Mapping[str, Any]) -> Dict[str, Any]:
        merged = dict(episode_meta or {})
        for (key, value) in (step_meta or {}).items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
            else:
                merged[key] = value
        return merged

    def sample_value_weight(self, sample_index: int, *, key: str='value_weight', default: float=1.0) -> float:
        sample_ref = self._sample_ref_from_index(int(sample_index))
        episode = self.episodes[sample_ref.episode_index]
        value = self._value_weight_for_step(episode, sample_ref.step_index, key=key, default=default)
        return self._coerce_value_weight(value, context=self._sample_context(episode, sample_ref))

    def _value_weight_for_step(self, episode: Mapping[str, Any], step_index: int, *, key: str='value_weight', default: float=1.0) -> Any:
        episode_meta = episode.get('meta_data', {}) if isinstance(episode.get('meta_data', {}), Mapping) else {}
        value = episode_meta.get(key, default)
        steps = episode.get('steps') or []
        if steps:
            step = steps[int(step_index)]
            step_meta = step.get('meta_data', {}) if isinstance(step.get('meta_data', {}), Mapping) else {}
            return step_meta.get(key, value)
        table_value = self._step_table_value_or_none(episode, key, int(step_index))
        if table_value is not None:
            return table_value
        for override in episode.get('step_overrides', []) or []:
            if int(override.get('step_index', -1)) == int(step_index):
                override_meta = override.get('meta_data', {}) if isinstance(override.get('meta_data', {}), Mapping) else {}
                return override_meta.get(key, value)
        return value

    @staticmethod
    def _coerce_value_weight(value: Any, *, context: Mapping[str, Any]) -> float:
        try:
            weight = float(value)
        except (TypeError, ValueError) as error:
            raise CanonicalSampleLoadError(f'Canonical sample value_weight must be numeric, got {value!r}.', context=context, cause=error) from error
        if not math.isfinite(weight) or weight < 0:
            raise CanonicalSampleLoadError(f'Canonical sample value_weight must be finite and non-negative, got {weight}.', context=context, cause=ValueError('invalid value_weight'))
        return weight

    def _validate_minimal_schema(self) -> None:
        if 'embodiment' not in self.manifest:
            raise ValueError('Canonical dataset manifest is missing `embodiment`')
        if 'episodes' not in self.manifest:
            raise ValueError('Canonical dataset manifest is missing `episodes`')
        for ep in self.episodes:
            storage = ep.get('storage', {})
            streams = storage.get('streams', {})
            for stream_key in ep.get('timeline', {}).get('stream_hz', {}):
                if stream_key not in streams:
                    raise ValueError(f'timeline.stream_hz references missing storage stream `{stream_key}`')
            for step in ep.get('steps', []):
                obs = step.get('observation', {})
                if 'language' in obs:
                    raise ValueError('Step observation must not contain `language`; build it in transforms')
                for camera_name in obs.get('cameras', {}):
                    key = f'camera.{camera_name}'
                    if key not in streams:
                        raise ValueError(f'Observation camera `{camera_name}` references missing stream `{key}`')
                state_ref = obs.get('state')
                if state_ref is not None:
                    if isinstance(state_ref, Mapping) and 'streams' in state_ref:
                        for stream_key in state_ref.get('streams', {}):
                            if stream_key not in streams:
                                raise ValueError(f'Observation state references missing stream `{stream_key}`')
                    elif 'robot_state' not in streams:
                        raise ValueError('Observation state references missing `robot_state` stream')
                if step.get('action') is not None and 'action' not in streams:
                    raise ValueError('Step action references missing `action` stream')
            for sub_task in ep.get('sub_tasks', []):
                sub_goal = sub_task.get('sub_goal') or {}
                reps = sub_goal.get('representations', {})
                if 'language' in reps:
                    raise ValueError('sub_goal.representations must not contain `language`')
                if 'object_state' in reps:
                    raise ValueError('sub_goal.representations must not contain `object_state`')
                visual = reps.get('visual')
                if visual and visual.get('key') not in streams:
                    raise ValueError(f"sub_goal visual references missing stream `{visual.get('key')}`")
                latent = reps.get('latent')
                if latent and latent.get('key') not in streams:
                    raise ValueError(f"sub_goal latent references missing stream `{latent.get('key')}`")
