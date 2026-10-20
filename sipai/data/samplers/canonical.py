from __future__ import annotations
import hashlib
import itertools
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler
from sipai.data.datasets import CanonicalEmbodiedDataset

def _deterministic_safe_hash(input_tuple: tuple[Any, ...]) -> int:
    tuple_string = repr(input_tuple).encode('utf-8')
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)
    return int(sha256.hexdigest(), 16) & 340282366920938463463374607431768211455


def _source_dataset_name(episode: Mapping[str, Any]) -> str:
    info = episode.get('source_info') or {}
    return str(info.get('sampling_source') or info.get('source_dataset') or info.get('raw_dataset') or '__unknown__')

class MemoryEventBalancedSampler(Sampler[int]):

    def __init__(self, dataset: CanonicalEmbodiedDataset, *, event_ratio: float=0.4, guard_ratio: float=0.0, counterfactual_pair_ratio: float=0.0, counterfactual_pair_ratio_overrides: Mapping[str, Mapping[str, float]] | None=None, counterfactual_group_ratio: float=0.0, counterfactual_group_ratio_overrides: Mapping[str, float] | None=None, counterfactual_group_tasks: Sequence[str] | None=None, counterfactual_action_mix: Mapping[str, Mapping[str, float]] | None=None, task_target_mix: Mapping[str, Mapping[str, float]] | None=None, task_final_mix: Mapping[str, float] | None=None, epoch_length: int | None=None, seed: int=42, rank: int | None=None, world_size: int | None=None, batch_size: int=1, drop_last: bool=False, locality_aware: bool=True, locality_shuffle_within_episode: bool=True) -> None:
        if not isinstance(dataset, CanonicalEmbodiedDataset):
            raise TypeError('MemoryEventBalancedSampler requires CanonicalEmbodiedDataset')
        self.dataset = dataset
        self.event_ratio = float(event_ratio)
        self.guard_ratio = float(guard_ratio)
        self.counterfactual_pair_ratio = float(counterfactual_pair_ratio)
        self.counterfactual_pair_ratio_overrides = {str(task): {str(pool): float(ratio) for (pool, ratio) in overrides.items()} for (task, overrides) in (counterfactual_pair_ratio_overrides or {}).items()}
        self.counterfactual_group_ratio = float(counterfactual_group_ratio)
        self.counterfactual_group_ratio_overrides = {str(task): float(ratio) for (task, ratio) in (counterfactual_group_ratio_overrides or {}).items()}
        self.counterfactual_group_tasks = tuple(sorted((str(task) for task in counterfactual_group_tasks or ())))
        self.counterfactual_action_mix = {str(task): {str(variant): float(ratio) for (variant, ratio) in mix.items()} for (task, mix) in (counterfactual_action_mix or {}).items()}
        self.task_target_mix = {str(task): {str(pool): float(ratio) for (pool, ratio) in mix.items()} for (task, mix) in (task_target_mix or {}).items()}
        self.task_final_mix = {str(task): float(ratio) for (task, ratio) in (task_final_mix or {}).items()}
        if not math.isfinite(self.event_ratio) or self.event_ratio < 0.0:
            raise ValueError('event_ratio must be finite and non-negative.')
        if not math.isfinite(self.guard_ratio) or self.guard_ratio < 0.0:
            raise ValueError('guard_ratio must be finite and non-negative.')
        if not math.isfinite(self.counterfactual_pair_ratio) or not 0.0 <= self.counterfactual_pair_ratio <= 1.0:
            raise ValueError('counterfactual_pair_ratio must lie in [0, 1].')
        for (task, overrides) in self.counterfactual_pair_ratio_overrides.items():
            for (pool, ratio) in overrides.items():
                if pool not in {'memory_decision', 'completion_guard'}:
                    raise ValueError(f'Unknown counterfactual pair-ratio pool {task}/{pool}')
                if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                    raise ValueError(f'Counterfactual pair ratio override {task}/{pool} must lie in [0, 1]')
        if not math.isfinite(self.counterfactual_group_ratio) or not 0.0 <= self.counterfactual_group_ratio <= 1.0:
            raise ValueError('counterfactual_group_ratio must lie in [0, 1].')
        if any((not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0 for ratio in self.counterfactual_group_ratio_overrides.values())):
            raise ValueError('counterfactual_group_ratio_overrides values must lie in [0, 1].')
        if (self.counterfactual_group_ratio > 0.0 or self.counterfactual_group_ratio_overrides) and (not self.counterfactual_group_tasks):
            raise ValueError('counterfactual batch grouping requires at least one configured task.')
        if not 0.0 < self.event_ratio + self.guard_ratio < 1.0:
            raise ValueError('event_ratio + guard_ratio must lie strictly between 0 and 1.')
        self.seed = int(seed)
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.locality_aware = bool(locality_aware)
        self.locality_shuffle_within_episode = bool(locality_shuffle_within_episode)
        if rank is None or world_size is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1:
            raise ValueError(f'world_size must be >= 1, got {self.world_size}')
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f'rank must be in [0, {self.world_size}), got {self.rank}')
        self._tasks = self._build_task_pools()
        if not self._tasks:
            raise ValueError('MemoryEventBalancedSampler found no episode.extra.memory_events metadata. Build the derived memory-event manifest before training.')
        unknown_override_tasks = set(self.counterfactual_pair_ratio_overrides).difference(self._tasks)
        if unknown_override_tasks:
            raise ValueError(f'Unknown counterfactual pair-ratio tasks: {sorted(unknown_override_tasks)}')
        unknown_group_override_tasks = set(self.counterfactual_group_ratio_overrides).difference(self.counterfactual_group_tasks)
        if unknown_group_override_tasks:
            raise ValueError(f'counterfactual group-ratio overrides must name configured group tasks: {sorted(unknown_group_override_tasks)}')
        self._event_task_names = tuple(sorted((task for (task, pool) in self._tasks.items() if pool['event_episodes'])))
        self._guard_task_names = tuple(sorted((task for (task, pool) in self._tasks.items() if pool['guard_episodes'])))
        self._validate_counterfactual_action_mix()
        self._counterfactual_action_hierarchy_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._reserve_counterfactual_action_anchors()
        if self.event_ratio > 0 and (not self._event_task_names):
            raise ValueError('MemoryEventBalancedSampler found no trainable memory-decision windows.')
        if self.guard_ratio > 0 and (not self._guard_task_names):
            raise ValueError('MemoryEventBalancedSampler found no trainable completion-guard windows.')
        self._base_indices = [index for pool in self._tasks.values() for episode in pool['episodes'] for index in episode['base_indices']]
        if not self._base_indices:
            raise ValueError('MemoryEventBalancedSampler found no observable base sample anchors.')
        self._base_indices_by_task = {task: [index for episode in pool['episodes'] for index in episode['base_indices']] for (task, pool) in self._tasks.items()}
        self._global_batch_size = self.batch_size * self.world_size
        special_ratio = self.event_ratio + self.guard_ratio
        minimum_length = math.ceil(len(self._base_indices) / (1.0 - special_ratio))
        self._global_len = math.ceil(minimum_length / self._global_batch_size) * self._global_batch_size
        if epoch_length is not None and int(epoch_length) != self._global_len:
            raise ValueError(f'MemoryEventBalancedSampler epoch_length is derived from complete observable-base coverage and decision/guard overlay ratios; expected {self._global_len}, got {int(epoch_length)}.')
        self._special_count = self._global_len - len(self._base_indices)
        self._event_count = round(self._special_count * self.event_ratio / special_ratio)
        self._guard_count = self._special_count - self._event_count
        self._validate_task_target_mix()
        self._validate_task_final_mix()
        self._final_pool_task_counts: dict[str, Counter[str]] | None = None
        if self.task_final_mix:
            self._configure_final_task_mix()
        self._scheduled_base_count = self._global_len - self._event_count - self._guard_count
        self.epoch = 0
        self._num_samples = self._global_len // self.world_size
        self._set_epoch_state()

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._set_epoch_state()

    def _set_epoch_state(self) -> None:
        self._set_special_task_schedules()
        self._set_counterfactual_action_schedules()
        self._set_base_permutation()
        self._counterfactual_grouped_batch_starts: list[int] = []
        self._counterfactual_grouped_batches: list[dict[str, Any]] = []
        self._counterfactual_grouped_samples = 0
        if self.locality_aware:
            schedule = self._build_locality_schedule()
        elif self._counterfactual_grouping_enabled():
            schedule = [self._sample_global_position(position) for position in self._global_positions_for_rank()]
        else:
            schedule = None
        if schedule is not None and self._counterfactual_grouping_enabled():
            schedule = self._apply_counterfactual_batch_grouping(schedule)
        self._local_schedule = schedule

    def _counterfactual_grouping_enabled(self) -> bool:
        return bool(self.counterfactual_group_tasks) and any((self._counterfactual_group_ratio_for_task(task) > 0.0 for task in self.counterfactual_group_tasks))

    def _counterfactual_group_ratio_for_task(self, task: str) -> float:
        return float(self.counterfactual_group_ratio_overrides.get(task, self.counterfactual_group_ratio))

    def _validate_task_target_mix(self) -> None:
        allowed = {'base', 'memory_decision', 'completion_guard'}
        for (task, mix) in self.task_target_mix.items():
            if task not in self._tasks:
                raise ValueError(f'Unknown task_target_mix task={task!r}')
            if set(mix) != allowed:
                raise ValueError(f'task_target_mix[{task!r}] must define exactly {sorted(allowed)}')
            if any((not math.isfinite(value) or value < 0.0 for value in mix.values())):
                raise ValueError(f'task_target_mix[{task!r}] contains an invalid ratio')
            if not math.isclose(sum(mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-09):
                raise ValueError(f'task_target_mix[{task!r}] ratios must sum to 1')
            if mix['base'] <= 0.0:
                raise ValueError(f'task_target_mix[{task!r}].base must be positive')

    def _validate_counterfactual_action_mix(self) -> None:
        available = getattr(self.dataset, 'counterfactual_action_pools', {})
        for (task, mix) in self.counterfactual_action_mix.items():
            if task not in self._tasks or task not in available:
                raise ValueError(f'Counterfactual-action task {task!r} has no manifest-backed virtual samples')
            expected_variants = set(available[task])
            if set(mix) != expected_variants:
                raise ValueError(f'counterfactual_action_mix[{task!r}] must define exactly {sorted(expected_variants)}')
            if any((not math.isfinite(value) or value <= 0.0 for value in mix.values())):
                raise ValueError(f'counterfactual_action_mix[{task!r}] ratios must be finite and positive')
            if not math.isclose(sum(mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-09):
                raise ValueError(f'counterfactual_action_mix[{task!r}] ratios must sum to 1')
            if task not in self._event_task_names:
                raise ValueError(f'Counterfactual-action task {task!r} has no memory-decision stream')
            if any((not available[task].get(variant) for variant in expected_variants)):
                raise ValueError(f'Counterfactual-action task {task!r} has an empty action variant')

    def _reserve_counterfactual_action_anchors(self) -> None:
        available = getattr(self.dataset, 'counterfactual_action_pools', {})
        for task in self.counterfactual_action_mix:
            positive = set(map(int, available[task].get('positive', ())))
            if not positive:
                continue
            removed = 0
            for episode in self._tasks[task]['episodes']:
                before = len(episode['base_indices'])
                episode['base_indices'] = [index for index in episode['base_indices'] if index not in positive]
                removed += before - len(episode['base_indices'])
            if removed != len(positive):
                raise ValueError(f'Counterfactual-action task {task!r} reserved {removed}/{len(positive)} positive anchors')

    def _validate_task_final_mix(self) -> None:
        if not self.task_final_mix:
            return
        configured = set(self.task_final_mix)
        available = set(self._tasks)
        if configured != available:
            raise ValueError(f'task_final_mix must define exactly every memory task; missing={sorted(available - configured)}, unknown={sorted(configured - available)}')
        if any((not math.isfinite(value) or value <= 0.0 for value in self.task_final_mix.values())):
            raise ValueError('task_final_mix ratios must be finite and positive')
        if not math.isclose(sum(self.task_final_mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-09):
            raise ValueError('task_final_mix ratios must sum to 1')

    @staticmethod
    def _apportion_counts(total: int, ratios: Mapping[str, float]) -> Counter[str]:
        ratio_sum = sum((float(ratio) for ratio in ratios.values()))
        raw = {task: int(total) * float(ratio) / ratio_sum for (task, ratio) in ratios.items()}
        counts: Counter[str] = Counter({task: math.floor(value) for (task, value) in raw.items()})
        remaining = int(total) - sum(counts.values())
        order = sorted(raw, key=lambda task: (-(raw[task] - counts[task]), task))
        for task in order[:remaining]:
            counts[task] += 1
        return counts

    def _special_task_counts(self, sample_pool: str, total: int, task_names: Sequence[str]) -> Counter[str]:
        counts: Counter[str] = Counter()
        fixed_tasks = set()
        for (task, mix) in self.task_target_mix.items():
            target = mix[sample_pool]
            if target <= 0.0:
                continue
            if task not in task_names:
                raise ValueError(f'task_target_mix[{task!r}] requests unavailable pool {sample_pool!r}')
            base_count = len(self._base_indices_by_task[task])
            counts[task] = round(base_count * target / mix['base'])
            fixed_tasks.add(task)
        if sum(counts.values()) > total:
            raise ValueError(f'task_target_mix over-allocates {sample_pool}: requested {sum(counts.values())}, available {total}')
        remaining_tasks = [task for task in task_names if task not in fixed_tasks]
        if not remaining_tasks and sum(counts.values()) < total:
            remaining_tasks = list(task_names)
        for ordinal in range(total - sum(counts.values())):
            counts[remaining_tasks[ordinal % len(remaining_tasks)]] += 1
        return counts

    def _configure_final_task_mix(self) -> None:
        legacy_event_counts = self._special_task_counts('memory_decision', self._event_count, self._event_task_names)
        legacy_guard_counts = self._special_task_counts('completion_guard', self._guard_count, self._guard_task_names)
        final_counts = self._apportion_counts(self._global_len, self.task_final_mix)
        pool_counts = {'base': Counter(), 'memory_decision': Counter(), 'completion_guard': Counter()}
        for task in sorted(self._tasks):
            baseline = {'base': len(self._base_indices_by_task[task]), 'memory_decision': legacy_event_counts[task], 'completion_guard': legacy_guard_counts[task]}
            scaled = self._apportion_counts(final_counts[task], baseline)
            for (sample_pool, count) in scaled.items():
                if count > 0 and sample_pool != 'base':
                    available_tasks = self._event_task_names if sample_pool == 'memory_decision' else self._guard_task_names
                    if task not in available_tasks:
                        raise ValueError(f'task_final_mix[{task!r}] requests unavailable pool {sample_pool!r}')
                pool_counts[sample_pool][task] = count
        self._final_pool_task_counts = pool_counts
        self._event_count = sum(pool_counts['memory_decision'].values())
        self._guard_count = sum(pool_counts['completion_guard'].values())
        self._special_count = self._event_count + self._guard_count

    def _set_special_task_schedules(self) -> None:
        self._special_task_schedules: dict[str, list[str]] = {}
        self._special_task_occurrences: dict[str, list[int]] = {}
        for (sample_pool, total, task_names) in (('memory_decision', self._event_count, self._event_task_names), ('completion_guard', self._guard_count, self._guard_task_names)):
            counts = self._final_pool_task_counts[sample_pool] if self._final_pool_task_counts is not None else self._special_task_counts(sample_pool, total, task_names)
            schedule = [task for task in task_names for _ in range(counts[task])]
            random.Random(_deterministic_safe_hash(('memory_special_task_schedule', sample_pool, self.epoch, self.seed))).shuffle(schedule)
            occurrences = []
            seen: Counter[str] = Counter()
            for task in schedule:
                occurrences.append(seen[task])
                seen[task] += 1
            self._special_task_schedules[sample_pool] = schedule
            self._special_task_occurrences[sample_pool] = occurrences

    def _set_counterfactual_action_schedules(self) -> None:
        self._counterfactual_action_schedules: dict[str, list[str]] = {}
        self._counterfactual_action_occurrences: dict[str, list[int]] = {}
        event_schedule = self._special_task_schedules['memory_decision']
        for (task, ratios) in self.counterfactual_action_mix.items():
            total = event_schedule.count(task)
            counts = self._apportion_counts(total, ratios)
            variants = [variant for variant in sorted(ratios) for _ in range(counts[variant])]
            random.Random(_deterministic_safe_hash(('counterfactual_action_schedule', task, self.epoch, self.seed))).shuffle(variants)
            seen: Counter[str] = Counter()
            occurrences = []
            for variant in variants:
                occurrences.append(seen[variant])
                seen[variant] += 1
            self._counterfactual_action_schedules[task] = variants
            self._counterfactual_action_occurrences[task] = occurrences

    def _apply_counterfactual_batch_grouping(self, schedule: list[int]) -> list[int]:
        if self.batch_size != 4:
            raise ValueError('counterfactual batch grouping currently requires batch_size == 4')
        positions = list(self._global_positions_for_rank())
        if len(positions) != len(schedule):
            raise RuntimeError('Counterfactual grouping schedule/position length mismatch')
        result = list(schedule)
        task_plans: list[dict[str, Any]] = []
        all_source_batches: set[int] = set()
        for task in self.counterfactual_group_tasks:
            if task not in self._tasks:
                raise ValueError(f'Unknown counterfactual_group_task={task!r}')
            group_ratio = self._counterfactual_group_ratio_for_task(task)
            if group_ratio <= 0.0:
                continue
            virtual_pair_map = getattr(self.dataset, 'counterfactual_action_pair_pools', {}).get(task, {})
            if virtual_pair_map:
                offsets_by_kind: dict[str, list[int]] = {'decision_positive': [], 'decision_negative': []}
                eligible_virtual_indices = {int(index) for kind in offsets_by_kind for index in self.dataset.counterfactual_action_pools[task][kind]}
                for (local_offset, global_position) in enumerate(positions):
                    (sample_pool, _, _) = self._pool_slot(global_position)
                    if sample_pool != 'memory_decision' or int(result[local_offset]) not in eligible_virtual_indices:
                        continue
                    spec = self.dataset.counterfactual_action_spec(int(result[local_offset]))
                    kind = str((spec or {}).get('sample_kind', ''))
                    if kind in offsets_by_kind:
                        offsets_by_kind[kind].append(local_offset)
                if math.isclose(group_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
                    local_decisions = sum((len(offsets) for offsets in offsets_by_kind.values()))
                    grouped_pairs = math.ceil(local_decisions / 2)
                    source_offsets = [offset for offsets in offsets_by_kind.values() for offset in offsets]
                else:
                    pair_capacity = min((len(offsets_by_kind[kind]) for kind in offsets_by_kind))
                    grouped_pairs = math.floor(pair_capacity * group_ratio + 1e-12)
                    source_offsets = offsets_by_kind['decision_positive'][:grouped_pairs] + offsets_by_kind['decision_negative'][:grouped_pairs]
                if grouped_pairs:
                    source_offsets.sort()
                    all_source_batches.update((offset // self.batch_size for offset in source_offsets))
                    (full_batches, tail_pairs) = divmod(grouped_pairs, 2)
                    task_plans.append({'task': task, 'source_offsets': source_offsets, 'templates': ['virtual_action_pairs'] * full_batches + ['virtual_action_pair_tail'] * tail_pairs})
                continue
            pair_map = self._tasks[task]['counterfactual_pairs']
            event_pairs = sorted((pair_id for (pair_id, members) in pair_map.items() if len(members) == 2 and {member['sample_pool'] for member in members} == {'memory_decision'}))
            event_pair_indices = {int(index) for pair_id in event_pairs for member in pair_map[pair_id] for index in member['counterfactual_indices']}
            if task == 'cover':
                causal_groups = self._tasks[task]['causal_groups']
                if not event_pairs or not causal_groups:
                    raise ValueError('Cover batch grouping needs mapping pairs and four-member completion groups')
                role_indices: dict[int, set[int]] = {role: set() for role in range(4)}
                for (group_id, members) in causal_groups.items():
                    roles = {int(member['causal_group_role']) for member in members}
                    if len(members) != 4 or roles != {0, 1, 2, 3}:
                        raise ValueError(f'Cover causal group {group_id!r} must contain roles 0..3')
                    for member in members:
                        role_indices[int(member['causal_group_role'])].update(map(int, member['indices']))
                offsets_by_role: dict[int, list[int]] = {role: [] for role in range(4)}
                mapping_offsets: list[int] = []
                for (local_offset, global_position) in enumerate(positions):
                    (sample_pool, special_ordinal, _) = self._pool_slot(global_position)
                    if sample_pool not in {'memory_decision', 'completion_guard'}:
                        continue
                    if self._task_for_special(sample_pool, special_ordinal) != task:
                        continue
                    sample_index = int(result[local_offset])
                    matched_role = next((role for (role, values) in role_indices.items() if sample_index in values), None)
                    if matched_role is not None:
                        offsets_by_role[matched_role].append(local_offset)
                    elif sample_pool == 'memory_decision' and sample_index in event_pair_indices:
                        mapping_offsets.append(local_offset)
                completion_capacity = min((len(offsets_by_role[role]) for role in range(4)))
                completion_groups = math.floor(completion_capacity * group_ratio + 1e-12)
                mapping_groups = math.floor(len(mapping_offsets) // self.batch_size * group_ratio + 1e-12)
                source_offsets = [offset for role in range(4) for offset in offsets_by_role[role][:completion_groups]]
                source_offsets.extend(mapping_offsets[:mapping_groups * self.batch_size])
                templates = ['cover_completion'] * completion_groups + ['pairs'] * mapping_groups
            else:
                if len(event_pairs) < 2:
                    raise ValueError(f'counterfactual_group_task={task!r} needs at least two verified event pairs')
                event_offsets = []
                for (local_offset, global_position) in enumerate(positions):
                    (sample_pool, event_ordinal, _) = self._pool_slot(global_position)
                    if sample_pool != 'memory_decision':
                        continue
                    if self._task_for_special(sample_pool, event_ordinal) != task:
                        continue
                    if int(result[local_offset]) in event_pair_indices:
                        event_offsets.append(local_offset)
                pair_groups = math.floor(len(event_offsets) // self.batch_size * group_ratio + 1e-12)
                source_offsets = event_offsets[:pair_groups * self.batch_size]
                templates = ['pairs'] * pair_groups
            if not templates:
                continue
            source_offsets.sort()
            all_source_batches.update((offset // self.batch_size for offset in source_offsets))
            task_plans.append({'task': task, 'source_offsets': source_offsets, 'templates': templates})
        task_plans.sort(key=lambda plan: any((template in {'virtual_action_pairs', 'virtual_action_pair_tail'} for template in plan['templates'])))
        used_destination_batches: set[int] = set()
        for plan in task_plans:
            task = plan['task']
            source_offsets = plan['source_offsets']
            templates = plan['templates']
            pair_map = self._tasks[task]['counterfactual_pairs']
            virtual_pair_map = getattr(self.dataset, 'counterfactual_action_pair_pools', {}).get(task, {})
            event_pairs = sorted((pair_id for (pair_id, members) in pair_map.items() if {member['sample_pool'] for member in members} == {'memory_decision'}))
            virtual_action_plan = bool(templates) and all((template in {'virtual_action_pairs', 'virtual_action_pair_tail'} for template in templates))
            if virtual_action_plan:
                source_offset_set = set(source_offsets)
                batch_occupancy = Counter((offset // self.batch_size for offset in source_offsets))
                destination_batches = sorted((batch_index for batch_index in range(len(result) // self.batch_size) if batch_index not in used_destination_batches), key=lambda batch_index: (-batch_occupancy[batch_index], batch_index))[:len(templates)]
                if len(destination_batches) < len(templates):
                    raise ValueError(f'Not enough batches to group {task}: need {len(templates)}, got {len(destination_batches)}')
                destination_offsets_by_group = [list(range(batch_index * self.batch_size, batch_index * self.batch_size + (2 if template == 'virtual_action_pair_tail' else self.batch_size))) for (batch_index, template) in zip(destination_batches, templates, strict=True)]
                destination_offset_set = {offset for group_offsets in destination_offsets_by_group for offset in group_offsets}
                relocation_source_offsets = sorted(source_offset_set - destination_offset_set)
                displaced_source_offsets = sorted(destination_offset_set - source_offset_set)
                displaced_values = [result[offset] for offset in displaced_source_offsets]
                if len(relocation_source_offsets) > len(displaced_values):
                    raise RuntimeError('Virtual counterfactual grouping relocation is unbalanced')
            else:
                destination_batches = [batch_index for batch_index in range(len(result) // self.batch_size) if batch_index not in all_source_batches and batch_index not in used_destination_batches]
                relocation_source_offsets = list(source_offsets)
                displaced_values = []
                destination_offsets_by_group = [list(range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)) for batch_index in destination_batches[:len(templates)]]
            if len(destination_batches) < len(templates):
                raise ValueError(f'Not enough free batches to group {task}: need {len(templates)}, got {len(destination_batches)}')
            relocation_source_position = 0
            virtual_pairs_by_recipient_target: dict[str, list[str]] = defaultdict(list)
            for (pair_id, members) in virtual_pair_map.items():
                candidates = [int(index) for kind in ('decision_positive', 'decision_negative') for index in members[kind]]
                targets = set()
                for index in candidates:
                    spec = self.dataset.counterfactual_action_spec(index) or {}
                    hierarchy = tuple(spec.get('sampling_hierarchy') or ())
                    targets.add(str(spec.get('recipient_target') or (hierarchy[0] if hierarchy else '')))
                if len(targets) != 1 or not next(iter(targets)):
                    raise ValueError(f'Virtual counterfactual pair {pair_id!r} has inconsistent recipient targets')
                virtual_pairs_by_recipient_target[next(iter(targets))].append(str(pair_id))
            for pair_ids in virtual_pairs_by_recipient_target.values():
                pair_ids.sort()
            virtual_target_orders: dict[int, list[str]] = {}
            virtual_target_pair_orders: dict[tuple[str, int], list[str]] = {}

            def virtual_pair_at(ordinal: int) -> str:
                targets = sorted(virtual_pairs_by_recipient_target)
                (target_sweep, target_position) = divmod(int(ordinal), len(targets))
                target_order = virtual_target_orders.get(target_sweep)
                if target_order is None:
                    target_order = list(targets)
                    random.Random(_deterministic_safe_hash(('counterfactual_virtual_recipient_target_order', task, self.epoch, self.rank, target_sweep, self.seed))).shuffle(target_order)
                    virtual_target_orders[target_sweep] = target_order
                target = target_order[target_position]
                pair_ids = virtual_pairs_by_recipient_target[target]
                target_occurrence = target_sweep
                (pair_sweep, pair_position) = divmod(target_occurrence, len(pair_ids))
                cache_key = (target, pair_sweep)
                pair_order = virtual_target_pair_orders.get(cache_key)
                if pair_order is None:
                    pair_order = list(pair_ids)
                    random.Random(_deterministic_safe_hash(('counterfactual_virtual_recipient_episode_order', task, target, self.epoch, self.rank, pair_sweep, self.seed))).shuffle(pair_order)
                    virtual_target_pair_orders[cache_key] = pair_order
                return pair_order[pair_position]
            num_groups = len(templates)
            virtual_pair_draw_counts: Counter[str] = Counter()
            for group_ordinal in range(num_groups):
                template = templates[group_ordinal]
                pair_ids_for_group: list[str] = []
                causal_group_ids: list[str] = []
                if template in {'virtual_action_pairs', 'virtual_action_pair_tail'}:
                    pair_count = len(virtual_pair_map)
                    selected_pairs = [virtual_pair_at(group_ordinal * 2)]
                    if template == 'virtual_action_pairs':
                        second_pair = virtual_pair_at(group_ordinal * 2 + 1)
                        if second_pair == selected_pairs[0] and pair_count > 1:
                            alternate = next((pair_id for pair_id in sorted(virtual_pair_map) if pair_id != selected_pairs[0]))
                            second_pair = alternate
                        selected_pairs.append(second_pair)
                    group_values = []
                    for (pair_position, pair_id) in enumerate(selected_pairs):
                        pair_occurrence = virtual_pair_draw_counts[pair_id]
                        virtual_pair_draw_counts[pair_id] += 1
                        members = virtual_pair_map[pair_id]
                        for kind in ('decision_positive', 'decision_negative'):
                            candidates = sorted(map(int, members[kind]))
                            if kind == 'decision_negative':
                                by_donor_target: dict[str, list[int]] = defaultdict(list)
                                for candidate in candidates:
                                    spec = self.dataset.counterfactual_action_spec(candidate) or {}
                                    hierarchy = tuple(spec.get('sampling_hierarchy') or ())
                                    donor_target = str(spec.get('donor_target') or (hierarchy[2] if len(hierarchy) > 2 else ''))
                                    if not donor_target:
                                        raise ValueError(f'Virtual counterfactual pair {pair_id!r} has an empty donor target')
                                    by_donor_target[donor_target].append(candidate)
                                donor_targets = sorted(by_donor_target)
                                donor_cycle = pair_occurrence
                                random.Random(_deterministic_safe_hash(('counterfactual_virtual_donor_target_order', task, pair_id, self.epoch, donor_cycle // len(donor_targets), self.seed))).shuffle(donor_targets)
                                candidates = by_donor_target[donor_targets[donor_cycle % len(donor_targets)]]
                            choice = _deterministic_safe_hash(('counterfactual_virtual_pair_member', task, pair_id, kind, group_ordinal, pair_position, self.epoch, self.seed)) % len(candidates)
                            group_values.append(candidates[choice])
                    pair_ids_for_group = selected_pairs
                elif template == 'cover_completion':
                    causal_groups = self._tasks[task]['causal_groups']
                    group_ids = sorted(causal_groups)
                    (sweep, group_position) = divmod(group_ordinal, len(group_ids))
                    random.Random(_deterministic_safe_hash(('cover_completion_group_order', self.epoch, self.rank, sweep, self.seed))).shuffle(group_ids)
                    group_id = group_ids[group_position]
                    members = sorted(causal_groups[group_id], key=lambda group: int(group['causal_group_role']))
                    group_values = []
                    for member in members:
                        anchors = member['causal_group_indices']
                        role = int(member['causal_group_role'])
                        anchor = _deterministic_safe_hash(('cover_completion_group_anchor', group_id, role, group_ordinal, self.epoch, self.seed)) % len(anchors)
                        group_values.append(anchors[anchor])
                    causal_group_ids = [group_id]
                else:
                    if task == 'cover':
                        red_pairs = [pair_id for pair_id in event_pairs if pair_map[pair_id][0]['counterfactual_family'] == 'cover:uncover_red']
                        green_pairs = [pair_id for pair_id in event_pairs if str(pair_map[pair_id][0]['counterfactual_family']).startswith('cover:uncover_green:')]
                        pair_pools = (red_pairs, green_pairs) if green_pairs else (red_pairs, red_pairs)
                    elif task == 'press_by_number':
                        pair_ids_by_family: dict[str, list[str]] = {}
                        for pair_id in event_pairs:
                            families = {str(member['counterfactual_family']) for member in pair_map[pair_id]}
                            if len(families) != 1:
                                raise ValueError(f'Press counterfactual pair {pair_id!r} crosses families')
                            pair_ids_by_family.setdefault(next(iter(families)), []).append(pair_id)
                        family_names = sorted(pair_ids_by_family)
                        pair_pools = []
                        for position in range(2):
                            selection_ordinal = group_ordinal * 2 + position
                            (family_sweep, family_position) = divmod(selection_ordinal, len(family_names))
                            family_order = list(family_names)
                            random.Random(_deterministic_safe_hash(('memory_counterfactual_batch_family_order', task, self.epoch, self.rank, family_sweep, self.seed))).shuffle(family_order)
                            family = family_order[family_position]
                            candidates = sorted(pair_ids_by_family[family])
                            (pair_sweep, pair_position) = divmod(family_sweep, len(candidates))
                            random.Random(_deterministic_safe_hash(('memory_counterfactual_batch_family_pair_order', task, family, self.epoch, self.rank, pair_sweep, self.seed))).shuffle(candidates)
                            candidates = candidates[pair_position:] + candidates[:pair_position]
                            pair_pools.append(candidates)
                        pair_pools = tuple(pair_pools)
                    else:
                        pair_pools = (event_pairs, event_pairs)
                    for (position, candidates) in enumerate(pair_pools):
                        order = list(candidates)
                        (sweep, pair_position) = divmod(group_ordinal, len(order))
                        random.Random(_deterministic_safe_hash(('memory_counterfactual_batch_order', task, position, self.epoch, self.rank, sweep, self.seed))).shuffle(order)
                        selected = order[pair_position]
                        if selected in pair_ids_for_group and len(order) > 1:
                            selected = order[(pair_position + 1) % len(order)]
                        pair_ids_for_group.append(selected)
                    group_values = []
                    for pair_id in pair_ids_for_group:
                        members = sorted(pair_map[pair_id], key=lambda group: int(group['counterfactual_role']))
                        if len(members) != 2:
                            raise ValueError(f'Counterfactual batch pair {pair_id!r} must have exactly two members')
                        if any((not member['counterfactual_indices'] for member in members)):
                            raise ValueError(f'Counterfactual batch pair {pair_id!r} has no deployed-prefix anchors')
                        aligned_count = min((len(member['counterfactual_indices']) for member in members))
                        anchor = _deterministic_safe_hash(('memory_counterfactual_batch_anchor', pair_id, group_ordinal, self.epoch, self.seed)) % aligned_count
                        group_values.extend((members[0]['counterfactual_indices'][anchor], members[1]['counterfactual_indices'][anchor]))
                destination_batch = destination_batches[group_ordinal]
                destination_start = destination_batch * self.batch_size
                destination_offsets = destination_offsets_by_group[group_ordinal]
                displaced = None if virtual_action_plan else [result[offset] for offset in destination_offsets]
                for (offset, value) in zip(destination_offsets, group_values, strict=True):
                    result[offset] = int(value)
                if not virtual_action_plan:
                    group_sources = relocation_source_offsets[relocation_source_position:relocation_source_position + self.batch_size]
                    relocation_source_position += self.batch_size
                    if len(group_sources) != self.batch_size:
                        raise RuntimeError('Counterfactual grouping exhausted relocation source offsets')
                    if displaced is None:
                        raise RuntimeError('Counterfactual grouping lost displaced destination values')
                    for (offset, value) in zip(group_sources, displaced, strict=True):
                        result[offset] = int(value)
                used_destination_batches.add(destination_batch)
                self._counterfactual_grouped_batch_starts.append(destination_start)
                self._counterfactual_grouped_batches.append({'task': task, 'kind': template, 'start': destination_start, 'pair_ids': pair_ids_for_group, 'causal_group_ids': causal_group_ids})
                self._counterfactual_grouped_samples += len(group_values)
            if virtual_action_plan:
                for (offset, value) in zip(relocation_source_offsets, displaced_values[:len(relocation_source_offsets)], strict=True):
                    result[offset] = int(value)
            elif relocation_source_position != len(relocation_source_offsets):
                raise RuntimeError('Counterfactual grouping left unused relocation source offsets')
        return result

    def __iter__(self) -> Iterator[int]:
        if self._local_schedule is not None:
            yield from self._local_schedule
            return
        for position in self._global_positions_for_rank():
            yield self._sample_global_position(position)

    def _global_positions_for_rank(self) -> Iterator[int]:
        num_global_batches = self._global_len // self._global_batch_size
        for global_batch_index in range(num_global_batches):
            start = global_batch_index * self._global_batch_size + self.rank * self.batch_size
            for position in range(start, start + self.batch_size):
                yield position

    def _sample_global_position(self, position: int) -> int:
        (sample_pool, pool_ordinal, specials_before) = self._pool_slot(position)
        if sample_pool == 'memory_decision':
            return self._sample_event(pool_ordinal)
        if sample_pool == 'completion_guard':
            return self._sample_guard(pool_ordinal)
        base_ordinal = position - specials_before
        return self._sample_base(base_ordinal)

    def _pool_slot(self, position: int) -> tuple[str, int, int]:
        specials_before = position * self._special_count // self._global_len
        specials_after = (position + 1) * self._special_count // self._global_len
        if specials_after == specials_before:
            return ('base', position - specials_before, specials_before)
        special_ordinal = specials_before
        events_before = special_ordinal * self._event_count // max(1, self._special_count)
        events_after = (special_ordinal + 1) * self._event_count // max(1, self._special_count)
        if events_after > events_before:
            return ('memory_decision', events_before, specials_before)
        return ('completion_guard', special_ordinal - events_before, specials_before)

    def _build_locality_schedule(self) -> list[int]:
        pool_slots: list[str] = []
        stream_indices: dict[str, list[int]] = {'base': [], 'memory_decision': [], 'completion_guard': []}
        for position in self._global_positions_for_rank():
            (sample_pool, pool_ordinal, specials_before) = self._pool_slot(position)
            pool_slots.append(sample_pool)
            if sample_pool == 'memory_decision':
                stream_indices[sample_pool].append(self._sample_event(pool_ordinal))
            elif sample_pool == 'completion_guard':
                stream_indices[sample_pool].append(self._sample_guard(pool_ordinal))
            else:
                base_ordinal = position - specials_before
                stream_indices[sample_pool].append(self._sample_base(base_ordinal))
        (layout, sample_episodes) = self._locality_layout(*stream_indices.values())
        stream_iters = {sample_pool: iter(self._order_stream_by_locality(indices, layout=layout, sample_episodes=sample_episodes, stream=sample_pool)) for (sample_pool, indices) in stream_indices.items()}
        schedule = [next(stream_iters[sample_pool]) for sample_pool in pool_slots]
        if len(schedule) != self._num_samples:
            raise RuntimeError(f'MemoryEventBalancedSampler locality schedule length mismatch: expected {self._num_samples}, got {len(schedule)}.')
        return schedule

    def _locality_layout(self, *index_streams: list[int]) -> tuple[list[tuple[str, list[int]]], dict[int, tuple[str, int]]]:
        sample_episodes: dict[int, tuple[str, int]] = {}
        shard_episodes: dict[str, set[int]] = {}
        all_indices = [sample_index for stream in index_streams for sample_index in stream]
        for sample_index in set(all_indices):
            ref = self.dataset._sample_ref_from_index(int(sample_index))
            episode_index = int(ref.episode_index)
            episode = self.dataset.episodes[episode_index]
            shard_path = self._episode_shard_path(episode)
            sample_episodes[int(sample_index)] = (shard_path, episode_index)
            shard_episodes.setdefault(shard_path, set()).add(episode_index)
        rng = random.Random(_deterministic_safe_hash(('memory_locality_layout', self.epoch, self.rank, self.seed)))
        shard_order = sorted(shard_episodes)
        rng.shuffle(shard_order)
        layout: list[tuple[str, list[int]]] = []
        for shard_path in shard_order:
            episode_order = sorted(shard_episodes[shard_path])
            rng.shuffle(episode_order)
            layout.append((shard_path, episode_order))
        return (layout, sample_episodes)

    def _order_stream_by_locality(self, indices: list[int], *, layout: list[tuple[str, list[int]]], sample_episodes: Mapping[int, tuple[str, int]], stream: str) -> list[int]:
        episode_samples: dict[tuple[str, int], list[int]] = {}
        for sample_index in indices:
            key = sample_episodes[int(sample_index)]
            episode_samples.setdefault(key, []).append(int(sample_index))
        ordered: list[int] = []
        for (shard_path, episode_order) in layout:
            for episode_index in episode_order:
                samples = episode_samples.get((shard_path, episode_index), [])
                if self.locality_shuffle_within_episode and len(samples) > 1:
                    random.Random(_deterministic_safe_hash(('memory_locality_episode', stream, self.epoch, self.rank, episode_index, self.seed))).shuffle(samples)
                ordered.extend(samples)
        if len(ordered) != len(indices):
            raise RuntimeError(f'MemoryEventBalancedSampler lost {stream} samples during locality ordering: expected {len(indices)}, got {len(ordered)}.')
        return ordered

    @staticmethod
    def _episode_shard_path(episode: Mapping[str, Any]) -> str:
        storage = episode.get('storage') or {}
        extra = storage.get('extra') if isinstance(storage, Mapping) else None
        if isinstance(extra, Mapping) and extra.get('shard_path'):
            return str(extra['shard_path'])
        streams = storage.get('streams') if isinstance(storage, Mapping) else None
        if isinstance(streams, Mapping):
            for stream in streams.values():
                if isinstance(stream, Mapping) and stream.get('shard_path'):
                    return str(stream['shard_path'])
        return '__unsharded__'

    def _sample_event(self, event_ordinal: int) -> int:
        return self._sample_special('memory_decision', event_ordinal)

    def _sample_guard(self, guard_ordinal: int) -> int:
        return self._sample_special('completion_guard', guard_ordinal)

    def _sample_special(self, sample_pool: str, ordinal: int) -> int:
        rng = random.Random(_deterministic_safe_hash(('memory_special', sample_pool, self.epoch, int(ordinal), self.seed)))
        task = self._task_for_special(sample_pool, ordinal)
        if sample_pool == 'memory_decision' and task in self.counterfactual_action_mix:
            task_occurrence = self._special_task_occurrences[sample_pool][int(ordinal)]
            variant = self._counterfactual_action_schedules[task][task_occurrence]
            variant_occurrence = self._counterfactual_action_occurrences[task][task_occurrence]
            indices = self.dataset.counterfactual_action_pools[task][variant]
            if indices and self.dataset.counterfactual_action_spec(int(indices[0])) is not None:
                hierarchy_index = self._counterfactual_action_hierarchy_index(task, variant, indices)
                hierarchy_depth = int(hierarchy_index['depth'])
                occurrence = int(variant_occurrence)
                prefix: tuple[str, ...] = ()
                candidates = list(map(int, indices)) if hierarchy_depth == 0 else []
                for depth in range(hierarchy_depth):
                    grouped = hierarchy_index['groups'][prefix]
                    keys = sorted(grouped)
                    (sweep, position) = divmod(occurrence, len(keys))
                    random.Random(_deterministic_safe_hash(('counterfactual_action_hierarchy', task, variant, depth, self.epoch, sweep, self.seed))).shuffle(keys)
                    selected_key = keys[position]
                    candidates = grouped[selected_key]
                    prefix = (*prefix, selected_key)
                    occurrence = sweep
                order = sorted(candidates)
                (cycle, cycle_position) = divmod(occurrence, len(order))
                random.Random(_deterministic_safe_hash(('counterfactual_action_hierarchy_leaf', task, variant, self.epoch, cycle, self.seed))).shuffle(order)
                return int(order[cycle_position])
            (cycle, cycle_position) = divmod(variant_occurrence, len(indices))
            order = list(indices)
            random.Random(_deterministic_safe_hash(('counterfactual_action_variant', task, variant, self.epoch, cycle, self.seed))).shuffle(order)
            return int(order[cycle_position])
        counterfactual = self._sample_counterfactual_group(sample_pool, task, ordinal)
        if counterfactual is not None:
            (group, anchor_seed) = counterfactual
            anchors = group['counterfactual_indices']
            return anchors[anchor_seed % len(anchors)]
        episode_key = 'event_episodes' if sample_pool == 'memory_decision' else 'guard_episodes'
        group_key = 'events' if sample_pool == 'memory_decision' else 'guards'
        episodes = self._tasks[task][episode_key]
        stratum_key = 'event_strata' if sample_pool == 'memory_decision' else 'guard_strata'
        strata = self._tasks[task][stratum_key]
        if strata:
            occurrence = self._special_task_occurrences[sample_pool][int(ordinal)]
            stratum_names = sorted(strata)
            (sweep, stratum_position) = divmod(occurrence, len(stratum_names))
            random.Random(_deterministic_safe_hash(('memory_special_stratum_order', sample_pool, task, self.epoch, sweep, self.seed))).shuffle(stratum_names)
            groups = strata[stratum_names[stratum_position]]
            group = groups[self._weighted_choice([item['weight'] for item in groups], rng)]
            anchors = group['indices']
            return anchors[rng.randrange(len(anchors))]
        episode = episodes[rng.randrange(len(episodes))]
        groups = episode[group_key]
        group = groups[self._weighted_choice([item['weight'] for item in groups], rng)]
        anchors = group['indices']
        return anchors[rng.randrange(len(anchors))]

    def _counterfactual_action_hierarchy_index(self, task: str, variant: str, indices: Sequence[int]) -> dict[str, Any]:
        cache_key = (str(task), str(variant))
        cached = self._counterfactual_action_hierarchy_cache.get(cache_key)
        if cached is not None:
            return cached
        groups: dict[tuple[str, ...], dict[str, list[int]]] = {}
        hierarchy_depth: int | None = None
        for raw_index in indices:
            index = int(raw_index)
            spec = self.dataset.counterfactual_action_spec(index)
            if spec is None:
                raise ValueError(f'Counterfactual action pool {task}/{variant} mixes physical and virtual samples')
            hierarchy = tuple((str(value) for value in spec.get('sampling_hierarchy', ())))
            if hierarchy_depth is None:
                hierarchy_depth = len(hierarchy)
            elif len(hierarchy) != hierarchy_depth:
                raise ValueError(f'Counterfactual action hierarchy depth differs inside {task}/{variant}')
            for (depth, value) in enumerate(hierarchy):
                groups.setdefault(hierarchy[:depth], {}).setdefault(value, []).append(index)
        resolved = {'depth': hierarchy_depth or 0, 'groups': groups}
        self._counterfactual_action_hierarchy_cache[cache_key] = resolved
        return resolved

    def _sample_counterfactual_group(self, sample_pool: str, task: str, ordinal: int) -> tuple[dict[str, Any], int] | None:
        pair_ratio = self.counterfactual_pair_ratio_overrides.get(task, {}).get(sample_pool, self.counterfactual_pair_ratio)
        if pair_ratio <= 0.0:
            return None
        task_cycle = self._special_task_occurrences[sample_pool][int(ordinal)]
        draws_before = math.floor(task_cycle * pair_ratio + 1e-12)
        draws_after = math.floor((task_cycle + 1) * pair_ratio + 1e-12)
        if draws_after == draws_before:
            return None
        pair_key = 'event_counterfactual_pairs' if sample_pool == 'memory_decision' else 'guard_counterfactual_pairs'
        pair_map = self._tasks[task][pair_key]
        if not pair_map:
            return None
        pair_draw_ordinal = draws_after - 1
        (pair_number, requested_role) = divmod(pair_draw_ordinal, 2)
        pair_ids = sorted(pair_map)
        if task == 'press_by_number':
            pair_ids_by_family: dict[str, list[str]] = {}
            for (candidate_id, candidate_members) in pair_map.items():
                families = {str(member['counterfactual_family']) for member in candidate_members}
                if len(families) != 1:
                    raise ValueError(f'Counterfactual pair {candidate_id!r} crosses families')
                pair_ids_by_family.setdefault(next(iter(families)), []).append(candidate_id)
            family_names = sorted(pair_ids_by_family)
            (family_sweep, family_position) = divmod(pair_number, len(family_names))
            random.Random(_deterministic_safe_hash(('memory_counterfactual_family_order', sample_pool, task, self.epoch, family_sweep, self.seed))).shuffle(family_names)
            family = family_names[family_position]
            pair_ids = sorted(pair_ids_by_family[family])
            (pair_sweep, pair_position) = divmod(family_sweep, len(pair_ids))
            random.Random(_deterministic_safe_hash(('memory_counterfactual_family_pair_order', sample_pool, task, family, self.epoch, pair_sweep, self.seed))).shuffle(pair_ids)
            pair_id = pair_ids[pair_position]
        else:
            (sweep, pair_position) = divmod(pair_number, len(pair_ids))
            random.Random(_deterministic_safe_hash(('memory_counterfactual_order', sample_pool, task, self.epoch, sweep, self.seed))).shuffle(pair_ids)
            pair_id = pair_ids[pair_position]
        members = sorted(pair_map[pair_id], key=lambda group: int(group['counterfactual_role']))
        group = members[requested_role % len(members)]
        anchor_seed = _deterministic_safe_hash(('memory_counterfactual_anchor', pair_id, pair_number, self.epoch, self.seed))
        return (group, anchor_seed)

    def _task_for_special(self, sample_pool: str, ordinal: int) -> str:
        return self._special_task_schedules[sample_pool][int(ordinal)]

    def _set_base_permutation(self) -> None:
        size = len(self._base_indices)
        self._scheduled_base_task_counts = Counter({task: len(indices) for (task, indices) in self._base_indices_by_task.items()})
        self._base_task_schedule: list[str] | None = None
        self._base_task_occurrences: list[int] | None = None
        self._base_task_cycle_permutations: dict[tuple[str, int], tuple[int, int]] = {}
        if self.task_final_mix:
            if self._final_pool_task_counts is None:
                raise RuntimeError('task_final_mix pool quotas were not initialized')
            base_counts = self._final_pool_task_counts['base']
            if sum(base_counts.values()) != self._scheduled_base_count:
                raise RuntimeError(f'task_final_mix base quotas do not preserve the global base-stream size: expected {self._scheduled_base_count}, got {sum(base_counts.values())}')
            schedule = [task for task in sorted(base_counts) for _ in range(base_counts[task])]
            random.Random(_deterministic_safe_hash(('memory_base_task_schedule', self.epoch, self.seed))).shuffle(schedule)
            occurrences: list[int] = []
            seen: Counter[str] = Counter()
            for task in schedule:
                occurrences.append(seen[task])
                seen[task] += 1
            self._scheduled_base_task_counts = base_counts
            self._base_task_schedule = schedule
            self._base_task_occurrences = occurrences
            return
        rng = random.Random(_deterministic_safe_hash(('memory_base', self.epoch, self.seed)))
        multiplier = rng.randrange(1, size + 1)
        while math.gcd(multiplier, size) != 1:
            multiplier = multiplier % size + 1
        self._base_multiplier = multiplier
        self._base_offset = rng.randrange(size)

    def _sample_base(self, base_ordinal: int) -> int:
        if self._base_task_schedule is None or self._base_task_occurrences is None:
            permuted = (self._base_multiplier * base_ordinal + self._base_offset) % len(self._base_indices)
            return self._base_indices[permuted]
        task = self._base_task_schedule[int(base_ordinal)]
        occurrence = self._base_task_occurrences[int(base_ordinal)]
        indices = self._base_indices_by_task[task]
        (cycle, cycle_position) = divmod(occurrence, len(indices))
        key = (task, cycle)
        if key not in self._base_task_cycle_permutations:
            rng = random.Random(_deterministic_safe_hash(('memory_base_task', task, self.epoch, cycle, self.seed)))
            multiplier = rng.randrange(1, len(indices) + 1)
            while math.gcd(multiplier, len(indices)) != 1:
                multiplier = multiplier % len(indices) + 1
            self._base_task_cycle_permutations[key] = (multiplier, rng.randrange(len(indices)))
        (multiplier, offset) = self._base_task_cycle_permutations[key]
        return indices[(multiplier * cycle_position + offset) % len(indices)]

    def _build_task_pools(self) -> dict[str, dict[str, Any]]:
        pools: dict[str, dict[str, Any]] = {}
        anchor_stride = int((self.dataset.state_action_config.get('sampling') or {}).get('anchor_stride', 1))
        for (compact_pos, compact) in enumerate(self.dataset.episode_index):
            if compact.num_samples <= 0:
                continue
            episode = self.dataset.episodes[int(compact.episode_index)]
            extra = episode.get('extra') or {}
            task = str(extra.get('memory_task') or '')
            raw_events = extra.get('memory_events') or []
            if not task or not isinstance(raw_events, list):
                continue
            global_start = int(self.dataset.sample_cumsum[compact_pos])
            local_indices = compact.sample_local_indices if compact.sample_local_indices is not None else tuple(range(compact.num_samples))
            step_to_global = {int(local_index) * anchor_stride: global_start + position for (position, local_index) in enumerate(local_indices)}
            event_groups: list[dict[str, Any]] = []
            guard_groups: list[dict[str, Any]] = []
            labelled_anchor_set: set[int] = set()
            excluded_anchor_set: set[int] = set()
            for event in raw_events:
                if not isinstance(event, Mapping):
                    continue
                weight = float(event.get('sampling_weight', 0.0))
                declared_pool = event.get('sample_pool')
                sample_pool = str(declared_pool if declared_pool is not None else 'memory_decision' if weight > 0 else 'excluded')
                start = int(event['anchor_start'])
                end = int(event['anchor_end'])
                sampled_steps = [step for step in range(start, end + 1) if step in step_to_global]
                indices = [step_to_global[step] for step in sampled_steps]
                if not indices:
                    continue
                overlap = labelled_anchor_set.intersection(indices)
                if overlap:
                    raise ValueError(f"Overlapping memory event anchors in episode {episode.get('episode_id')!r}: {sorted(overlap)[:5]}")
                labelled_anchor_set.update(indices)
                if sample_pool == 'excluded':
                    excluded_anchor_set.update(indices)
                    continue
                if sample_pool not in {'memory_decision', 'completion_guard'}:
                    raise ValueError(f'Unsupported memory event sample_pool={sample_pool!r}')
                if weight <= 0 or not math.isfinite(weight):
                    raise ValueError(f"Trainable memory event {event.get('event_id')!r} has invalid sampling_weight={weight}")
                counterfactual = event.get('counterfactual') or {}
                causal_group = event.get('causal_group') or {}
                supervision_start = event.get('supervision_start_step')
                supervision_end = event.get('supervision_end_step')
                center_step = event.get('center_step')
                counterfactual_steps = sampled_steps
                if counterfactual.get('pair_id') is not None:
                    explicit_steps = event.get('counterfactual_anchor_steps')
                    if explicit_steps is not None:
                        if not isinstance(explicit_steps, list) or len(set(explicit_steps)) != len(explicit_steps):
                            raise ValueError(f"Counterfactual event {event.get('event_id')!r} has invalid explicit anchors")
                        counterfactual_steps = [int(step) for step in explicit_steps if int(step) in sampled_steps]
                        if len(counterfactual_steps) != len(explicit_steps):
                            raise ValueError(f"Counterfactual event {event.get('event_id')!r} has explicit anchors outside its sampled event window")
                    elif supervision_start is not None:
                        counterfactual_steps = [step for step in sampled_steps if step + 1 <= int(supervision_end) and step + 10 >= int(supervision_start) and (center_step is None or step <= int(center_step))]
                    if not counterfactual_steps:
                        raise ValueError(f"Counterfactual event {event.get('event_id')!r} has no supervised action inside the deployed 10-step prefix")
                causal_group_steps = sampled_steps
                if causal_group.get('group_id') is not None:
                    explicit_steps = event.get('causal_group_anchor_steps')
                    if explicit_steps is not None:
                        if not isinstance(explicit_steps, list) or len(set(explicit_steps)) != len(explicit_steps):
                            raise ValueError(f"Causal-group event {event.get('event_id')!r} has invalid anchors")
                        causal_group_steps = [int(step) for step in explicit_steps if int(step) in sampled_steps]
                        if len(causal_group_steps) != len(explicit_steps):
                            raise ValueError(f"Causal-group event {event.get('event_id')!r} has anchors outside its event window")
                    if not causal_group_steps:
                        raise ValueError(f"Causal-group event {event.get('event_id')!r} has no usable anchors")
                group = {'indices': indices, 'counterfactual_indices': [step_to_global[step] for step in counterfactual_steps], 'causal_group_indices': [step_to_global[step] for step in causal_group_steps], 'weight': weight, 'event_type': str(event.get('event_type', '')), 'counterfactual_pair_id': counterfactual.get('pair_id'), 'counterfactual_role': counterfactual.get('role'), 'counterfactual_family': counterfactual.get('family'), 'causal_group_id': causal_group.get('group_id'), 'causal_group_role': causal_group.get('role'), 'sample_pool': sample_pool, 'sampling_stratum': event.get('sampling_stratum')}
                if sample_pool == 'memory_decision':
                    event_groups.append(group)
                else:
                    guard_groups.append(group)
            all_indices = list(step_to_global.values())
            record = {'events': event_groups, 'guards': guard_groups, 'excluded_indices': sorted(excluded_anchor_set), 'base_indices': [index for index in all_indices if index not in excluded_anchor_set]}
            pool = pools.setdefault(task, {'episodes': [], 'event_episodes': [], 'guard_episodes': [], 'event_counterfactual_pairs': {}, 'guard_counterfactual_pairs': {}, 'counterfactual_pairs': {}, 'causal_groups': {}, 'event_strata': {}, 'guard_strata': {}})
            pool['episodes'].append(record)
            if event_groups:
                pool['event_episodes'].append(record)
                for group in event_groups:
                    stratum = group['sampling_stratum']
                    if stratum is not None:
                        pool['event_strata'].setdefault(str(stratum), []).append(group)
                    pair_id = group['counterfactual_pair_id']
                    if pair_id is not None:
                        pool['event_counterfactual_pairs'].setdefault(str(pair_id), []).append(group)
                        pool['counterfactual_pairs'].setdefault(str(pair_id), []).append(group)
            if guard_groups:
                pool['guard_episodes'].append(record)
                for group in guard_groups:
                    stratum = group['sampling_stratum']
                    if stratum is not None:
                        pool['guard_strata'].setdefault(str(stratum), []).append(group)
                    pair_id = group['counterfactual_pair_id']
                    if pair_id is not None:
                        pool['guard_counterfactual_pairs'].setdefault(str(pair_id), []).append(group)
                        pool['counterfactual_pairs'].setdefault(str(pair_id), []).append(group)
            for group in (*event_groups, *guard_groups):
                group_id = group['causal_group_id']
                if group_id is not None:
                    pool['causal_groups'].setdefault(str(group_id), []).append(group)
        for (task, pool) in pools.items():
            for (groups_key, strata_key) in (('events', 'event_strata'), ('guards', 'guard_strata')):
                all_groups = [group for episode in pool['episodes'] for group in episode[groups_key]]
                tagged = sum((group['sampling_stratum'] is not None for group in all_groups))
                if tagged not in {0, len(all_groups)}:
                    raise ValueError(f'Task {task!r} must label either every or no {groups_key} with sampling_stratum; got {tagged}/{len(all_groups)}')
                if tagged and sum((len(groups) for groups in pool[strata_key].values())) != len(all_groups):
                    raise RuntimeError(f'Task {task!r} lost stratified {groups_key} while building sampler pools')
        return pools

    @staticmethod
    def _weighted_choice(weights: list[float], rng: random.Random) -> int:
        total = sum((weight for weight in weights if weight > 0))
        if total <= 0 or not math.isfinite(total):
            return rng.randrange(len(weights))
        target = rng.random() * total
        cumulative = 0.0
        for (index, weight) in enumerate(weights):
            cumulative += max(0.0, weight)
            if target <= cumulative:
                return index
        return len(weights) - 1

    def debug_summary(self) -> dict[str, Any]:
        return {'type': type(self).__name__, 'target_event_ratio': self.event_ratio, 'target_guard_ratio': self.guard_ratio, 'actual_event_ratio': self._event_count / self._global_len, 'actual_guard_ratio': self._guard_count / self._global_len, 'counterfactual_pair_ratio': self.counterfactual_pair_ratio, 'counterfactual_pair_ratio_overrides': self.counterfactual_pair_ratio_overrides, 'counterfactual_group_ratio': self.counterfactual_group_ratio, 'counterfactual_group_ratio_overrides': self.counterfactual_group_ratio_overrides, 'counterfactual_group_tasks': list(self.counterfactual_group_tasks), 'counterfactual_action_mix': self.counterfactual_action_mix, 'counterfactual_action_samples': {task: dict(Counter(schedule)) for (task, schedule) in self._counterfactual_action_schedules.items()}, 'task_target_mix': self.task_target_mix, 'task_final_mix': self.task_final_mix, 'counterfactual_grouped_batches': len(self._counterfactual_grouped_batch_starts), 'counterfactual_grouped_samples': self._counterfactual_grouped_samples, 'counterfactual_grouped_batches_by_task': dict(Counter((item['task'] for item in self._counterfactual_grouped_batches))), 'counterfactual_grouped_batches_by_kind': dict(Counter((item['kind'] for item in self._counterfactual_grouped_batches))), 'base_samples': self._scheduled_base_count, 'available_base_samples': len(self._base_indices), 'event_samples': self._event_count, 'guard_samples': self._guard_count, 'tasks': {task: {'episodes': len(pool['episodes']), 'event_episodes': len(pool['event_episodes']), 'guard_episodes': len(pool['guard_episodes']), 'events': sum((len(episode['events']) for episode in pool['event_episodes'])), 'guards': sum((len(episode['guards']) for episode in pool['guard_episodes'])), 'event_counterfactual_pairs': len(pool['event_counterfactual_pairs']), 'guard_counterfactual_pairs': len(pool['guard_counterfactual_pairs']), 'causal_groups': len(pool['causal_groups']), 'event_strata': len(pool['event_strata']), 'guard_strata': len(pool['guard_strata']), 'excluded_samples': sum((len(episode['excluded_indices']) for episode in pool['episodes'])), 'base_samples': sum((len(episode['base_indices']) for episode in pool['episodes'])), 'scheduled_base_samples': self._scheduled_base_task_counts[task], 'scheduled_event_samples': self._special_task_schedules['memory_decision'].count(task), 'scheduled_guard_samples': self._special_task_schedules['completion_guard'].count(task)} for (task, pool) in self._tasks.items()}, 'length': len(self), 'rank': self.rank, 'world_size': self.world_size, 'epoch': self.epoch, 'locality_aware': self.locality_aware, 'locality_shuffle_within_episode': self.locality_shuffle_within_episode}


class ShardLocalitySampler(Sampler[int]):

    def __init__(self, dataset: CanonicalEmbodiedDataset, *, shuffle_shards: bool=True, shuffle_episodes: bool=True, shuffle_within_episode: bool=False, shuffle_episode_blocks: bool=False, episode_block_size: int | None=None, shard_partition: str='balanced', worker_shard_affinity: bool=False, num_workers: int=0, batch_size: int=1, seed: int=0, rank: int | None=None, world_size: int | None=None, drop_last: bool=False) -> None:
        if not isinstance(dataset, CanonicalEmbodiedDataset):
            raise TypeError('ShardLocalitySampler requires CanonicalEmbodiedDataset')
        self.dataset = dataset
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_episodes = bool(shuffle_episodes)
        self.shuffle_within_episode = bool(shuffle_within_episode)
        self.shuffle_episode_blocks = bool(shuffle_episode_blocks)
        self.episode_block_size = int(episode_block_size) if episode_block_size is not None else None
        self.shard_partition = str(shard_partition or 'balanced').lower()
        self.worker_shard_affinity = bool(worker_shard_affinity)
        self.num_workers = max(0, int(num_workers))
        self.batch_size = max(1, int(batch_size))
        if self.shard_partition not in {'balanced', 'round_robin'}:
            raise ValueError(f'Unsupported shard_partition: {shard_partition}')
        if self.episode_block_size is not None and self.episode_block_size <= 0:
            raise ValueError(f'episode_block_size must be positive when set, got {episode_block_size}')
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if rank is None or world_size is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1:
            raise ValueError(f'world_size must be >= 1, got {self.world_size}')
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f'rank must be in [0, {self.world_size}), got {self.rank}')
        self.epoch = 0
        self._episode_records = self._build_episode_records(dataset)
        self._shard_to_episode_positions = self._build_shard_groups(self._episode_records)
        self._shard_sample_counts = self._build_shard_sample_counts(self._episode_records, self._shard_to_episode_positions)
        self._num_samples = self._rank_num_samples(self.epoch)

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices_for_rank(self.epoch, self.rank))

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._num_samples = self._rank_num_samples(self.epoch)

    @property
    def shard_paths(self) -> list[str]:
        return list(self._shard_to_episode_positions)

    def shard_paths_for_rank(self, epoch: int | None=None, rank: int | None=None) -> list[str]:
        epoch = self.epoch if epoch is None else int(epoch)
        rank = self.rank if rank is None else int(rank)
        assignments = self._shard_assignments(epoch)
        return list(assignments[rank]) if rank < len(assignments) else []

    def _shard_assignments(self, epoch: int) -> list[list[str]]:
        rng = random.Random(self.seed + int(epoch))
        shards = self.shard_paths
        if self.shuffle_shards:
            rng.shuffle(shards)
        if self.shard_partition == 'round_robin':
            return [shards[rank::self.world_size] for rank in range(self.world_size)]
        ranked_shards = sorted(shards, key=lambda shard: self._shard_sample_counts.get(shard, 0), reverse=True)
        assignments: list[list[str]] = [[] for _ in range(self.world_size)]
        loads = [0 for _ in range(self.world_size)]
        for shard in ranked_shards:
            rank = min(range(self.world_size), key=lambda item: (loads[item], item))
            assignments[rank].append(shard)
            loads[rank] += int(self._shard_sample_counts.get(shard, 0))
        if self.shuffle_shards:
            for local_shards in assignments:
                rng.shuffle(local_shards)
        return assignments

    def _rank_num_samples(self, epoch: int) -> int:
        lengths = [len(self._indices_for_rank(epoch, rank, equalize=False)) for rank in range(self.world_size)]
        if not lengths:
            return 0
        return min(lengths) if self.drop_last else max(lengths)

    def _indices_for_rank(self, epoch: int, rank: int, *, equalize: bool=True) -> list[int]:
        rng = random.Random(self.seed + int(epoch))
        local_shards = self.shard_paths_for_rank(epoch, rank)
        if self.worker_shard_affinity and self.num_workers > 1:
            indices = self._worker_affinity_indices(local_shards, rng)
        else:
            indices = self._indices_from_shards(local_shards, rng)
        if not equalize or self.world_size == 1:
            return indices
        target_len = self._rank_num_samples(epoch)
        if len(indices) > target_len:
            return indices[:target_len]
        if len(indices) < target_len:
            if not indices and self._episode_records:
                fallback = self._episode_records[rank % len(self._episode_records)]
                indices = list(range(fallback['start'], fallback['end']))
            if indices:
                repeats = math.ceil((target_len - len(indices)) / len(indices))
                indices = indices + (indices * repeats)[:target_len - len(indices)]
        return indices

    def _indices_from_shards(self, shard_paths: list[str], rng: random.Random) -> list[int]:
        indices: list[int] = []
        for shard_path in shard_paths:
            episode_positions = list(self._shard_to_episode_positions.get(shard_path, ()))
            if self.shuffle_episodes:
                rng.shuffle(episode_positions)
            for episode_pos in episode_positions:
                indices.extend(self._episode_indices(episode_pos, rng))
        return indices

    def _worker_affinity_indices(self, shard_paths: list[str], rng: random.Random) -> list[int]:
        worker_shards = self._assign_shards_to_workers(shard_paths)
        worker_batches: list[list[list[int]]] = []
        for shards in worker_shards:
            worker_indices = self._indices_from_shards(list(shards), rng)
            batches = [worker_indices[i:i + self.batch_size] for i in range(0, len(worker_indices), self.batch_size)]
            worker_batches.append([batch for batch in batches if batch])
        interleaved: list[int] = []
        max_batches = max((len(batches) for batches in worker_batches), default=0)
        for batch_pos in range(max_batches):
            for worker_id in range(len(worker_batches)):
                if batch_pos < len(worker_batches[worker_id]):
                    interleaved.extend(worker_batches[worker_id][batch_pos])
        return interleaved

    def _assign_shards_to_workers(self, shard_paths: list[str]) -> list[list[str]]:
        workers = max(1, self.num_workers)
        assignments: list[list[str]] = [[] for _ in range(workers)]
        loads = [0 for _ in range(workers)]
        for shard in sorted(shard_paths, key=lambda item: self._shard_sample_counts.get(item, 0), reverse=True):
            worker_id = min(range(workers), key=lambda item: (loads[item], item))
            assignments[worker_id].append(shard)
            loads[worker_id] += int(self._shard_sample_counts.get(shard, 0))
        return assignments

    def _episode_indices(self, episode_pos: int, rng: random.Random) -> list[int]:
        record = self._episode_records[episode_pos]
        indices = list(range(int(record['start']), int(record['end'])))
        if not indices:
            return indices
        if self.episode_block_size is not None:
            blocks = [indices[i:i + self.episode_block_size] for i in range(0, len(indices), self.episode_block_size)]
            if self.shuffle_episode_blocks:
                rng.shuffle(blocks)
            if self.shuffle_within_episode:
                for block in blocks:
                    rng.shuffle(block)
            return [idx for block in blocks for idx in block]
        if self.shuffle_within_episode:
            rng.shuffle(indices)
        return indices

    @staticmethod
    def _build_episode_records(dataset: CanonicalEmbodiedDataset) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for (compact_pos, compact) in enumerate(dataset.episode_index):
            start = int(dataset.sample_cumsum[compact_pos])
            end = int(dataset.sample_cumsum[compact_pos + 1])
            if compact.num_samples <= 0 or end <= start:
                continue
            episode = dataset.episodes[int(compact.episode_index)]
            shard_path = ShardLocalitySampler._episode_shard_path(episode)
            records.append({'compact_pos': compact_pos, 'episode_index': int(compact.episode_index), 'episode_id': episode.get('episode_id'), 'shard_path': shard_path, 'start': start, 'end': end})
        return records

    @staticmethod
    def _episode_shard_path(episode: Mapping[str, Any]) -> str:
        storage = episode.get('storage') or {}
        extra = storage.get('extra') if isinstance(storage, Mapping) else None
        if isinstance(extra, Mapping) and extra.get('shard_path'):
            return str(extra['shard_path'])
        streams = storage.get('streams') if isinstance(storage, Mapping) else None
        if isinstance(streams, Mapping):
            for stream in streams.values():
                if isinstance(stream, Mapping) and stream.get('shard_path'):
                    return str(stream['shard_path'])
        return '__unsharded__'

    @staticmethod
    def _build_shard_groups(records: list[dict[str, Any]]) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for (episode_pos, record) in enumerate(records):
            groups.setdefault(str(record['shard_path']), []).append(episode_pos)
        return groups

    @staticmethod
    def _build_shard_sample_counts(records: list[dict[str, Any]], groups: Mapping[str, list[int]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for (shard_path, episode_positions) in groups.items():
            counts[shard_path] = sum((int(records[episode_pos]['end']) - int(records[episode_pos]['start']) for episode_pos in episode_positions))
        return counts

    def debug_shard_summary(self, *, max_shards: int=16) -> dict[str, Any]:
        local_shards = self.shard_paths_for_rank(self.epoch, self.rank)
        local_indices = self._indices_for_rank(self.epoch, self.rank, equalize=False)
        return {'type': type(self).__name__, 'rank': self.rank, 'world_size': self.world_size, 'epoch': self.epoch, 'num_shards_total': len(self.shard_paths), 'num_shards_local': len(local_shards), 'local_shards_head': local_shards[:max_shards], 'num_samples_local_unpadded': len(local_indices), 'num_samples_local_equalized': len(self), 'drop_last': self.drop_last, 'shard_partition': self.shard_partition, 'worker_shard_affinity': self.worker_shard_affinity, 'num_workers': self.num_workers, 'batch_size': self.batch_size}


class ShardBufferedBlockRandomSampler(ShardLocalitySampler):

    def __init__(self, dataset: CanonicalEmbodiedDataset, *, active_shards_per_rank: int=4, block_size: int=32, virtual_shard_max_samples: int | None=None, shuffle_shards: bool=True, shuffle_episodes: bool=True, shuffle_blocks: bool=True, shuffle_within_block: bool=False, shard_partition: str='balanced', seed: int=0, rank: int | None=None, world_size: int | None=None, batch_size: int=1, drop_last: bool=False) -> None:
        self.active_shards_per_rank = max(1, int(active_shards_per_rank))
        self.block_size = max(1, int(block_size))
        self.virtual_shard_max_samples = max(1, int(virtual_shard_max_samples)) if virtual_shard_max_samples is not None else None
        self.shuffle_blocks = bool(shuffle_blocks)
        self.shuffle_within_block = bool(shuffle_within_block)
        super().__init__(dataset, shuffle_shards=shuffle_shards, shuffle_episodes=shuffle_episodes, shuffle_within_episode=False, shuffle_episode_blocks=False, episode_block_size=None, shard_partition=shard_partition, worker_shard_affinity=False, num_workers=0, batch_size=batch_size, seed=seed, rank=rank, world_size=world_size, drop_last=drop_last)
        self._shard_units = self._build_shard_units()
        self._unit_sample_counts = {str(unit['unit_id']): sum((self._episode_num_samples(pos) for pos in unit['episode_positions'])) for unit in self._shard_units}
        self._num_samples = self._rank_num_samples(self.epoch)

    def shard_paths_for_rank(self, epoch: int | None=None, rank: int | None=None) -> list[str]:
        epoch = self.epoch if epoch is None else int(epoch)
        rank = self.rank if rank is None else int(rank)
        assignments = self._unit_assignments(epoch)
        if rank >= len(assignments):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for unit in assignments[rank]:
            shard_path = str(unit['shard_path'])
            if shard_path not in seen:
                seen.add(shard_path)
                out.append(shard_path)
        return out

    def _rank_num_samples(self, epoch: int) -> int:
        if not hasattr(self, '_shard_units'):
            return 0
        lengths = [len(self._indices_for_rank(epoch, rank, equalize=False)) for rank in range(self.world_size)]
        if not lengths:
            return 0
        if self.drop_last:
            return min(lengths)
        return max(lengths)

    def _indices_for_rank(self, epoch: int, rank: int, *, equalize: bool=True) -> list[int]:
        if not hasattr(self, '_shard_units'):
            return []
        rng = random.Random(self.seed + int(epoch))
        assignments = self._unit_assignments(epoch)
        units = assignments[rank] if rank < len(assignments) else []
        indices = self._indices_from_units(list(units), rng)
        if not equalize or self.world_size == 1:
            return indices
        target_len = self._rank_num_samples(epoch)
        if len(indices) > target_len:
            return indices[:target_len]
        if len(indices) < target_len:
            if not indices and self._episode_records:
                fallback = self._episode_records[rank % len(self._episode_records)]
                indices = list(range(fallback['start'], fallback['end']))
            if indices:
                repeats = math.ceil((target_len - len(indices)) / len(indices))
                indices = indices + (indices * repeats)[:target_len - len(indices)]
        return indices

    def _build_shard_units(self) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for (shard_path, positions) in self._shard_to_episode_positions.items():
            if not self.virtual_shard_max_samples:
                units.append({'unit_id': str(shard_path), 'shard_path': str(shard_path), 'unit_index': 0, 'episode_positions': list(positions)})
                continue
            current: list[int] = []
            current_samples = 0
            unit_index = 0
            for episode_pos in positions:
                episode_samples = self._episode_num_samples(episode_pos)
                if current and current_samples + episode_samples > self.virtual_shard_max_samples:
                    units.append({'unit_id': f'{shard_path}::virtual{unit_index}', 'shard_path': str(shard_path), 'unit_index': unit_index, 'episode_positions': current})
                    unit_index += 1
                    current = []
                    current_samples = 0
                current.append(episode_pos)
                current_samples += episode_samples
            if current:
                units.append({'unit_id': f'{shard_path}::virtual{unit_index}', 'shard_path': str(shard_path), 'unit_index': unit_index, 'episode_positions': current})
        return units

    def _unit_assignments(self, epoch: int) -> list[list[dict[str, Any]]]:
        if not hasattr(self, '_shard_units'):
            return [[] for _ in range(self.world_size)]
        rng = random.Random(self.seed + int(epoch))
        units = list(self._shard_units)
        if self.shuffle_shards:
            rng.shuffle(units)
        if self.shard_partition == 'round_robin':
            return [units[rank::self.world_size] for rank in range(self.world_size)]
        ranked_units = sorted(units, key=lambda unit: self._unit_sample_counts.get(str(unit['unit_id']), 0), reverse=True)
        assignments: list[list[dict[str, Any]]] = [[] for _ in range(self.world_size)]
        loads = [0 for _ in range(self.world_size)]
        for unit in ranked_units:
            rank = min(range(self.world_size), key=lambda item: (loads[item], item))
            assignments[rank].append(unit)
            loads[rank] += int(self._unit_sample_counts.get(str(unit['unit_id']), 0))
        if self.shuffle_shards:
            for local_units in assignments:
                rng.shuffle(local_units)
        return assignments

    def _indices_from_units(self, units: list[dict[str, Any]], rng: random.Random) -> list[int]:
        pending = list(units)
        active: list[dict[str, Any]] = []
        for _ in range(min(self.active_shards_per_rank, len(pending))):
            active.append(self._activate_unit(pending.pop(0), rng))
        indices: list[int] = []
        while active:
            active_pos = rng.randrange(len(active))
            unit_state = active[active_pos]
            blocks = unit_state['blocks']
            block_pos = unit_state['cursor']
            indices.extend(blocks[block_pos])
            unit_state['cursor'] = block_pos + 1
            if unit_state['cursor'] >= len(blocks):
                active.pop(active_pos)
                if pending:
                    active.append(self._activate_unit(pending.pop(0), rng))
        return indices

    def _activate_unit(self, unit: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        blocks = self._unit_blocks(unit, rng)
        if self.shuffle_blocks:
            rng.shuffle(blocks)
        return {'unit': unit, 'blocks': blocks, 'cursor': 0}

    def _unit_blocks(self, unit: dict[str, Any], rng: random.Random) -> list[list[int]]:
        episode_positions = list(unit['episode_positions'])
        if self.shuffle_episodes:
            rng.shuffle(episode_positions)
        blocks: list[list[int]] = []
        for episode_pos in episode_positions:
            record = self._episode_records[episode_pos]
            episode_indices = list(range(int(record['start']), int(record['end'])))
            for offset in range(0, len(episode_indices), self.block_size):
                block = episode_indices[offset:offset + self.block_size]
                if self.shuffle_within_block:
                    rng.shuffle(block)
                if block:
                    blocks.append(block)
        return blocks

    def _episode_num_samples(self, episode_pos: int) -> int:
        record = self._episode_records[episode_pos]
        return int(record['end']) - int(record['start'])

    def debug_shard_summary(self, *, max_shards: int=16) -> dict[str, Any]:
        base = super().debug_shard_summary(max_shards=max_shards)
        assignments = self._unit_assignments(self.epoch)
        unit_loads = [sum((self._unit_sample_counts.get(str(unit['unit_id']), 0) for unit in rank_units)) for rank_units in assignments]
        base.update({'partition_strategy': 'shard_buffered_block_random', 'active_shards_per_rank': self.active_shards_per_rank, 'block_size': self.block_size, 'virtual_shard_max_samples': self.virtual_shard_max_samples, 'shuffle_blocks': self.shuffle_blocks, 'shuffle_within_block': self.shuffle_within_block, 'num_shard_units': len(self._shard_units), 'rank_unit_loads': unit_loads, 'rank_unit_load_min': min(unit_loads) if unit_loads else 0, 'rank_unit_load_max': max(unit_loads) if unit_loads else 0})
        return base

class MixtureShardLocalSampler(ShardBufferedBlockRandomSampler):

    def __init__(self, dataset: CanonicalEmbodiedDataset, *, dataset_weights: Mapping[str, float] | None=None, replacement: bool=True, epoch_length: int | str | None=None, value_weight: Mapping[str, Any] | None=None, phase_repeat: Mapping[str, Any] | None=None, active_shards_per_rank: int=4, block_size: int=32, virtual_shard_max_samples: int | None=None, shuffle_shards: bool=True, shuffle_episodes: bool=True, shuffle_blocks: bool=True, shuffle_within_block: bool=False, shard_partition: str='balanced', seed: int=0, rank: int | None=None, world_size: int | None=None, batch_size: int=1, drop_last: bool=False) -> None:
        self.replacement = bool(replacement)
        if not self.replacement:
            raise ValueError('MixtureShardLocalSampler currently requires replacement=true.')
        self._configured_epoch_length = epoch_length
        self._value_weight_cfg = self._validate_value_weight_config(value_weight)
        self._phase_repeat_cfg = self._validate_phase_repeat_config(phase_repeat)
        self.anchor_multiplicity = self._build_phase_repeat_multiplicity(dataset, self._phase_repeat_cfg)
        self._weighted_prefix = np.r_[0, np.cumsum(self.anchor_multiplicity)] if self.anchor_multiplicity is not None else None
        super().__init__(dataset, active_shards_per_rank=active_shards_per_rank, block_size=block_size, virtual_shard_max_samples=virtual_shard_max_samples, shuffle_shards=shuffle_shards, shuffle_episodes=shuffle_episodes, shuffle_blocks=shuffle_blocks, shuffle_within_block=shuffle_within_block, shard_partition=shard_partition, seed=seed, rank=rank, world_size=world_size, batch_size=batch_size, drop_last=drop_last)
        self._source_units = self._build_source_units()
        if not self._source_units:
            raise ValueError('MixtureShardLocalSampler found no non-empty source datasets.')
        raw_weights = {name: float((dataset_weights or {}).get(name, 1.0)) for name in self._source_units}
        invalid = {name: weight for (name, weight) in raw_weights.items() if weight <= 0 or not math.isfinite(weight)}
        if invalid:
            raise ValueError(f'Invalid source dataset weights: {invalid}')
        total_weight = sum(raw_weights.values())
        self._source_sampling_weights = {name: weight / total_weight for (name, weight) in raw_weights.items()}
        self._global_len = self._resolve_epoch_length(epoch_length)
        self._num_samples = self._rank_num_samples(self.epoch)

    @staticmethod
    def _validate_phase_repeat_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {'enabled': False}
        if not isinstance(config, Mapping):
            raise TypeError('sampler.phase_repeat must be a mapping.')
        unknown = set(config) - {'enabled', 'phase_ranges', 'repeat_factor', 'source_names'}
        if unknown:
            raise ValueError(f'Unsupported sampler.phase_repeat key(s): {sorted(unknown)}')
        enabled = bool(config.get('enabled', True))
        if not enabled:
            return {'enabled': False}
        phase_ranges = str(config.get('phase_ranges', '')).strip()
        if not phase_ranges:
            raise ValueError('sampler.phase_repeat.phase_ranges is required when enabled')
        repeat_factor = int(config.get('repeat_factor', 3))
        if repeat_factor != 3:
            raise ValueError('sampler.phase_repeat only supports repeat_factor=3')
        raw_source_names = config.get('source_names', ())
        if isinstance(raw_source_names, (str, bytes)):
            raw_source_names = [raw_source_names]
        source_names = tuple((str(name) for name in raw_source_names))
        if not source_names or len(set(source_names)) != len(source_names):
            raise ValueError('sampler.phase_repeat.source_names must be a non-empty unique sequence')
        return {'enabled': True, 'phase_ranges': phase_ranges, 'repeat_factor': repeat_factor, 'source_names': source_names}

    @classmethod
    def _build_phase_repeat_multiplicity(cls, dataset: CanonicalEmbodiedDataset, config: Mapping[str, Any]) -> np.ndarray | None:
        if not config.get('enabled', False):
            return None
        multiplicity = np.ones(len(dataset), dtype=np.int64)
        ranges_path = Path(str(config['phase_ranges']))
        plan = json.loads(ranges_path.read_text(encoding='utf-8'))
        ranges = plan.get('episodes')
        if not isinstance(ranges, Mapping) or not ranges:
            raise ValueError(f'Phase repeat plan has no episode ranges: {ranges_path}')
        source_names = set(config['source_names'])
        repeat_factor = int(config['repeat_factor'])
        selected_episodes = 0
        matched_plan_keys: set[str] = set()
        episode_ranges: dict[int, tuple[tuple[int, int], ...]] = {}
        for (episode_index, episode) in enumerate(dataset.episodes):
            source_name = _source_dataset_name(episode)
            if source_name not in source_names:
                continue
            aliases = cls._phase_repeat_episode_aliases(episode)
            matching_keys = [key for key in aliases if key in ranges]
            if len(matching_keys) != 1:
                raise ValueError(f"Phase repeat requires exactly one plan entry for every selected-source episode; episode={episode.get('episode_id')!r}, source={source_name!r}, matching_keys={matching_keys!r}")
            key = matching_keys[0]
            num_steps = int((episode.get('step_index') or {}).get('num_steps', 0))
            episode_ranges[episode_index] = cls._normalize_phase_repeat_ranges(ranges[key], episode_id=str(episode.get('episode_id', '')), num_steps=num_steps)
            matched_plan_keys.add(key)
            selected_episodes += 1
        if selected_episodes == 0:
            raise ValueError(f'Phase repeat found no episodes from configured sources {sorted(source_names)!r}')
        unmatched = sorted(set((str(key) for key in ranges)) - matched_plan_keys)
        if unmatched:
            raise ValueError(f'Phase repeat plan contains ranges not matched by configured sources; first unmatched keys: {unmatched[:8]!r}')
        if dataset.representation_transform is not None:
            step_stride = int((dataset.state_action_config.get('sampling') or {}).get('anchor_stride', 1))
        else:
            step_stride = 1 if dataset.sample_window_size <= 1 else int(dataset.sample_stride)
        for (compact_pos, compact) in enumerate(dataset.episode_index):
            bounds = episode_ranges.get(int(compact.episode_index))
            if bounds is None:
                continue
            local_indices = compact.sample_local_indices
            if local_indices is None:
                local_indices_array = np.arange(int(compact.num_samples), dtype=np.int64)
            else:
                local_indices_array = np.asarray(local_indices, dtype=np.int64)
            step_indices = local_indices_array * step_stride
            selected = np.zeros(step_indices.shape, dtype=np.bool_)
            for (begin, end) in bounds:
                selected |= (step_indices >= begin) & (step_indices < end)
            selected_positions = np.flatnonzero(selected)
            global_start = int(dataset.sample_cumsum[compact_pos])
            multiplicity[global_start + selected_positions] = repeat_factor
        if not np.any(multiplicity == repeat_factor):
            raise ValueError('Phase repeat plan matched episodes but no legal training anchors')
        return multiplicity

    @staticmethod
    def _normalize_phase_repeat_ranges(entry: Any, *, episode_id: str, num_steps: int) -> tuple[tuple[int, int], ...]:
        if not isinstance(entry, Mapping):
            raise ValueError(f'Phase repeat entry for episode={episode_id!r} must be a mapping')
        has_range = 'range' in entry
        has_ranges = 'ranges' in entry
        if has_range == has_ranges:
            raise ValueError(f"Phase repeat entry for episode={episode_id!r} must define exactly one of 'range' or 'ranges'")
        raw_ranges = [entry['range']] if has_range else entry['ranges']
        if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)) or (not raw_ranges):
            raise ValueError(f'Phase repeat ranges for episode={episode_id!r} must be a non-empty sequence')
        normalized: list[tuple[int, int]] = []
        for raw_bounds in raw_ranges:
            if not isinstance(raw_bounds, Sequence) or isinstance(raw_bounds, (str, bytes)) or len(raw_bounds) != 2:
                raise ValueError(f'Phase repeat range for episode={episode_id!r} must contain [begin, end]')
            (begin, end) = (int(value) for value in raw_bounds)
            if not 0 <= begin < end <= num_steps:
                raise ValueError(f'Invalid phase repeat range [{begin}, {end}) for episode={episode_id!r} with {num_steps} steps')
            if normalized and begin < normalized[-1][1]:
                raise ValueError(f'Phase repeat ranges for episode={episode_id!r} must be ordered and non-overlapping')
            normalized.append((begin, end))
        return tuple(normalized)

    @staticmethod
    def _phase_repeat_episode_aliases(episode: Mapping[str, Any]) -> tuple[str, ...]:
        storage_extra = (episode.get('storage') or {}).get('extra') or {}
        source_info = episode.get('source_info') or {}
        candidates = (episode.get('episode_id'), source_info.get('original_episode_id'), storage_extra.get('episode_id'))
        aliases: list[str] = []
        for candidate in candidates:
            if candidate is None:
                continue
            value = str(candidate)
            if value not in aliases:
                aliases.append(value)
        return tuple(aliases)

    def _episode_num_samples(self, episode_pos: int) -> int:
        if not self._phase_repeat_cfg.get('enabled', False):
            return super()._episode_num_samples(episode_pos)
        assert self._weighted_prefix is not None
        record = self._episode_records[episode_pos]
        return int(self._weighted_prefix[int(record['end'])] - self._weighted_prefix[int(record['start'])])

    def _unit_blocks(self, unit: dict[str, Any], rng: random.Random) -> list[list[int]]:
        blocks = super()._unit_blocks(unit, rng)
        if not self._phase_repeat_cfg.get('enabled', False):
            return blocks
        assert self.anchor_multiplicity is not None
        repeated: list[list[int]] = []
        for block in blocks:
            groups = itertools.groupby(block, key=lambda index: int(self.anchor_multiplicity[index]))
            for (weight, members) in groups:
                piece = list(members)
                repeated.extend((piece.copy() for _ in range(weight)))
        return repeated

    @property
    def source_datasets(self) -> list[str]:
        return list(self._source_units)

    @property
    def source_sampling_weights(self) -> dict[str, float]:
        return dict(self._source_sampling_weights)

    def _rank_num_samples(self, epoch: int) -> int:
        del epoch
        if not hasattr(self, '_global_len'):
            return 0
        if self.world_size == 1:
            return self._global_len
        global_batch = self.batch_size * self.world_size
        (full_batches, remainder) = divmod(self._global_len, global_batch)
        count = full_batches * self.batch_size
        if remainder == 0 or self.drop_last:
            return count
        rank_start = self.rank * self.batch_size
        extra = max(0, min(self.batch_size, remainder - rank_start))
        return count + extra

    def _indices_for_rank(self, epoch: int, rank: int, *, equalize: bool=True) -> list[int]:
        del equalize
        target_len = self._rank_num_samples(epoch)
        if target_len <= 0:
            return []
        rng = random.Random(_deterministic_safe_hash(('mixture_shard_local', self.seed, int(epoch), int(rank))))
        source_states = self._initial_source_states(epoch=int(epoch), rank=int(rank), rng=rng)
        indices: list[int] = []
        while len(indices) < target_len:
            source_name = self._choose_active_source(source_states, rng)
            if source_name is None:
                break
            block = self._next_source_block(source_states[source_name], rng)
            if not block:
                continue
            indices.extend(block)
        return indices[:target_len]

    def _resolve_epoch_length(self, epoch_length: int | str | None) -> int:
        total = sum((int(self._unit_sample_counts.get(str(unit['unit_id']), 0)) for units in self._source_units.values() for unit in units))
        if epoch_length is None or str(epoch_length).lower() == 'auto':
            return total
        return max(1, int(epoch_length))

    def _build_source_units(self) -> dict[str, list[dict[str, Any]]]:
        source_units: dict[str, list[dict[str, Any]]] = {}
        for unit in self._shard_units:
            positions = list(unit.get('episode_positions', []))
            if not positions:
                continue
            positions_by_source: dict[str, list[int]] = {}
            for position in positions:
                episode = self.dataset.episodes[int(self._episode_records[int(position)]['episode_index'])]
                source_name = _source_dataset_name(episode)
                positions_by_source.setdefault(source_name, []).append(int(position))
            for (source_name, source_positions) in positions_by_source.items():
                source_unit = dict(unit)
                source_unit['episode_positions'] = source_positions
                if len(positions_by_source) > 1:
                    source_unit['unit_id'] = f"{unit['unit_id']}::sampling_source={source_name}"
                    self._unit_sample_counts[str(source_unit['unit_id'])] = sum((self._episode_num_samples(position) for position in source_positions))
                source_units.setdefault(source_name, []).append(source_unit)
        return source_units

    def _initial_source_states(self, *, epoch: int, rank: int, rng: random.Random) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for (source_name, units) in self._source_units.items():
            assigned = self._assign_source_units(units, epoch=epoch, rank=rank)
            if not assigned:
                warnings.warn(f'MixtureShardLocalSampler source {source_name!r} has fewer shard units than world_size={self.world_size}; rank={rank} will reuse all source units, which may duplicate IO across ranks. Lower virtual_shard_max_samples or increase source shard count to preserve strict rank-local reads.', RuntimeWarning, stacklevel=2)
                assigned = list(units)
            if self.shuffle_shards:
                rng.shuffle(assigned)
            states[source_name] = {'source': source_name, 'template_units': list(assigned)}
            self._reset_source_state(states[source_name], rng)
        return states

    def _assign_source_units(self, units: list[dict[str, Any]], *, epoch: int, rank: int) -> list[dict[str, Any]]:
        rng = random.Random(_deterministic_safe_hash(('source_unit_assign', self.seed, int(epoch), len(units))))
        ordered = list(units)
        if self.shuffle_shards:
            rng.shuffle(ordered)
        if self.shard_partition == 'round_robin':
            return ordered[rank::self.world_size]
        ranked_units = sorted(ordered, key=lambda unit: self._unit_sample_counts.get(str(unit['unit_id']), 0), reverse=True)
        assignments: list[list[dict[str, Any]]] = [[] for _ in range(self.world_size)]
        loads = [0 for _ in range(self.world_size)]
        for unit in ranked_units:
            target_rank = min(range(self.world_size), key=lambda item: (loads[item], item))
            assignments[target_rank].append(unit)
            loads[target_rank] += int(self._unit_sample_counts.get(str(unit['unit_id']), 0))
        return assignments[rank] if rank < len(assignments) else []

    def _reset_source_state(self, state: dict[str, Any], rng: random.Random) -> None:
        pending = list(state['template_units'])
        if self.shuffle_shards:
            rng.shuffle(pending)
        state['pending'] = pending
        state['active'] = []
        self._fill_active_source_units(state, rng)

    def _fill_active_source_units(self, state: dict[str, Any], rng: random.Random) -> None:
        pending = state['pending']
        active = state['active']
        while len(active) < self.active_shards_per_rank and pending:
            active.append(self._activate_unit(pending.pop(0), rng))

    def _activate_unit(self, unit: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        state = super()._activate_unit(unit, rng)
        if self._value_weight_enabled:
            weights = [self._block_value_weight(block) for block in state['blocks']]
            if not any((weight > 0 for weight in weights)):
                weights = [1.0 for _ in state['blocks']]
            state['block_weights'] = weights
            state['available_block_positions'] = list(range(len(state['blocks'])))
            state['remaining_draws'] = len(state['blocks'])
        return state

    def _choose_active_source(self, states: Mapping[str, dict[str, Any]], rng: random.Random) -> str | None:
        active_sources = [name for (name, state) in states.items() if state.get('active') or state.get('pending') or state.get('template_units')]
        if not active_sources:
            return None
        weights = [self._source_sampling_weights[name] for name in active_sources]
        total = sum(weights)
        pick = rng.random() * total
        cumulative = 0.0
        for (name, weight) in zip(active_sources, weights, strict=True):
            cumulative += weight
            if pick <= cumulative:
                return name
        return active_sources[-1]

    def _next_source_block(self, state: dict[str, Any], rng: random.Random) -> list[int]:
        self._fill_active_source_units(state, rng)
        active = state['active']
        if not active:
            self._reset_source_state(state, rng)
            active = state['active']
            if not active:
                return []
        active_pos = self._choose_active_unit_pos(active, rng)
        unit_state = active[active_pos]
        blocks = unit_state['blocks']
        block_pos = self._choose_block_pos(unit_state, rng)
        block = list(blocks[block_pos])
        if self._value_weight_enabled:
            unit_state['remaining_draws'] = int(unit_state.get('remaining_draws', len(blocks))) - 1
        else:
            unit_state['cursor'] = block_pos + 1
        if self._value_weight_enabled and unit_state['remaining_draws'] <= 0 or (not self._value_weight_enabled and unit_state['cursor'] >= len(blocks)):
            active.pop(active_pos)
            self._fill_active_source_units(state, rng)
        return block

    @property
    def _value_weight_enabled(self) -> bool:
        return bool(self._value_weight_cfg.get('enabled', False))

    @staticmethod
    def _validate_value_weight_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
        if config is None:
            return {'enabled': False, 'key': 'value_weight', 'default': 1.0, 'mode': 'block_mean', 'exponent': 1.0, 'replacement': True, 'clip': None}
        if not isinstance(config, Mapping):
            raise TypeError('sampler.value_weight must be a mapping.')
        mode = str(config.get('mode', 'block_mean')).lower()
        if mode not in {'block_mean', 'block_sum', 'block_max'}:
            raise ValueError('sampler.value_weight.mode must be one of: block_mean, block_sum, block_max.')
        default = float(config.get('default', 1.0))
        exponent = float(config.get('exponent', 1.0))
        if default < 0:
            raise ValueError('sampler.value_weight.default must be non-negative.')
        if exponent < 0 or not math.isfinite(exponent):
            raise ValueError('sampler.value_weight.exponent must be finite and non-negative.')
        clip = config.get('clip', None)
        if clip is not None:
            if not isinstance(clip, (list, tuple)) or len(clip) != 2:
                raise ValueError('sampler.value_weight.clip must be a [min, max] sequence.')
            clip = (float(clip[0]), float(clip[1]))
            if not math.isfinite(clip[0]) or not math.isfinite(clip[1]) or clip[0] < 0 or (clip[1] < clip[0]):
                raise ValueError('sampler.value_weight.clip must be finite and satisfy 0 <= min <= max.')
        return {'enabled': bool(config.get('enabled', True)), 'key': str(config.get('key', 'value_weight')), 'default': default, 'mode': mode, 'exponent': exponent, 'replacement': bool(config.get('replacement', True)), 'clip': clip}

    def _block_value_weight(self, block: list[int]) -> float:
        if not block:
            return 0.0
        key = str(self._value_weight_cfg.get('key', 'value_weight'))
        default = float(self._value_weight_cfg.get('default', 1.0))
        values = [float(self.dataset.sample_value_weight(index, key=key, default=default)) for index in block]
        clip = self._value_weight_cfg.get('clip')
        if clip is not None:
            (lo, hi) = (float(clip[0]), float(clip[1]))
            values = [min(max(value, lo), hi) for value in values]
        mode = str(self._value_weight_cfg.get('mode', 'block_mean'))
        if mode == 'block_sum':
            weight = sum(values)
        elif mode == 'block_max':
            weight = max(values)
        else:
            weight = sum(values) / float(len(values))
        exponent = float(self._value_weight_cfg.get('exponent', 1.0))
        return max(0.0, weight) ** exponent

    def _choose_active_unit_pos(self, active: list[dict[str, Any]], rng: random.Random) -> int:
        if not self._value_weight_enabled:
            return rng.randrange(len(active))
        weights = []
        for unit_state in active:
            block_weights = unit_state.get('block_weights', [])
            positions = self._available_block_positions(unit_state)
            weights.append(sum((float(block_weights[pos]) for pos in positions)))
        return self._weighted_choice_pos(weights, rng)

    def _choose_block_pos(self, unit_state: dict[str, Any], rng: random.Random) -> int:
        if not self._value_weight_enabled:
            return int(unit_state['cursor'])
        positions = self._available_block_positions(unit_state)
        weights = [float(unit_state.get('block_weights', [])[pos]) for pos in positions]
        block_pos = positions[self._weighted_choice_pos(weights, rng)]
        if not bool(self._value_weight_cfg.get('replacement', True)):
            positions.remove(block_pos)
        return block_pos

    def _available_block_positions(self, unit_state: dict[str, Any]) -> list[int]:
        if bool(self._value_weight_cfg.get('replacement', True)):
            return list(range(len(unit_state.get('blocks', []))))
        return unit_state.setdefault('available_block_positions', list(range(len(unit_state.get('blocks', [])))))

    @staticmethod
    def _weighted_choice_pos(weights: list[float], rng: random.Random) -> int:
        if not weights:
            raise ValueError('Cannot choose from an empty weight list.')
        total = sum((weight for weight in weights if weight > 0))
        if total <= 0 or not math.isfinite(total):
            return rng.randrange(len(weights))
        pick = rng.random() * total
        cumulative = 0.0
        for (pos, weight) in enumerate(weights):
            if weight <= 0:
                continue
            cumulative += weight
            if pick <= cumulative:
                return pos
        return len(weights) - 1

    def debug_shard_summary(self, *, max_shards: int=16) -> dict[str, Any]:
        base = super().debug_shard_summary(max_shards=max_shards)
        phase_repeat = dict(self._phase_repeat_cfg)
        if self.anchor_multiplicity is not None:
            phase_repeat.update(repeated_anchors=int(np.sum(self.anchor_multiplicity > 1)), ordinary_anchors=int(np.sum(self.anchor_multiplicity == 1)), weighted_round_size=int(self.anchor_multiplicity.sum()))
        base.update({'partition_strategy': 'mixture_shard_local', 'replacement': self.replacement, 'epoch_length': self._global_len, 'source_sampling_weights': self.source_sampling_weights, 'source_num_units': {name: len(units) for (name, units) in self._source_units.items()}, 'value_weight': dict(self._value_weight_cfg), 'phase_repeat': phase_repeat})
        return base

class MemoryMixtureSampler(Sampler[int]):

    def __init__(self, dataset: CanonicalEmbodiedDataset, *, memory_ratio: float=0.5, epoch_length: int | str | None='auto', dataset_weights: Mapping[str, float] | None=None, memory: Mapping[str, Any] | None=None, mixture: Mapping[str, Any] | None=None, seed: int=42, rank: int | None=None, world_size: int | None=None, batch_size: int=1, drop_last: bool=False) -> None:
        if not isinstance(dataset, CanonicalEmbodiedDataset):
            raise TypeError('MemoryMixtureSampler requires CanonicalEmbodiedDataset')
        self.dataset = dataset
        self.memory_ratio = float(memory_ratio)
        if not math.isfinite(self.memory_ratio) or not 0.0 < self.memory_ratio < 1.0:
            raise ValueError('memory_ratio must lie strictly between 0 and 1')
        self.seed = int(seed)
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        if rank is None or world_size is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1:
            raise ValueError(f'world_size must be >= 1, got {self.world_size}')
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f'rank must be in [0, {self.world_size}), got {self.rank}')
        memory_cfg = dict(memory or {})
        mixture_cfg = dict(mixture or {})
        self._reject_reserved_subsampler_keys(memory_cfg, 'memory')
        self._reject_reserved_subsampler_keys(mixture_cfg, 'mixture')
        memory_cfg.setdefault('seed', self.seed)
        memory_cfg.update(rank=self.rank, world_size=self.world_size, batch_size=self.batch_size, drop_last=True)
        mixture_cfg.setdefault('seed', self.seed + 1)
        mixture_cfg.update(rank=self.rank, world_size=self.world_size, batch_size=self.batch_size, drop_last=True)
        self.memory_sampler = MemoryEventBalancedSampler(dataset, **memory_cfg)
        self.mixture_sampler = MixtureShardLocalSampler(dataset, dataset_weights=dataset_weights, **mixture_cfg)
        self._global_batch_size = self.batch_size * self.world_size
        self._global_len = self._resolve_global_length(epoch_length)
        self._num_samples = self._global_len // self.world_size
        self.epoch = 0
        self._set_stream_schedule()

    @staticmethod
    def _reject_reserved_subsampler_keys(config: Mapping[str, Any], name: str) -> None:
        reserved = {'type', 'rank', 'world_size', 'batch_size', 'drop_last', 'dataset_weights'}
        unknown = sorted(reserved.intersection(config))
        if unknown:
            raise ValueError(f'{name} sampler config may not override {unknown}')

    def _resolve_global_length(self, epoch_length: int | str | None) -> int:
        if epoch_length is None or str(epoch_length).lower() == 'auto':
            memory_length = int(self.memory_sampler._global_len)
            mixture_length = int(self.mixture_sampler._global_len)
            requested = max(math.ceil(memory_length / self.memory_ratio), math.ceil(mixture_length / (1.0 - self.memory_ratio)))
            return math.ceil(requested / self._global_batch_size) * self._global_batch_size
        requested = int(epoch_length)
        if requested <= 0:
            raise ValueError('epoch_length must be positive')
        if self.drop_last:
            resolved = requested // self._global_batch_size * self._global_batch_size
        else:
            resolved = math.ceil(requested / self._global_batch_size) * self._global_batch_size
        if resolved <= 0:
            raise ValueError('epoch_length is smaller than one global batch')
        return resolved

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._set_stream_schedule()

    def _set_stream_schedule(self) -> None:
        local_batches = self._num_samples // self.batch_size
        if local_batches < 2:
            raise ValueError('MemoryMixtureSampler epoch_length must provide at least two local batches so both memory and mixture streams are represented')
        memory_batches = round(local_batches * self.memory_ratio)
        memory_batches = min(max(1, memory_batches), local_batches - 1)
        self._stream_schedule = ['memory'] * memory_batches + ['mixture'] * (local_batches - memory_batches)
        random.Random(_deterministic_safe_hash(('memory_mixture_streams', self.seed, self.epoch))).shuffle(self._stream_schedule)

    def __iter__(self) -> Iterator[int]:
        streams = {'memory': self._batch_stream(self.memory_sampler, 'memory'), 'mixture': self._batch_stream(self.mixture_sampler, 'mixture')}
        return (index for stream_name in self._stream_schedule for index in next(streams[stream_name]))

    def _batch_stream(self, sampler: Sampler[int], stream_name: str) -> Iterator[list[int]]:
        cycle = 0
        iterator: Iterator[int] = iter(())
        while True:
            batch: list[int] = []
            while len(batch) < self.batch_size:
                try:
                    batch.append(int(next(iterator)))
                except StopIteration:
                    sampler.set_epoch(_deterministic_safe_hash(('memory_mixture_subepoch', stream_name, self.seed, self.epoch, cycle)))
                    cycle += 1
                    iterator = iter(sampler)
                    try:
                        batch.append(int(next(iterator)))
                    except StopIteration as exc:
                        raise RuntimeError(f'MemoryMixtureSampler {stream_name} stream produced no complete samples') from exc
            yield batch

    @property
    def source_sampling_weights(self) -> dict[str, float]:
        return self.mixture_sampler.source_sampling_weights

    def debug_summary(self) -> dict[str, Any]:
        counts = Counter(self._stream_schedule)
        return {'type': type(self).__name__, 'length': len(self), 'global_length': self._global_len, 'memory_ratio': self.memory_ratio, 'actual_memory_ratio': counts['memory'] / len(self._stream_schedule), 'stream_batches': dict(counts), 'source_sampling_weights': self.source_sampling_weights, 'memory': self.memory_sampler.debug_summary(), 'mixture': self.mixture_sampler.debug_shard_summary(), 'rank': self.rank, 'world_size': self.world_size, 'epoch': self.epoch}

def sampler_debug_summary(sampler: Any, *, max_items: int=32) -> dict[str, Any]:
    indices = list(iter(sampler))[:max_items]
    dataset = getattr(sampler, 'dataset', None)
    refs = []
    if dataset is not None and hasattr(dataset, '_sample_ref_from_index'):
        for idx in indices:
            ref = dataset._sample_ref_from_index(int(idx))
            episode = dataset.episodes[int(ref.episode_index)]
            refs.append({'idx': int(idx), 'episode_index': int(ref.episode_index), 'episode_id': episode.get('episode_id'), 'shard_path': ShardLocalitySampler._episode_shard_path(episode), 'step_index': int(ref.step_index)})
    return {'type': type(sampler).__name__, 'length': len(sampler) if hasattr(sampler, '__len__') else None, 'rank': getattr(sampler, 'rank', None), 'world_size': getattr(sampler, 'world_size', None), 'epoch': getattr(sampler, 'epoch', None), 'indices_head': indices, 'refs_head': refs}
