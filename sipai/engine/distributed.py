from __future__ import annotations
import os
from dataclasses import dataclass, replace
from typing import Any
import torch
import torch.distributed as dist

@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    local_world_size: int = 1
    node_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

def resolve_distributed_context_from_env(launcher: str='none', backend: str | None=None) -> DistributedContext:
    launcher = (launcher or 'none').lower()
    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    enabled = launcher in {'pytorch', 'torchrun', 'env'} or env_world_size > 1
    if not enabled:
        return DistributedContext(enabled=False)
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = env_world_size
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', str(world_size)))
    if local_world_size <= 0:
        raise ValueError(f'LOCAL_WORLD_SIZE must be positive, got {local_world_size}.')
    node_rank = int(os.environ.get('GROUP_RANK', os.environ.get('NODE_RANK', str(rank // local_world_size))))
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size, local_world_size=local_world_size, node_rank=node_rank, backend=backend)

def init_distributed_from_env(launcher: str='none', backend: str | None=None) -> DistributedContext:
    context = resolve_distributed_context_from_env(launcher=launcher, backend=backend)
    if not context.enabled:
        return context
    backend = context.backend or ('nccl' if torch.cuda.is_available() else 'gloo')
    context = replace(context, backend=backend)
    if backend == 'nccl' and torch.cuda.is_available():
        torch.cuda.set_device(context.local_rank)
    if not dist.is_available():
        raise RuntimeError('torch.distributed is not available in this PyTorch build.')
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=context.rank, world_size=context.world_size)
    return context

def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        if context.backend == 'nccl' and torch.cuda.is_available():
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()
