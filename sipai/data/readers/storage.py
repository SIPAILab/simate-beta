from __future__ import annotations
import atexit
import io
import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
logger = logging.getLogger(__name__)

class CacheMissError(RuntimeError):
    pass

def build_storage_reader(data_root: str | Path, backend: str='simate_unified', **reader_options: Any) -> BaseStorageReader:
    backend = (backend or 'simate_unified').lower()
    if backend == 'reference':
        return ReferenceStorageReader(data_root)
    if backend == 'simate_unified':
        allowed_options = {'video_cache_dir', 'table_cache_dir', 'max_cached_tables', 'max_cached_videos', 'max_cached_frames', 'max_cached_frame_bytes', 'max_cached_video_readers', 'max_cached_state_rows', 'max_cached_step_rows', 'video_decoder', 'metrics_dir', 'cache_miss_policy', 'read_mode', 'video_mode', 'video_tmp_dir', 'shard_fd_cache_size', 'member_index_cache_size', 'video_tmp_max_bytes_per_worker', 'video_tmp_required_free_bytes', 'offset_memfd_chunk_bytes', 'max_cached_video_bytes_per_worker'}
        unknown_options = set(reader_options) - allowed_options
        if unknown_options:
            raise ValueError(f'Unsupported SiMate reader options: {sorted(unknown_options)}')
        return SiMateUnifiedStorageReader(data_root, **reader_options)
    raise ValueError(f'Unsupported canonical storage reader backend: {backend}')

@dataclass(frozen=True)
class StreamRef:
    stream_key: str
    path: str
    address: str
    index: Optional[int] = None
    index_range: Optional[tuple[int, int]] = None
    timestamp: Optional[float] = None
    time_range: Optional[tuple[float, float]] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {'stream_key': self.stream_key, 'path': self.path, 'address': self.address}
        if self.index is not None:
            out['index'] = self.index
        if self.index_range is not None:
            out['index_range'] = list(self.index_range)
        if self.timestamp is not None:
            out['timestamp'] = self.timestamp
        if self.time_range is not None:
            out['time_range'] = list(self.time_range)
        return out

class BaseStorageReader:

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)

    def read(self, episode: Mapping[str, Any], stream_key: str, *, index: Optional[int]=None, index_range: Optional[Iterable[int]]=None, timestamp: Optional[float]=None, time_range: Optional[Iterable[float]]=None, interpolate: bool=False) -> Any:
        raise NotImplementedError

    def _stream(self, episode: Mapping[str, Any], stream_key: str) -> Mapping[str, Any]:
        streams = episode.get('storage', {}).get('streams', {})
        if stream_key not in streams:
            raise KeyError(f'Stream `{stream_key}` is not declared in episode.storage.streams')
        return streams[stream_key]

    def cache_stats(self) -> Dict[str, Any]:
        return {}

    def read_many(self, episode: Mapping[str, Any], stream_key: str, *, indices: Iterable[int]) -> List[Any]:
        return [self.read(episode, stream_key, index=int(index)) for index in indices]

    def step_table_value(self, episode: Mapping[str, Any], column: str, step_index: int) -> Any:
        raise NotImplementedError(f'{type(self).__name__} does not expose step table values')

    def step_table_values(self, episode: Mapping[str, Any], column: str, step_indices: Iterable[int]) -> List[Any]:
        return [self.step_table_value(episode, column, int(step_index)) for step_index in step_indices]

    def stream_length(self, episode: Mapping[str, Any], stream_key: str) -> int | None:
        return None

    def _stream_path(self, episode: Mapping[str, Any], stream: Mapping[str, Any]) -> Path:
        storage = episode.get('storage', {})
        root = storage.get('root', '')
        path = stream.get('path', '')
        return self.data_root / root / path

class ReferenceStorageReader(BaseStorageReader):

    def read(self, episode: Mapping[str, Any], stream_key: str, *, index: Optional[int]=None, index_range: Optional[Iterable[int]]=None, timestamp: Optional[float]=None, time_range: Optional[Iterable[float]]=None, interpolate: bool=False) -> Dict[str, Any]:
        stream = self._stream(episode, stream_key)
        path = str(self._stream_path(episode, stream))
        address = str(stream.get('address', ''))
        int_range = _to_int_range(index_range)
        if int_range is not None:
            (start, end) = int_range
            return [StreamRef(stream_key=stream_key, path=path, address=address, index=idx, index_range=int_range, timestamp=timestamp, time_range=_to_float_range(time_range)).as_dict() for idx in range(start, end)]
        out = StreamRef(stream_key=stream_key, path=path, address=address, index=index, timestamp=timestamp, time_range=_to_float_range(time_range)).as_dict()
        if interpolate:
            out['interpolate'] = True
        return out

class _LruCache:

    def __init__(self, max_items: int, on_evict: Callable[[Any], None] | None=None):
        self.max_items = max(0, int(max_items))
        self.on_evict = on_evict
        self._items: OrderedDict[Any, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: Any) -> bool:
        return key in self._items

    def get(self, key: Any) -> Any | None:
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: Any, value: Any) -> list[Any]:
        old = self._items.pop(key, None)
        if old is not None and old is not value and (self.on_evict is not None):
            self.on_evict(old)
        if self.max_items <= 0:
            if self.on_evict is not None:
                self.on_evict(value)
            return [value]
        self._items[key] = value
        self._items.move_to_end(key)
        return self.evict_until()

    def pop(self, key: Any) -> Any | None:
        return self._items.pop(key, None)

    def evict_until(self, predicate: Callable[[], bool] | None=None) -> list[Any]:
        evicted: list[Any] = []
        while self._items and (len(self._items) > self.max_items or (predicate is not None and predicate())):
            (_, value) = self._items.popitem(last=False)
            evicted.append(value)
            if self.on_evict is not None:
                self.on_evict(value)
        return evicted

    def clear(self) -> None:
        values = list(self._items.values())
        self._items.clear()
        if self.on_evict is not None:
            for value in values:
                self.on_evict(value)

    def values(self) -> list[Any]:
        return list(self._items.values())

@dataclass
class VideoCacheEntry:
    mode: str
    path: Path
    size: int = 0
    fd: int | None = None
    owned_file: bool = False

    @property
    def alive(self) -> bool:
        if self.fd is not None and self.fd < 0:
            return False
        if self.owned_file and (not self.path.exists()):
            return False
        return True

    def close_fd(self) -> None:
        if self.fd is None or self.fd < 0:
            return
        try:
            os.close(self.fd)
        finally:
            self.fd = -1

class _OffsetShardStore:

    def __init__(self, owner: 'SiMateUnifiedStorageReader'):
        self.owner = owner

    def member_index_path(self, shard_path: str) -> Path:
        shard = self.owner.data_root / shard_path
        return shard.with_suffix('.members.parquet')

    def member_index(self, shard_path: str) -> Dict[str, Dict[str, Any]]:
        owner = self.owner
        if shard_path in owner._member_index_cache:
            owner._stats['member_index_cache_hits'] += 1
            owner._member_index_cache.move_to_end(shard_path)
            return owner._member_index_cache[shard_path]
        owner._stats['member_index_cache_misses'] += 1
        import pyarrow.parquet as pq
        sidecar = self.member_index_path(shard_path)
        if not sidecar.exists():
            raise FileNotFoundError(f'SiMate member offset sidecar does not exist: {sidecar}')
        rows = pq.read_table(sidecar).to_pylist()
        members = {str(row['member_path']): dict(row) for row in rows}
        if len(members) != len(rows):
            raise ValueError(f'Duplicate member_path values in {sidecar}')
        shard_size = (owner.data_root / shard_path).stat().st_size
        for (member_path, row) in members.items():
            offset = int(row.get('offset_data') or 0)
            size = int(row.get('size') or 0)
            if offset < 0 or size < 0 or offset + size > shard_size:
                raise ValueError(f'Invalid offset range for {shard_path}:{member_path}: offset_data={offset}, size={size}, shard_size={shard_size}')
        owner._member_index_cache[shard_path] = members
        owner._member_index_cache.move_to_end(shard_path)
        owner._stats['member_index_load_count'] += 1
        owner._stats['member_index_rows'] += len(rows)
        while len(owner._member_index_cache) > owner.member_index_cache_size:
            owner._member_index_cache.popitem(last=False)
            owner._stats['member_index_cache_evictions'] += 1
        return members

    def member_info(self, shard_path: str, member_path: str) -> Dict[str, Any]:
        members = self.member_index(shard_path)
        try:
            return members[member_path]
        except KeyError as exc:
            raise FileNotFoundError(f'{member_path} is missing from {self.member_index_path(shard_path)}') from exc

    def shard_fd(self, shard_path: str) -> Any:
        owner = self.owner
        if owner.shard_fd_cache_size > 0 and shard_path in owner._shard_fd_cache:
            owner._stats['shard_fd_cache_hits'] += 1
            owner._shard_fd_cache.move_to_end(shard_path)
            return owner._shard_fd_cache[shard_path]
        owner._stats['shard_fd_cache_misses'] += 1
        fd = (owner.data_root / shard_path).open('rb')
        owner._stats['shard_fd_open_count'] += 1
        if owner.shard_fd_cache_size <= 0:
            return fd
        owner._shard_fd_cache[shard_path] = fd
        owner._shard_fd_cache.move_to_end(shard_path)
        while len(owner._shard_fd_cache) > owner.shard_fd_cache_size:
            (_, evicted) = owner._shard_fd_cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
            owner._stats['shard_fd_evictions'] += 1
        return fd

    def close(self) -> None:
        owner = self.owner
        for fd in list(owner._shard_fd_cache.values()):
            try:
                fd.close()
            except Exception:
                pass
        owner._shard_fd_cache.clear()

    def read_member_bytes(self, shard_path: str, member_path: str) -> bytes:
        owner = self.owner
        if owner.read_mode == 'offset':
            info = self.member_info(shard_path, member_path)
            offset = int(info['offset_data'])
            size = int(info['size'])
            t0 = time.perf_counter()
            fd = self.shard_fd(shard_path)
            close_after = owner.shard_fd_cache_size <= 0
            try:
                data = os.pread(fd.fileno(), size, offset)
            finally:
                if close_after:
                    try:
                        fd.close()
                    except Exception:
                        pass
            if len(data) != size:
                raise OSError(f'Incomplete offset read for {shard_path}:{member_path}: got {len(data)} of {size}')
            elapsed = time.perf_counter() - t0
            owner._stats['offset_member_reads'] += 1
            owner._stats['offset_member_read_bytes'] += size
            owner._stats['offset_member_read_sec_x1e6'] += int(elapsed * 1000000)
            return data
        shard = owner.data_root / shard_path
        owner._stats['tar_open_count'] += 1
        with tarfile.open(shard) as tar:
            extracted = tar.extractfile(member_path)
            if extracted is None:
                raise FileNotFoundError(f'{member_path} is missing from {shard}')
            return extracted.read()

    def copy_member_to_fd(self, shard_path: str, member_path: str, out_fd: int, *, chunk_bytes: int | None=None) -> int:
        owner = self.owner
        info = self.member_info(shard_path, member_path)
        offset = int(info['offset_data'])
        size = int(info['size'])
        chunk_size = max(1, int(chunk_bytes or owner.offset_memfd_chunk_bytes))
        fd = self.shard_fd(shard_path)
        close_after = owner.shard_fd_cache_size <= 0
        copied = 0
        chunks = 0
        t0 = time.perf_counter()
        try:
            while copied < size:
                want = min(chunk_size, size - copied)
                chunk = os.pread(fd.fileno(), want, offset + copied)
                if not chunk:
                    raise OSError(f'Incomplete offset read for {shard_path}:{member_path}: got {copied} of {size}')
                owner._write_all_to_fd(out_fd, chunk)
                copied += len(chunk)
                chunks += 1
        finally:
            if close_after:
                try:
                    fd.close()
                except Exception:
                    pass
        if copied != size:
            raise OSError(f'Incomplete offset copy for {shard_path}:{member_path}: got {copied} of {size}')
        elapsed = time.perf_counter() - t0
        owner._stats['offset_member_reads'] += 1
        owner._stats['offset_member_read_bytes'] += size
        owner._stats['offset_member_read_chunks'] += chunks
        owner._stats['offset_member_read_sec_x1e6'] += int(elapsed * 1000000)
        return size

class _TableStore:

    def __init__(self, owner: 'SiMateUnifiedStorageReader'):
        self.owner = owner

    def table(self, shard_path: str, member_path: str) -> Any:
        import pyarrow.parquet as pq
        owner = self.owner
        key = (shard_path, member_path)
        if key in owner._table_cache:
            owner._stats['table_cache_hits'] += 1
            owner._table_cache.move_to_end(key)
            return owner._table_cache[key]
        owner._stats['table_cache_misses'] += 1
        if owner.read_mode == 'offset':
            data = owner._read_member_bytes(shard_path, member_path)
            owner._stats['offset_table_reads'] += 1
            table = pq.read_table(io.BytesIO(data))
        elif owner.table_cache_dir is not None:
            table_path = self.materialized_table(shard_path, member_path)
            table = pq.read_table(table_path)
        else:
            owner._stats['table_tar_open_count'] += 1
            table = pq.read_table(io.BytesIO(owner._read_member_bytes(shard_path, member_path)))
        owner._table_cache[key] = table
        owner._table_cache.move_to_end(key)
        while len(owner._table_cache) > owner.max_cached_tables:
            owner._table_cache.popitem(last=False)
        return table

    def materialized_table(self, shard_path: str, member_path: str) -> Path:
        owner = self.owner
        if owner.table_cache_dir is None:
            raise RuntimeError('table_cache_dir is not configured')
        digest = hashlib.sha1(f'{shard_path}:{member_path}'.encode('utf-8')).hexdigest()[:16]
        suffix = Path(member_path).suffix or '.parquet'
        out_path = owner.table_cache_dir / f'{digest}_{Path(member_path).name}'
        if out_path.exists():
            return out_path
        owner._handle_cache_miss('table', shard_path, member_path, out_path)
        shard = owner.data_root / shard_path
        tmp_path = out_path.with_suffix(f'{suffix}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        owner._stats['tar_open_count'] += 1
        owner._stats['table_tar_open_count'] += 1
        owner._stats['table_materialize_count'] += 1
        try:
            with tarfile.open(shard) as tar:
                info = tar.getmember(member_path)
                extracted = tar.extractfile(info)
                if extracted is None:
                    raise FileNotFoundError(f'{member_path} is missing from {shard}')
                with tmp_path.open('wb') as f:
                    while True:
                        chunk = extracted.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            if tmp_path.stat().st_size != info.size:
                raise OSError(f'Incomplete table cache write for {member_path}')
            owner._stats['table_materialize_bytes'] += int(info.size)
            try:
                tmp_path.replace(out_path)
            except FileExistsError:
                tmp_path.unlink(missing_ok=True)
        finally:
            tmp_path.unlink(missing_ok=True)
        return out_path

class _VideoStore:

    def __init__(self, owner: 'SiMateUnifiedStorageReader'):
        self.owner = owner

    def cache_entry(self, key: tuple[str, str], entry: VideoCacheEntry) -> None:
        owner = self.owner
        owner._video_cache_active_bytes += int(entry.size or 0)
        owner._stats['video_cache_active_bytes'] = int(owner._video_cache_active_bytes)
        owner._video_cache_peak_bytes = max(owner._video_cache_peak_bytes, owner._video_cache_active_bytes)
        owner._stats['video_cache_peak_bytes'] = int(owner._video_cache_peak_bytes)
        evicted = owner._video_cache.put(key, entry)
        self.count_evictions(evicted)
        if owner.max_cached_video_bytes_per_worker > 0:
            byte_evicted = owner._video_cache.evict_until(lambda : owner._video_cache_active_bytes > owner.max_cached_video_bytes_per_worker)
            if byte_evicted:
                owner._stats['video_cache_byte_evictions'] += len(byte_evicted)
                self.count_evictions(byte_evicted)

    def count_evictions(self, entries: list[VideoCacheEntry]) -> None:
        stats = self.owner._stats
        for entry in entries:
            if entry.mode == 'offset_tmpfile':
                stats['offset_video_tmp_cache_evictions'] += 1
            elif entry.mode == 'offset_memfd_opencv':
                stats['offset_memfd_cache_evictions'] += 1

    def release_entry(self, entry: VideoCacheEntry) -> None:
        owner = self.owner
        size = int(getattr(entry, 'size', 0) or 0)
        try:
            owner._drop_video_reader_for_path(entry.path)
            if entry.fd is not None:
                entry.close_fd()
            if entry.owned_file:
                entry.path.unlink(missing_ok=True)
        finally:
            owner._video_cache_active_bytes = max(0, owner._video_cache_active_bytes - size)
            owner._stats['video_cache_active_bytes'] = int(owner._video_cache_active_bytes)
            if entry.mode == 'offset_memfd_opencv':
                owner._memfd_active_bytes = max(0, owner._memfd_active_bytes - size)
                owner._stats['offset_memfd_active_bytes'] = int(owner._memfd_active_bytes)
                owner._stats['offset_memfd_delete_count'] += 1
            elif entry.mode == 'offset_tmpfile':
                owner._video_tmp_active_bytes = max(0, owner._video_tmp_active_bytes - size)
                owner._stats['offset_video_tmp_active_bytes'] = int(owner._video_tmp_active_bytes)
                owner._stats['offset_video_tmp_delete_count'] += 1

    def ensure_tmp_space(self, next_bytes: int) -> None:
        owner = self.owner
        if owner.video_tmp_max_bytes_per_worker and owner._video_tmp_active_bytes + next_bytes > owner.video_tmp_max_bytes_per_worker:
            owner._stats['offset_video_tmp_space_errors'] += 1
            raise OSError(f'SiMate offset tmp video budget exceeded: active={owner._video_tmp_active_bytes}, next={next_bytes}, limit={owner.video_tmp_max_bytes_per_worker}')
        usage = shutil.disk_usage(owner.video_tmp_dir)
        if owner.video_tmp_required_free_bytes and usage.free - next_bytes < owner.video_tmp_required_free_bytes:
            owner._stats['offset_video_tmp_space_errors'] += 1
            raise OSError(f'Not enough free space in video_tmp_dir={owner.video_tmp_dir}: free={usage.free}, next={next_bytes}, required_free={owner.video_tmp_required_free_bytes}')

    def offset_tmp_video(self, shard_path: str, member_path: str) -> VideoCacheEntry:
        owner = self.owner
        key = (shard_path, member_path)
        cached = owner._video_cache.get(key)
        if isinstance(cached, VideoCacheEntry) and cached.mode == 'offset_tmpfile':
            if cached.alive:
                owner._stats['offset_video_tmp_cache_hits'] += 1
                return cached
            stale = owner._video_cache.pop(key)
            if isinstance(stale, VideoCacheEntry):
                self.release_entry(stale)
        owner._stats['offset_video_tmp_cache_misses'] += 1
        info = owner._member_info(shard_path, member_path)
        size = int(info['size'])
        self.ensure_tmp_space(size)
        owner.video_tmp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(member_path).suffix or '.mp4'
        (fd, tmp_name) = tempfile.mkstemp(suffix=suffix, prefix=f'simate_{os.getpid()}_{uuid.uuid4().hex}_', dir=str(owner.video_tmp_dir))
        os.close(fd)
        tmp_path = Path(tmp_name)
        t0 = time.perf_counter()
        try:
            data = owner._read_member_bytes(shard_path, member_path)
            tmp_path.write_bytes(data)
            if tmp_path.stat().st_size != size:
                raise OSError(f'Incomplete offset tmp video write for {member_path}')
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        elapsed = time.perf_counter() - t0
        owner._video_tmp_active_bytes += size
        owner._stats['offset_video_tmp_active_bytes'] = int(owner._video_tmp_active_bytes)
        owner._stats['offset_video_tmp_peak_bytes'] = max(int(owner._stats.get('offset_video_tmp_peak_bytes', 0)), int(owner._video_tmp_active_bytes))
        owner._stats['offset_video_tmp_create_count'] += 1
        owner._stats['offset_video_tmp_bytes'] += size
        owner._stats['offset_video_tmp_copy_sec_x1e6'] += int(elapsed * 1000000)
        entry = VideoCacheEntry(mode='offset_tmpfile', path=tmp_path, size=size, owned_file=True)
        self.cache_entry(key, entry)
        return entry

    def create_memfd(self, name: bytes=b'simate_video') -> int:
        memfd_create = getattr(os, 'memfd_create', None)
        if callable(memfd_create):
            return int(memfd_create(name.decode('ascii', errors='ignore') or 'simate_video', flags=0))
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        fd = int(libc.syscall(319, ctypes.c_char_p(name), ctypes.c_uint(0)))
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return fd

    def offset_memfd_video(self, shard_path: str, member_path: str) -> VideoCacheEntry:
        owner = self.owner
        key = (shard_path, member_path)
        cached = owner._video_cache.get(key)
        if isinstance(cached, VideoCacheEntry) and cached.mode == 'offset_memfd_opencv':
            if cached.alive:
                owner._stats['offset_memfd_cache_hits'] += 1
                return cached
            stale = owner._video_cache.pop(key)
            if isinstance(stale, VideoCacheEntry):
                self.release_entry(stale)
        owner._stats['offset_memfd_cache_misses'] += 1
        info = owner._member_info(shard_path, member_path)
        size = int(info['size'])
        t0 = time.perf_counter()
        fd = self.create_memfd()
        try:
            copied = owner._copy_member_to_fd(shard_path, member_path, fd)
            if copied != size:
                raise OSError(f'Incomplete offset memfd video copy for {member_path}: got {copied} of {size}')
            os.lseek(fd, 0, os.SEEK_SET)
            entry = VideoCacheEntry(mode='offset_memfd_opencv', path=Path(f'/proc/self/fd/{fd}'), size=size, fd=fd)
            fd = -1
        except Exception:
            if fd >= 0:
                try:
                    os.close(fd)
                except Exception:
                    pass
            raise
        elapsed = time.perf_counter() - t0
        owner._memfd_active_bytes += size
        owner._stats['offset_memfd_active_bytes'] = int(owner._memfd_active_bytes)
        owner._stats['offset_memfd_peak_bytes'] = max(int(owner._stats.get('offset_memfd_peak_bytes', 0)), int(owner._memfd_active_bytes))
        owner._stats['offset_memfd_create_count'] += 1
        owner._stats['offset_memfd_bytes'] += size
        owner._stats['offset_memfd_copy_sec_x1e6'] += int(elapsed * 1000000)
        self.cache_entry(key, entry)
        return entry

    def cleanup(self) -> None:
        self.owner._video_cache.clear()

    def materialized_video(self, shard_path: str, member_path: str) -> VideoCacheEntry:
        owner = self.owner
        key = (shard_path, member_path)
        cached = owner._video_cache.get(key)
        if isinstance(cached, VideoCacheEntry) and cached.mode == 'materialize':
            if cached.alive and cached.path.exists():
                owner._stats['video_cache_hits'] += 1
                return cached
            stale = owner._video_cache.pop(key)
            if isinstance(stale, VideoCacheEntry):
                self.release_entry(stale)
        digest = hashlib.sha1(f'{shard_path}:{member_path}'.encode('utf-8')).hexdigest()[:16]
        out_path = owner.video_cache_dir / f'{digest}_{Path(member_path).name}'
        if not out_path.exists():
            owner._stats['video_cache_misses'] += 1
            owner._handle_cache_miss('video', shard_path, member_path, out_path)
            owner._stats['video_materialize_count'] += 1
            shard = owner.data_root / shard_path
            tmp_path = out_path.with_suffix(f'{out_path.suffix}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
            try:
                owner._stats['tar_open_count'] += 1
                owner._stats['video_tar_open_count'] += 1
                with tarfile.open(shard) as tar:
                    info = tar.getmember(member_path)
                    extracted = tar.extractfile(info)
                    if extracted is None:
                        raise FileNotFoundError(f'{member_path} is missing from {shard}')
                    with tmp_path.open('wb') as f:
                        while True:
                            chunk = extracted.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                if tmp_path.stat().st_size != info.size:
                    raise OSError(f'Incomplete video cache write for {member_path}')
                owner._stats['video_materialize_bytes'] += int(info.size)
                try:
                    tmp_path.replace(out_path)
                except FileExistsError:
                    tmp_path.unlink(missing_ok=True)
            finally:
                tmp_path.unlink(missing_ok=True)
        size = out_path.stat().st_size if out_path.exists() else int(owner._member_info(shard_path, member_path).get('size') or 0)
        entry = VideoCacheEntry(mode='materialize', path=out_path, size=size, owned_file=False)
        self.cache_entry(key, entry)
        return entry

class _FrameDecoder:

    def __init__(self, owner: 'SiMateUnifiedStorageReader'):
        self.owner = owner

    def decode(self, shard_path: str, member_path: str, indices: list[int]) -> list[Any]:
        owner = self.owner
        t0 = time.perf_counter()
        try:
            if owner.read_mode == 'offset' and owner.video_mode == 'offset_memfd_opencv':
                entry = owner._video_store.offset_memfd_video(shard_path, member_path)
                path = entry.path
                allow_reader_cache = True
            elif owner.read_mode == 'offset' and owner.video_mode == 'offset_tmpfile':
                entry = owner._video_store.offset_tmp_video(shard_path, member_path)
                path = entry.path
                allow_reader_cache = False
            else:
                entry = owner._video_store.materialized_video(shard_path, member_path)
                path = entry.path
                allow_reader_cache = True
            if owner.video_decoder == 'decord':
                try:
                    if allow_reader_cache:
                        return self.decode_decord(path, indices)
                    return self.decode_decord_once(path, indices)
                except Exception:
                    owner._stats['decord_decode_failures'] += 1
                    return self.decode_opencv(path, member_path, indices, allow_reader_cache=allow_reader_cache)
            return self.decode_opencv(path, member_path, indices, allow_reader_cache=allow_reader_cache)
        finally:
            elapsed = time.perf_counter() - t0
            owner._stats['frame_decode_calls'] += 1
            owner._stats['frame_decode_frames'] += len(indices)
            owner._stats['frame_decode_sec_x1e6'] += int(elapsed * 1000000)

    def decode_decord(self, path: Path, indices: list[int]) -> list[Any]:
        owner = self.owner
        vr = owner._video_reader(path)
        frames = vr.get_batch([int(index) for index in indices]).asnumpy()
        owner._stats['decord_batch_decodes'] += 1
        return [frames[i].copy() for i in range(len(indices))]

    def decode_decord_once(self, path: Path, indices: list[int]) -> list[Any]:
        from decord import VideoReader, cpu
        owner = self.owner
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        frames = vr.get_batch([int(index) for index in indices]).asnumpy()
        owner._stats['decord_batch_decodes'] += 1
        return [frames[i].copy() for i in range(len(indices))]

    def decode_opencv(self, path: Path, member_path: str, indices: list[int], *, allow_reader_cache: bool=True) -> list[Any]:
        owner = self.owner
        (cap, cv2, should_release) = owner._opencv_capture(path, allow_reader_cache=allow_reader_cache)
        try:
            frames_by_index: dict[int, Any] = {}
            for run in self.contiguous_runs([int(index) for index in indices]):
                if len(run) > 1:
                    owner._stats['opencv_batch_decodes'] += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(run[0]))
                    for index in run:
                        (ok, frame) = cap.read()
                        if not ok or frame is None:
                            raise IndexError(f'Could not decode frame {index} from {member_path}')
                        frames_by_index[int(index)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    index = int(run[0])
                    owner._stats['opencv_single_decodes'] += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    (ok, frame) = cap.read()
                    if not ok or frame is None:
                        raise IndexError(f'Could not decode frame {index} from {member_path}')
                    frames_by_index[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return [frames_by_index[int(index)] for index in indices]
        finally:
            if should_release:
                cap.release()

    @staticmethod
    def contiguous_runs(indices: list[int]) -> list[list[int]]:
        if not indices:
            return []
        runs: list[list[int]] = [[int(indices[0])]]
        for index in indices[1:]:
            index = int(index)
            if index == runs[-1][-1] + 1:
                runs[-1].append(index)
            else:
                runs.append([index])
        return runs

class SiMateUnifiedStorageReader(BaseStorageReader):

    def __init__(self, data_root: str | Path, *, video_cache_dir: str | Path | None=None, table_cache_dir: str | Path | None=None, max_cached_tables: int=64, max_cached_videos: int=16, max_cached_frames: int=32, max_cached_frame_bytes: int=256 * 1024 * 1024, max_cached_video_readers: int=4, max_cached_state_rows: int=4096, max_cached_step_rows: int=4096, video_decoder: str='opencv', metrics_dir: str | Path | None=None, cache_miss_policy: str='error', read_mode: str='offset', video_mode: str='offset_memfd_opencv', video_tmp_dir: str | Path | None=None, shard_fd_cache_size: int=4, member_index_cache_size: int=16, video_tmp_max_bytes_per_worker: int=64 * 1024 * 1024 * 1024, video_tmp_required_free_bytes: int=8 * 1024 * 1024 * 1024, offset_memfd_chunk_bytes: int=32 * 1024 * 1024, max_cached_video_bytes_per_worker: int=0):
        super().__init__(data_root)
        self.read_mode = str(read_mode or 'offset').lower()
        if self.read_mode != 'offset':
            raise ValueError(f"SiPAI formal Simate training supports only read_mode='offset', got {read_mode!r}")
        self.video_mode = str(video_mode or 'offset_memfd_opencv').lower()
        if self.video_mode != 'offset_memfd_opencv':
            raise ValueError(f"SiPAI formal Simate training supports only video_mode='offset_memfd_opencv', got {video_mode!r}")
        self.video_cache_dir = Path(video_cache_dir or Path(tempfile.gettempdir()) / 'simate_unified_video_cache')
        if self.video_mode == 'materialize' or video_cache_dir is not None:
            self.video_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_base = video_tmp_dir or video_cache_dir or self.data_root / '.simate_tmp'
        self.video_tmp_dir = Path(tmp_base)
        if self.video_mode == 'offset_tmpfile':
            self.video_tmp_dir.mkdir(parents=True, exist_ok=True)
        self.table_cache_dir = Path(table_cache_dir) if table_cache_dir else None
        if self.table_cache_dir is not None and self.read_mode != 'offset':
            self.table_cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cached_tables = max(1, int(max_cached_tables))
        self.max_cached_videos = max(1, int(max_cached_videos))
        self.max_cached_frames = max(0, int(max_cached_frames))
        self.max_cached_frame_bytes = max(0, int(max_cached_frame_bytes))
        self.max_cached_video_readers = max(0, int(max_cached_video_readers))
        self.max_cached_state_rows = max(0, int(max_cached_state_rows))
        self.max_cached_step_rows = max(0, int(max_cached_step_rows))
        self.shard_fd_cache_size = max(0, int(shard_fd_cache_size))
        self.member_index_cache_size = max(1, int(member_index_cache_size))
        self.video_tmp_max_bytes_per_worker = max(0, int(video_tmp_max_bytes_per_worker))
        self.video_tmp_required_free_bytes = max(0, int(video_tmp_required_free_bytes))
        self.offset_memfd_chunk_bytes = max(1, int(offset_memfd_chunk_bytes))
        self.max_cached_video_bytes_per_worker = max(0, int(max_cached_video_bytes_per_worker))
        self._video_cache_active_bytes = 0
        self._video_cache_peak_bytes = 0
        self._video_tmp_active_bytes = 0
        self._memfd_active_bytes = 0
        self.video_decoder = str(video_decoder or 'opencv').lower()
        self.cache_miss_policy = str(cache_miss_policy or 'error').lower()
        if self.cache_miss_policy != 'error':
            raise ValueError(f"SiPAI formal Simate training supports only cache_miss_policy='error', got {cache_miss_policy!r}")
        self.metrics_dir = Path(metrics_dir) if metrics_dir else None
        if self.metrics_dir is not None:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            atexit.register(self._write_metrics_file)
        if self.video_decoder != 'opencv':
            raise ValueError(f"SiPAI formal Simate training supports only video_decoder='opencv', got {video_decoder!r}")
        self._table_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._member_index_cache: OrderedDict[str, Dict[str, Dict[str, Any]]] = OrderedDict()
        self._shard_fd_cache: OrderedDict[str, Any] = OrderedDict()
        self._offset_store = _OffsetShardStore(self)
        self._table_store = _TableStore(self)
        self._video_store = _VideoStore(self)
        self._frame_decoder = _FrameDecoder(self)
        atexit.register(self._close_shard_fds)
        self._video_cache = _LruCache(self.max_cached_videos, on_evict=self._release_video_cache_entry)
        atexit.register(self._cleanup_cached_videos)
        self._frame_cache: OrderedDict[tuple[str, str, int], Any] = OrderedDict()
        self._frame_cache_bytes = 0
        self._video_reader_cache: OrderedDict[Any, Any] = OrderedDict()
        self._state_cache: OrderedDict[tuple[str, str, int], Dict[str, Any]] = OrderedDict()
        self._step_cache: OrderedDict[tuple[str, str, int], Dict[str, Any]] = OrderedDict()
        self._stats: Dict[str, int] = {'table_cache_hits': 0, 'table_cache_misses': 0, 'video_cache_hits': 0, 'video_cache_misses': 0, 'frame_cache_hits': 0, 'frame_cache_misses': 0, 'frame_cache_evictions': 0, 'state_cache_hits': 0, 'state_cache_misses': 0, 'state_cache_evictions': 0, 'state_batch_reads': 0, 'state_batch_rows': 0, 'state_batch_unique_rows': 0, 'state_batch_deduped_rows': 0, 'step_cache_hits': 0, 'step_cache_misses': 0, 'step_cache_evictions': 0, 'step_batch_reads': 0, 'step_batch_rows': 0, 'video_reader_cache_hits': 0, 'video_reader_cache_misses': 0, 'video_reader_cache_evictions': 0, 'decord_batch_decodes': 0, 'decord_decode_failures': 0, 'opencv_batch_decodes': 0, 'opencv_single_decodes': 0, 'tar_open_count': 0, 'table_tar_open_count': 0, 'table_materialize_count': 0, 'table_materialize_bytes': 0, 'video_tar_open_count': 0, 'video_materialize_count': 0, 'video_materialize_bytes': 0, 'frame_decode_calls': 0, 'frame_decode_frames': 0, 'frame_decode_sec_x1e6': 0, 'cache_miss_errors': 0, 'cache_miss_warn_materialize': 0, 'member_index_cache_hits': 0, 'member_index_cache_misses': 0, 'member_index_cache_evictions': 0, 'member_index_load_count': 0, 'member_index_rows': 0, 'shard_fd_cache_hits': 0, 'shard_fd_cache_misses': 0, 'shard_fd_open_count': 0, 'shard_fd_evictions': 0, 'offset_member_reads': 0, 'offset_member_read_bytes': 0, 'offset_member_read_chunks': 0, 'offset_member_read_sec_x1e6': 0, 'offset_table_reads': 0, 'offset_video_tmp_create_count': 0, 'offset_video_tmp_bytes': 0, 'offset_video_tmp_active_bytes': 0, 'offset_video_tmp_peak_bytes': 0, 'offset_video_tmp_delete_count': 0, 'offset_video_tmp_copy_sec_x1e6': 0, 'offset_video_tmp_space_errors': 0, 'offset_video_tmp_cache_hits': 0, 'offset_video_tmp_cache_misses': 0, 'offset_video_tmp_cache_evictions': 0, 'offset_memfd_create_count': 0, 'offset_memfd_bytes': 0, 'offset_memfd_active_bytes': 0, 'offset_memfd_peak_bytes': 0, 'offset_memfd_delete_count': 0, 'offset_memfd_copy_sec_x1e6': 0, 'offset_memfd_cache_hits': 0, 'offset_memfd_cache_misses': 0, 'offset_memfd_cache_evictions': 0, 'video_cache_active_bytes': 0, 'video_cache_peak_bytes': 0, 'video_cache_byte_evictions': 0}

    def read(self, episode: Mapping[str, Any], stream_key: str, *, index: Optional[int]=None, index_range: Optional[Iterable[int]]=None, timestamp: Optional[float]=None, time_range: Optional[Iterable[float]]=None, interpolate: bool=False) -> Any:
        stream = self._stream(episode, stream_key)
        stream_type = str(stream.get('type') or '')
        if stream_type == 'camera' or stream_key.startswith('camera.'):
            return self._read_camera(episode, stream_key, stream, index=index, index_range=index_range)
        return self._read_state(episode, stream_key, stream, index=index)

    def read_many(self, episode: Mapping[str, Any], stream_key: str, *, indices: Iterable[int]) -> List[Any]:
        stream = self._stream(episode, stream_key)
        stream_type = str(stream.get('type') or '')
        int_indices = [int(index) for index in indices]
        if stream_type == 'camera' or stream_key.startswith('camera.'):
            return self._read_camera_many(episode, stream_key, stream, indices=int_indices)
        return self._read_state_many(episode, stream_key, stream, indices=int_indices)

    def cache_stats(self) -> Dict[str, Any]:
        stats = dict(self._stats)
        decode_sec = float(stats.get('frame_decode_sec_x1e6', 0)) / 1000000.0
        stats['frame_decode_sec'] = decode_sec
        stats['frame_decode_fps'] = float(stats.get('frame_decode_frames', 0)) / decode_sec if decode_sec > 0 else None
        stats['pid'] = os.getpid()
        return {**stats, 'table_cache_entries': len(self._table_cache), 'video_cache_entries': len(self._video_cache), 'frame_cache_entries': len(self._frame_cache), 'frame_cache_bytes': self._frame_cache_bytes, 'video_reader_cache_entries': len(self._video_reader_cache), 'state_cache_entries': len(self._state_cache), 'step_cache_entries': len(self._step_cache), 'max_cached_state_rows': self.max_cached_state_rows, 'max_cached_step_rows': self.max_cached_step_rows, 'max_cached_frames': self.max_cached_frames, 'max_cached_frame_bytes': self.max_cached_frame_bytes, 'video_cache_dir': str(self.video_cache_dir), 'table_cache_dir': str(self.table_cache_dir) if self.table_cache_dir is not None else None, 'cache_miss_policy': self.cache_miss_policy, 'read_mode': self.read_mode, 'video_mode': self.video_mode, 'video_tmp_dir': str(self.video_tmp_dir), 'member_index_cache_entries': len(self._member_index_cache), 'shard_fd_cache_entries': len(self._shard_fd_cache), 'shard_fd_cache_size': self.shard_fd_cache_size, 'member_index_cache_size': self.member_index_cache_size, 'video_tmp_max_bytes_per_worker': self.video_tmp_max_bytes_per_worker, 'video_tmp_required_free_bytes': self.video_tmp_required_free_bytes, 'offset_memfd_chunk_bytes': self.offset_memfd_chunk_bytes, 'max_cached_video_bytes_per_worker': self.max_cached_video_bytes_per_worker}

    def step_table_value(self, episode: Mapping[str, Any], column: str, step_index: int) -> Any:
        storage = episode.get('storage', {})
        extra = storage.get('extra', {}) if isinstance(storage, Mapping) else {}
        shard_path = str(extra.get('shard_path') or '')
        step_table_path = str(extra.get('step_table') or episode.get('step_index', {}).get('step_table', {}).get('path') or '')
        if not shard_path or not step_table_path:
            raise ValueError('SiMate episode is missing storage.extra.shard_path or step_table')
        row = self._step_table_row(shard_path, step_table_path, int(step_index))
        return row.get(column)

    def step_table_values(self, episode: Mapping[str, Any], column: str, step_indices: Iterable[int]) -> List[Any]:
        import pyarrow as pa
        indices = [int(step_index) for step_index in step_indices]
        if not indices:
            return []
        storage = episode.get('storage', {})
        extra = storage.get('extra', {}) if isinstance(storage, Mapping) else {}
        shard_path = str(extra.get('shard_path') or '')
        step_table_path = str(extra.get('step_table') or episode.get('step_index', {}).get('step_table', {}).get('path') or '')
        if not shard_path or not step_table_path:
            raise ValueError('SiMate episode is missing storage.extra.shard_path or step_table')
        table = self._table(shard_path, step_table_path)
        clipped = [min(max(0, index), table.num_rows - 1) for index in indices]
        self._stats['step_batch_reads'] += 1
        self._stats['step_batch_rows'] += len(clipped)
        if column not in table.column_names:
            return [None] * len(clipped)
        return table[column].take(pa.array(clipped)).to_pylist()

    def stream_length(self, episode: Mapping[str, Any], stream_key: str) -> int | None:
        stream = self._stream(episode, stream_key)
        stream_type = str(stream.get('type') or '')
        if stream_type == 'camera' or stream_key.startswith('camera.'):
            index_path = str(stream.get('index_path') or '')
            if not index_path:
                return None
            shard_path = str(stream.get('shard_path') or episode.get('storage', {}).get('extra', {}).get('shard_path'))
            return int(self._table(shard_path, index_path).num_rows)
        shard_path = str(stream.get('shard_path') or episode.get('storage', {}).get('extra', {}).get('shard_path'))
        return int(self._table(shard_path, str(stream.get('path'))).num_rows)

    def _step_table_row(self, shard_path: str, step_table_path: str, step_index: int) -> Dict[str, Any]:
        table = self._table(shard_path, step_table_path)
        idx = min(max(0, int(step_index)), table.num_rows - 1)
        cache_key = (shard_path, step_table_path, idx)
        cached = self._step_cache_get(cache_key)
        if cached is not None:
            self._stats['step_cache_hits'] += 1
            return cached
        self._stats['step_cache_misses'] += 1
        row = {name: table[name][idx].as_py() for name in table.column_names}
        self._step_cache_put(cache_key, row)
        return row

    def _step_cache_get(self, key: tuple[str, str, int]) -> Dict[str, Any] | None:
        if self.max_cached_step_rows <= 0:
            return None
        if key not in self._step_cache:
            return None
        self._step_cache.move_to_end(key)
        return self._step_cache[key]

    def _step_cache_put(self, key: tuple[str, str, int], row: Dict[str, Any]) -> None:
        if self.max_cached_step_rows <= 0:
            return
        self._step_cache[key] = row
        self._step_cache.move_to_end(key)
        while len(self._step_cache) > self.max_cached_step_rows:
            self._step_cache.popitem(last=False)
            self._stats['step_cache_evictions'] += 1

    def _read_camera(self, episode: Mapping[str, Any], stream_key: str, stream: Mapping[str, Any], *, index: Optional[int], index_range: Optional[Iterable[int]]) -> Any:
        shard_path = str(stream.get('shard_path') or episode.get('storage', {}).get('extra', {}).get('shard_path'))
        video_path = str(stream.get('path'))
        index_path = str(stream.get('index_path'))
        table = self._table(shard_path, index_path)
        if index_range is not None:
            (start, end) = _to_int_range(index_range) or (0, 0)
            indices = [min(max(0, int(idx)), table.num_rows - 1) for idx in range(start, end)]
            arrays = self._decode_video_frames(shard_path, video_path, indices)
            return [self._camera_payload_from_array(table, idx, array) for (idx, array) in zip(indices, arrays)]
        if index is None:
            raise ValueError('SiMate camera reads require index or index_range')
        idx = min(max(0, int(index)), table.num_rows - 1)
        array = self._decode_video_frame(shard_path, video_path, idx)
        return self._camera_payload_from_array(table, idx, array)

    def _read_camera_many(self, episode: Mapping[str, Any], stream_key: str, stream: Mapping[str, Any], *, indices: List[int]) -> List[Dict[str, Any]]:
        del stream_key
        if not indices:
            return []
        shard_path = str(stream.get('shard_path') or episode.get('storage', {}).get('extra', {}).get('shard_path'))
        video_path = str(stream.get('path'))
        index_path = str(stream.get('index_path'))
        table = self._table(shard_path, index_path)
        clipped = [min(max(0, int(index)), table.num_rows - 1) for index in indices]
        arrays = self._decode_video_frames(shard_path, video_path, clipped)
        return [self._camera_payload_from_array(table, idx, array) for (idx, array) in zip(clipped, arrays)]

    def _camera_payload_from_array(self, index_table: Any, index: int, array: Any) -> Dict[str, Any]:
        row = {name: index_table[name][index].as_py() for name in index_table.column_names}
        return {'type': 'camera', 'index': index, 'timestamp': row.get('timestamp'), 'pts': row.get('pts'), 'keyframe': row.get('keyframe'), 'valid': row.get('valid'), 'array': array}

    def _read_state(self, episode: Mapping[str, Any], stream_key: str, stream: Mapping[str, Any], *, index: Optional[int]) -> Dict[str, Any]:
        if index is None:
            raise ValueError('SiMate state reads require index')
        return self._read_state_many(episode, stream_key, stream, indices=[int(index)])[0]

    def _read_state_many(self, episode: Mapping[str, Any], stream_key: str, stream: Mapping[str, Any], *, indices: List[int]) -> List[Dict[str, Any]]:
        if not indices:
            return []
        shard_path = str(stream.get('shard_path') or episode.get('storage', {}).get('extra', {}).get('shard_path'))
        member_path = str(stream.get('path'))
        table = self._table(shard_path, member_path)
        clipped = [min(max(0, int(index)), table.num_rows - 1) for index in indices]
        out: List[Dict[str, Any] | None] = [None] * len(clipped)
        missing_indices: List[int] = []
        missing_positions: List[int] = []
        for (pos, idx) in enumerate(clipped):
            cache_key = (shard_path, member_path, idx)
            cached = self._state_cache_get(cache_key)
            if cached is not None:
                self._stats['state_cache_hits'] += 1
                out[pos] = self._copy_state_payload(cached)
            else:
                self._stats['state_cache_misses'] += 1
                missing_indices.append(idx)
                missing_positions.append(pos)
        if missing_indices:
            unique_indices = list(dict.fromkeys(missing_indices))
            rows = self._take_table_rows(table, unique_indices)
            rows_by_index = dict(zip(unique_indices, rows))
            self._stats['state_batch_reads'] += 1
            self._stats['state_batch_rows'] += len(missing_indices)
            self._stats['state_batch_unique_rows'] += len(unique_indices)
            self._stats['state_batch_deduped_rows'] += len(missing_indices) - len(unique_indices)
            payloads_by_index: Dict[int, Dict[str, Any]] = {}
            for idx in unique_indices:
                payload = self._state_payload_from_row(rows_by_index[idx], idx)
                self._state_cache_put((shard_path, member_path, idx), payload)
                payloads_by_index[idx] = payload
            for (pos, idx) in zip(missing_positions, missing_indices):
                out[pos] = self._copy_state_payload(payloads_by_index[idx])
        if any((item is None for item in out)):
            raise RuntimeError(f'Failed to read all state rows for {stream_key}')
        return list(out)

    def _take_table_rows(self, table: Any, indices: List[int]) -> List[Dict[str, Any]]:
        import pyarrow as pa
        taken = table.take(pa.array(indices))
        return taken.to_pylist()

    def _state_payload_from_row(self, row: Mapping[str, Any], index: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {'type': row.get('component') or 'state', 'index': int(index), 'timestamp': row.get('timestamp'), 'valid': row.get('valid'), 'message': dict(row)}
        if all((row.get(key) is not None for key in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'))):
            pose = [float(row[key]) for key in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
            out.update({'position': pose[:3], 'rotation_xyzw': pose[3:7], 'pose': pose})
        if row.get('width') is not None:
            out['width'] = float(row['width'])
        if row.get('values') is not None:
            values = row['values']
            out['values'] = values.tolist() if hasattr(values, 'tolist') else values
        return out

    def _state_cache_get(self, key: tuple[str, str, int]) -> Dict[str, Any] | None:
        if self.max_cached_state_rows <= 0:
            return None
        if key not in self._state_cache:
            return None
        self._state_cache.move_to_end(key)
        return self._state_cache[key]

    def _state_cache_put(self, key: tuple[str, str, int], payload: Dict[str, Any]) -> None:
        if self.max_cached_state_rows <= 0:
            return
        self._state_cache[key] = payload
        self._state_cache.move_to_end(key)
        while len(self._state_cache) > self.max_cached_state_rows:
            self._state_cache.popitem(last=False)
            self._stats['state_cache_evictions'] += 1

    def _copy_state_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        if isinstance(out.get('message'), Mapping):
            out['message'] = dict(out['message'])
        if isinstance(out.get('position'), list):
            out['position'] = list(out['position'])
        if isinstance(out.get('rotation_xyzw'), list):
            out['rotation_xyzw'] = list(out['rotation_xyzw'])
        if isinstance(out.get('pose'), list):
            out['pose'] = list(out['pose'])
        if isinstance(out.get('values'), list):
            out['values'] = [list(item) if isinstance(item, list) else item for item in out['values']]
        return out

    def _member_index_path(self, shard_path: str) -> Path:
        return self._offset_store.member_index_path(shard_path)

    def _member_index(self, shard_path: str) -> Dict[str, Dict[str, Any]]:
        return self._offset_store.member_index(shard_path)

    def _member_info(self, shard_path: str, member_path: str) -> Dict[str, Any]:
        return self._offset_store.member_info(shard_path, member_path)

    def _shard_fd(self, shard_path: str) -> Any:
        return self._offset_store.shard_fd(shard_path)

    def _close_shard_fds(self) -> None:
        self._offset_store.close()

    def _read_member_bytes(self, shard_path: str, member_path: str) -> bytes:
        return self._offset_store.read_member_bytes(shard_path, member_path)

    def _table(self, shard_path: str, member_path: str) -> Any:
        return self._table_store.table(shard_path, member_path)

    def _materialized_table(self, shard_path: str, member_path: str) -> Path:
        return self._table_store.materialized_table(shard_path, member_path)

    def _ensure_video_tmp_space(self, next_bytes: int) -> None:
        if self.video_tmp_max_bytes_per_worker and self._video_tmp_active_bytes + next_bytes > self.video_tmp_max_bytes_per_worker:
            self._stats['offset_video_tmp_space_errors'] += 1
            raise OSError(f'SiMate offset tmp video budget exceeded: active={self._video_tmp_active_bytes}, next={next_bytes}, limit={self.video_tmp_max_bytes_per_worker}')
        usage = shutil.disk_usage(self.video_tmp_dir)
        if self.video_tmp_required_free_bytes and usage.free - next_bytes < self.video_tmp_required_free_bytes:
            self._stats['offset_video_tmp_space_errors'] += 1
            raise OSError(f'Not enough free space in video_tmp_dir={self.video_tmp_dir}: free={usage.free}, next={next_bytes}, required_free={self.video_tmp_required_free_bytes}')

    def _cache_video_entry(self, key: tuple[str, str], entry: VideoCacheEntry) -> None:
        self._video_store.cache_entry(key, entry)

    def _count_video_evictions(self, entries: list[VideoCacheEntry]) -> None:
        self._video_store.count_evictions(entries)

    def _release_video_cache_entry(self, entry: VideoCacheEntry) -> None:
        self._video_store.release_entry(entry)

    def _ensure_video_tmp_space(self, next_bytes: int) -> None:
        self._video_store.ensure_tmp_space(next_bytes)

    def _offset_tmp_video(self, shard_path: str, member_path: str) -> VideoCacheEntry:
        return self._video_store.offset_tmp_video(shard_path, member_path)

    def _create_memfd(self, name: bytes=b'simate_video') -> int:
        return self._video_store.create_memfd(name)

    def _write_all_to_fd(self, fd: int, data: bytes) -> None:
        view = memoryview(data)
        total = 0
        while total < len(view):
            written = os.write(fd, view[total:])
            if written <= 0:
                raise OSError(f'Short fd write after {total} of {len(view)} bytes')
            total += written

    def _copy_member_to_fd(self, shard_path: str, member_path: str, out_fd: int, *, chunk_bytes: int | None=None) -> int:
        return self._offset_store.copy_member_to_fd(shard_path, member_path, out_fd, chunk_bytes=chunk_bytes)

    def _offset_memfd_video(self, shard_path: str, member_path: str) -> VideoCacheEntry:
        return self._video_store.offset_memfd_video(shard_path, member_path)

    def _cleanup_cached_videos(self) -> None:
        self._video_store.cleanup()

    def _materialized_video(self, shard_path: str, member_path: str) -> VideoCacheEntry:
        return self._video_store.materialized_video(shard_path, member_path)

    def _handle_cache_miss(self, kind: str, shard_path: str, member_path: str, local_path: Path) -> None:
        message = f'SiMate {kind} cache miss for {shard_path}:{member_path} -> {local_path}'
        if self.cache_miss_policy in {'error', 'skip'}:
            self._stats['cache_miss_errors'] += 1
            raise CacheMissError(message)
        if self.cache_miss_policy == 'warn_materialize':
            self._stats['cache_miss_warn_materialize'] += 1
            logger.warning('%s; materializing because cache_miss_policy=warn_materialize', message)

    def _decode_video_frame(self, shard_path: str, member_path: str, index: int) -> Any:
        return self._decode_video_frames(shard_path, member_path, [int(index)])[0]

    def _decode_video_frames(self, shard_path: str, member_path: str, indices: list[int]) -> list[Any]:
        if not indices:
            return []
        out: list[Any | None] = [None] * len(indices)
        missing_positions_by_index: OrderedDict[int, list[int]] = OrderedDict()
        for (pos, index) in enumerate(indices):
            index = int(index)
            key = (shard_path, member_path, index)
            cached = self._frame_cache_get(key)
            if cached is None:
                self._stats['frame_cache_misses'] += 1
                missing_positions_by_index.setdefault(index, []).append(pos)
            else:
                self._stats['frame_cache_hits'] += 1
                out[pos] = cached
        if missing_positions_by_index:
            decode_indices = sorted(missing_positions_by_index)
            decoded = self._decode_video_frames_uncached(shard_path, member_path, decode_indices)
            if len(decoded) != len(decode_indices):
                raise RuntimeError(f'Decoded {len(decoded)} frames for {len(decode_indices)} requested frames from {member_path}')
            for (index, array) in zip(decode_indices, decoded):
                self._frame_cache_put((shard_path, member_path, int(index)), array)
                for pos in missing_positions_by_index[index]:
                    out[pos] = array
        if any((array is None for array in out)):
            raise RuntimeError(f'Failed to fill requested frames from {member_path}')
        return list(out)

    def _decode_video_frames_uncached(self, shard_path: str, member_path: str, indices: list[int]) -> list[Any]:
        return self._frame_decoder.decode(shard_path, member_path, indices)

    def _write_metrics_file(self) -> None:
        if self.metrics_dir is None:
            return
        try:
            path = self.metrics_dir / f'simate_reader_metrics_{os.getpid()}.json'
            path.write_text(json.dumps(self.cache_stats(), ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    def _decode_video_frames_decord(self, path: Path, indices: list[int]) -> list[Any]:
        return self._frame_decoder.decode_decord(path, indices)

    def _decode_video_frames_decord_once(self, path: Path, indices: list[int]) -> list[Any]:
        return self._frame_decoder.decode_decord_once(path, indices)

    def _video_reader(self, path: Path) -> Any:
        cache_key = ('decord', path)
        if self.max_cached_video_readers > 0 and cache_key in self._video_reader_cache:
            self._stats['video_reader_cache_hits'] += 1
            self._video_reader_cache.move_to_end(cache_key)
            return self._video_reader_cache[cache_key]
        self._stats['video_reader_cache_misses'] += 1
        from decord import VideoReader, cpu
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        self._cache_video_reader(cache_key, reader)
        return reader

    def _opencv_capture(self, path: Path, *, allow_reader_cache: bool=True) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError('SiMate MP4 camera reads require decord or opencv-python/cv2') from exc
        cache_key = ('opencv', path)
        if allow_reader_cache and self.max_cached_video_readers > 0 and (cache_key in self._video_reader_cache):
            self._stats['video_reader_cache_hits'] += 1
            self._video_reader_cache.move_to_end(cache_key)
            return (self._video_reader_cache[cache_key], cv2, False)
        self._stats['video_reader_cache_misses'] += 1
        cap = cv2.VideoCapture(str(path))
        if allow_reader_cache and self.max_cached_video_readers > 0:
            self._cache_video_reader(cache_key, cap)
            return (cap, cv2, False)
        return (cap, cv2, True)

    def _drop_video_reader_for_path(self, path: Path) -> None:
        cache_key = ('opencv', path)
        reader = self._video_reader_cache.pop(cache_key, None)
        if reader is None:
            return
        close = getattr(reader, 'release', None) or getattr(reader, 'close', None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _cache_video_reader(self, cache_key: Any, reader: Any) -> None:
        if self.max_cached_video_readers <= 0:
            return
        self._video_reader_cache[cache_key] = reader
        self._video_reader_cache.move_to_end(cache_key)
        while len(self._video_reader_cache) > self.max_cached_video_readers:
            (_, evicted) = self._video_reader_cache.popitem(last=False)
            close = getattr(evicted, 'release', None) or getattr(evicted, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._stats['video_reader_cache_evictions'] += 1

    def _decode_video_frames_opencv(self, path: Path, member_path: str, indices: list[int], *, allow_reader_cache: bool=True) -> list[Any]:
        return self._frame_decoder.decode_opencv(path, member_path, indices, allow_reader_cache=allow_reader_cache)

    def _contiguous_runs(self, indices: list[int]) -> list[list[int]]:
        return self._frame_decoder.contiguous_runs(indices)

    def _frame_cache_get(self, key: tuple[str, str, int]) -> Any | None:
        if self.max_cached_frames <= 0 or self.max_cached_frame_bytes <= 0:
            return None
        if key not in self._frame_cache:
            return None
        self._frame_cache.move_to_end(key)
        return self._frame_cache[key]

    def _frame_cache_put(self, key: tuple[str, str, int], array: Any) -> None:
        if self.max_cached_frames <= 0 or self.max_cached_frame_bytes <= 0:
            return
        size = int(getattr(array, 'nbytes', 0) or 0)
        if size > self.max_cached_frame_bytes:
            return
        old = self._frame_cache.pop(key, None)
        if old is not None:
            self._frame_cache_bytes -= int(getattr(old, 'nbytes', 0) or 0)
        self._frame_cache[key] = array
        self._frame_cache.move_to_end(key)
        self._frame_cache_bytes += size
        while len(self._frame_cache) > self.max_cached_frames or self._frame_cache_bytes > self.max_cached_frame_bytes:
            (_, evicted) = self._frame_cache.popitem(last=False)
            self._frame_cache_bytes -= int(getattr(evicted, 'nbytes', 0) or 0)
            self._stats['frame_cache_evictions'] += 1

def _to_int_range(value: Optional[Iterable[int]]) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    (start, end) = list(value)
    return (int(start), int(end))

def _to_float_range(value: Optional[Iterable[float]]) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    (start, end) = list(value)
    return (float(start), float(end))
