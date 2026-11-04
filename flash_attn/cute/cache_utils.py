# Manage Ahead-of-Time (AOT) compiled kernels
import atexit
import contextlib
import fcntl
import hashlib
import logging
import os
import pickle
import sys
import tempfile
import time
from functools import lru_cache
from getpass import getuser
from pathlib import Path
from typing import Hashable, TypeAlias

import ctypes

import cutlass
import cutlass.cute as cute
import tvm_ffi
from cutlass.cutlass_dsl import JitCompiledFunction
from flash_attn.cute.fa_logging import fa_log

# Pre-load cute DSL runtime libraries with RTLD_GLOBAL so that their symbols
# (e.g. _cudaLibraryLoadData) are visible to .so modules loaded later via dlopen.
# Upstream cute.runtime.load_module loads these without RTLD_GLOBAL, which causes
# "undefined symbol" errors when loading cached kernels from disk.
for _lib_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
    if Path(_lib_path).exists():
        ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

CompileKeyType: TypeAlias = tuple[Hashable, ...]
CallableFunction: TypeAlias = JitCompiledFunction | tvm_ffi.Function

# Enable cache via `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`
CUTE_DSL_CACHE_ENABLED: bool = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0") == "1"


# Customize cache dir via `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`, default is
# `/tmp/${USER}/flash_attention_cute_dsl_cache``
CUTE_DSL_CACHE_DIR: str | None = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", None)


# Share compiled kernels across machines with Redis as an L2 behind the disk cache via
# `FLASH_ATTENTION_CUTE_DSL_CACHE_REDIS_URL=redis://...` (needs the cache enabled)
CUTE_DSL_CACHE_REDIS_URL: str | None = os.getenv("FLASH_ATTENTION_CUTE_DSL_CACHE_REDIS_URL", None)


def get_cache_path() -> Path:
    if CUTE_DSL_CACHE_DIR is not None:
        cache_dir = Path(CUTE_DSL_CACHE_DIR)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "flash_attention_cute_dsl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """
    Hash all CuTe Python sources plus runtime ABI stamps into a short fingerprint.

    The fingerprint changes whenever:
    - Any .py file under flash_attn/cute is added, removed, renamed, or modified.
    - The Python minor version changes (e.g. 3.13 -> 3.14).
    - The cutlass or tvm_ffi package version changes.

    Computed once per process and cached.
    """
    cute_root = Path(__file__).resolve().parent
    h = hashlib.sha256()

    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())

    for src in sorted(cute_root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(cute_root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)

    return h.hexdigest()


class FileLock:
    """Context manager for advisory file locks using fcntl.flock.

    Supports exclusive (write) and shared (read) locks.
    Always blocks with polling until the lock is acquired or timeout is reached.

    Usage:
        with FileLock(lock_path, exclusive=True, timeout=15, label="abc"):
            # do work under lock
    """

    def __init__(
        self,
        lock_path: Path,
        exclusive: bool,
        timeout: float = 15,
        label: str = "",
    ):
        """
        Args:
            lock_path: Path to the lock file on disk.
            exclusive: True for exclusive (write) lock, False for shared (read) lock.
            timeout: Max seconds to wait for lock acquisition before raising RuntimeError.
            label: Optional human-readable label for error messages.
        """
        self.lock_path: Path = lock_path
        self.exclusive: bool = exclusive
        self.timeout: float = timeout
        self.label: str = label
        self._fd: int = -1

    @property
    def _lock_label(self) -> str:
        kind = "exclusive" if self.exclusive else "shared"
        return f"{kind} {self.label}" if self.label else kind

    def __enter__(self) -> "FileLock":
        open_flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH

        self._fd = os.open(str(self.lock_path), open_flags)

        deadline = time.monotonic() + self.timeout
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.1)
        if not acquired:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(
                f"Timed out after {self.timeout}s waiting for "
                f"{self._lock_label} lock: {self.lock_path}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class JITCache:
    """
    In-memory cache for compiled functions.
    """

    def __init__(self):
        self.cache: dict[CompileKeyType, CallableFunction] = {}

    def __setitem__(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        self.cache[key] = fn

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        return self.cache[key]

    def __contains__(self, key: CompileKeyType) -> bool:
        return key in self.cache

    def clear(self) -> None:
        """
        Clear in-memory cache of compiled functions
        """
        self.cache.clear()


class JITPersistentCache(JITCache):
    """
    In-memory cache for compiled functions, which is also backed by persistent storage.
    Use cutedsl ahead-of-time (AOT) compilation, only supporting enable_tvm_ffi=True
    """

    EXPORT_FUNCTION_PREFIX = "func"
    LOCK_TIMEOUT_SECONDS = 15

    def __init__(self, cache_path: Path):
        super().__init__()
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path: Path = cache_path
        # hashes of on-disk entries that failed to load: the export replaces them
        self._unloadable: set[str] = set()

    def __setitem__(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        JITCache.__setitem__(self, key, fn)
        self._try_export_to_storage(key, fn)

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        # Use __contains__ to try populating in-memory cache with persistent storage
        self.__contains__(key)
        return JITCache.__getitem__(self, key)

    def __contains__(self, key: CompileKeyType) -> bool:
        # Checks in-memory cache first, then tries loading from storage.
        # When returning True, guarantees the in-memory cache is populated.
        if JITCache.__contains__(self, key):
            return True
        return self._try_load_from_storage(key)

    def _try_load_from_storage(self, key: CompileKeyType) -> bool:
        """
        Try to load a function from persistent storage into in-memory cache.
        Returns True if loaded successfully, False if not found on disk.
        Holds a shared lock during loading to prevent concurrent writes.
        """
        sha256_hex = self._key_to_hash(key)
        obj_path = self.cache_path / f"{sha256_hex}.o"
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=False,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            if obj_path.exists():
                fa_log(1, f"Loading compiled function from disk: {obj_path}")
                try:
                    m = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
                    fn = getattr(m, self.EXPORT_FUNCTION_PREFIX)
                except Exception as e:
                    # A bad entry (e.g. truncated by a killed or out-of-space export) is a miss:
                    # recompile, and the export replaces it (see _try_export_to_storage).
                    fa_log(
                        1, f"Unloadable cache entry {obj_path} ({type(e).__name__}); recompiling"
                    )
                    self._unloadable.add(sha256_hex)
                    return False
                JITCache.__setitem__(self, key, fn)
                return True
            else:
                fa_log(1, f"Cache miss on disk for key hash {sha256_hex}")
        return False

    def _try_export_to_storage(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        """Export a compiled function to persistent storage under exclusive lock."""
        sha256_hex = self._key_to_hash(key)
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=True,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            obj_path = self.cache_path / f"{sha256_hex}.o"
            # A nonempty entry is normally another process's export. One that this process failed
            # to load is replaced instead: its size does not make it valid. (If another process
            # replaced it meanwhile, overwriting a good entry with an equivalent one is harmless.)
            if (
                sha256_hex not in self._unloadable
                and obj_path.exists()
                and obj_path.stat().st_size > 0
            ):
                fa_log(1, f"Skipping export, already on disk: {obj_path}")
                return
            fa_log(1, f"Exporting compiled function to disk: {obj_path}")
            # Write to a temporary name and rename into place, so an interrupted or failed
            # write (a killed process, a full disk) never leaves a partial entry behind.
            tmp_path = self.cache_path / f".{sha256_hex}.{os.getpid()}.tmp.o"
            try:
                fn.export_to_c(
                    object_file_path=str(tmp_path),
                    function_name=self.EXPORT_FUNCTION_PREFIX,
                )
                if tmp_path.stat().st_size == 0:
                    raise OSError(f"empty object file {tmp_path}")
                os.replace(tmp_path, obj_path)
                self._unloadable.discard(sha256_hex)
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                fa_log(1, f"Export to disk failed for {obj_path} ({type(e).__name__}: {e})")
                return
            fa_log(1, f"Successfully exported compiled function to disk: {obj_path}")

    def _key_to_hash(self, key: CompileKeyType) -> str:
        return hashlib.sha256(pickle.dumps(key)).hexdigest()

    def _lock_path(self, sha256_hex: str) -> Path:
        return self.cache_path / f"{sha256_hex}.lock"

    def clear(self) -> None:
        """
        Not only clear the in-memory cache. Also purge persistent compilation cache.
        """
        fa_log(1, f"Clearing persistent cache at {self.cache_path}")
        super().clear()
        for child in self.cache_path.iterdir():
            child.unlink()


class _StderrUnlessConfigured(logging.Handler):
    """Prints to stderr only when no other non-Null handler the record propagates to exists: the
    NullHandler fa_logging puts on `flash_attn` otherwise hides it (it disables lastResort)."""

    def emit(self, record: logging.LogRecord) -> None:
        logger = logging.getLogger(record.name)
        while logger is not None:
            if any(
                h is not self and not isinstance(h, logging.NullHandler) for h in logger.handlers
            ):
                return
            logger = logger.parent if logger.propagate else None
        try:  # sys.stderr looked up per record, like lastResort, so a later redirect is followed
            sys.stderr.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


_redis_log = logging.getLogger("flash_attn.redis_cache")


class RedisL2:
    """Redis client for RedisJITPersistentCache, shared by every cache in the process.

    `redis` is imported on first use. The first failure (missing package, connect, timeout,
    auth, unreadable compile target) logs one warning and turns Redis off for the rest of the
    process, so the cache falls back to compiling and a Redis outage never fails the caller.
    """

    # Entries expire so stale fingerprints and any bad entry age out without an operator
    TTL_SECONDS = 14 * 24 * 3600

    def __init__(self, url: str):
        self._url = url
        self._client = None
        self._disabled = False
        self._fetch_failure_warned = False
        self.hits = self.misses = self.uploads = 0

    def get(self, namespace: str, sha256_hex: str) -> bytes | None:
        blob = self._call("get", namespace, sha256_hex)
        if not self._disabled:  # a failed or skipped GET is neither a hit nor a miss
            if blob is None:
                self.misses += 1
            else:
                self.hits += 1
        return blob

    def put(self, namespace: str, sha256_hex: str, value: bytes, replace: bool) -> None:
        if self._call("set", namespace, sha256_hex, value, nx=not replace, ex=self.TTL_SECONDS):
            self.uploads += 1

    def log_summary(self) -> None:
        _redis_log.info(
            "FA4 Redis compile cache: %d hits, %d misses, %d uploads%s",
            self.hits,
            self.misses,
            self.uploads,
            " (disabled after an error)" if self._disabled else "",
        )

    def report_fetch_failure(self, e: Exception) -> None:
        """A fetched entry that cannot reach the disk cache is compiled locally: warn once."""
        message = f"Writing a Redis entry to the disk cache failed ({type(e).__name__}: {e})"
        if self._fetch_failure_warned:
            fa_log(1, message)
            return
        self._fetch_failure_warned = True
        _redis_log.warning("FA4 Redis compile cache: %s; compiling locally", message)

    def _call(self, method: str, namespace: str, sha256_hex: str, *args, **kwargs):
        if self._disabled:
            return None
        try:
            key = _redis_key(namespace, sha256_hex)
            if self._client is None:
                import redis

                # Bounded by default; timeouts in the URL's query string take precedence
                self._client = redis.Redis.from_url(
                    self._url, socket_connect_timeout=2, socket_timeout=3
                )
            return getattr(self._client, method)(key, *args, **kwargs)
        except Exception as e:
            self._disabled = True
            # Only the exception type: the URL, which may surface in messages, carries the password
            _redis_log.warning(
                "FA4 Redis compile cache disabled for this process (%s on %s); compiling locally",
                type(e).__name__,
                method,
            )
            return None


def _start_redis_l2(url: str) -> RedisL2:
    # At INFO so the on/summary lines show whatever the FA level, with or without app logging
    _redis_log.setLevel(logging.INFO)
    _redis_log.addHandler(_StderrUnlessConfigured())
    redis_l2 = RedisL2(url)
    _redis_log.info("FA4 Redis compile cache on (L2 behind the disk cache)")
    atexit.register(redis_l2.log_summary)
    return redis_l2


_REDIS_L2: RedisL2 | None = (
    _start_redis_l2(CUTE_DSL_CACHE_REDIS_URL)
    if CUTE_DSL_CACHE_ENABLED and CUTE_DSL_CACHE_REDIS_URL
    else None
)


@lru_cache(maxsize=1)
def _compile_target() -> str:
    """GPU arch and CUDA toolkit the objects are built for. Part of the Redis key, which machines
    share; resolved on first use, when the process already runs on its GPU."""
    from cutlass.base_dsl import detect_gpu_arch

    arch = os.getenv("CUTE_DSL_ARCH") or detect_gpu_arch("")
    return f"{arch}-cu{cutlass.CUDA_VERSION.major}.{cutlass.CUDA_VERSION.minor}"


def _redis_key(namespace: str, sha256_hex: str) -> str:
    return f"{namespace}:{_compile_target()}:{sha256_hex}"


class RedisJITPersistentCache(JITPersistentCache):
    """
    JITPersistentCache with Redis as L2: a disk miss fetches the exported object from Redis
    into the disk cache, then loads it as a disk hit; each new export is uploaded.
    Keys are namespaced like the disk path (source fingerprint + cache name) plus the compile
    target, so machines with another GPU arch or CUDA toolkit never share an entry.
    """

    def __init__(self, cache_path: Path, redis_l2: RedisL2, key_prefix: str):
        super().__init__(cache_path)
        self.redis_l2: RedisL2 = redis_l2
        self.key_prefix: str = key_prefix

    def _try_load_from_storage(self, key: CompileKeyType) -> bool:
        if super()._try_load_from_storage(key):
            return True
        if not self._fetch_from_redis(self._key_to_hash(key)):
            return False
        return super()._try_load_from_storage(key)

    def _try_export_to_storage(self, key: CompileKeyType, fn: JitCompiledFunction) -> None:
        sha256_hex = self._key_to_hash(key)
        # An entry this process could not load may have come from Redis: replace it there too
        replace = sha256_hex in self._unloadable
        super()._try_export_to_storage(key, fn)
        try:
            if not self._has_loadable_candidate(sha256_hex):
                return  # the export failed: nothing on disk, or still the entry that failed to load
            blob = (self.cache_path / f"{sha256_hex}.o").read_bytes()
            self.redis_l2.put(self.key_prefix, sha256_hex, blob, replace=replace)
        except OSError as e:  # the disk entry vanished: the function is already cached in memory
            fa_log(1, f"Reading the exported entry for upload failed ({type(e).__name__}: {e})")

    def _fetch_from_redis(self, sha256_hex: str) -> bool:
        """Copy an entry from Redis into the disk cache; any reply, lock or disk error is a miss.
        The GET runs outside the entry's lock: a slow Redis must not hold the lock that upstream's
        load and export wait on (their lock timeouts raise)."""
        obj_path = self.cache_path / f"{sha256_hex}.o"
        tmp_path = self.cache_path / f".{sha256_hex}.{os.getpid()}.tmp.o"
        try:
            blob = self.redis_l2.get(self.key_prefix, sha256_hex)
            if blob is None:
                return False
            with FileLock(
                self._lock_path(sha256_hex),
                exclusive=True,
                timeout=self.LOCK_TIMEOUT_SECONDS,
                label=sha256_hex,
            ):
                if not self._has_loadable_candidate(sha256_hex):  # else another process wrote it
                    tmp_path.write_bytes(blob)
                    os.replace(tmp_path, obj_path)
        except Exception as e:
            with contextlib.suppress(OSError):  # cleanup must not raise either
                tmp_path.unlink(missing_ok=True)
            self.redis_l2.report_fetch_failure(e)
            return False
        fa_log(1, f"Fetched compiled function from Redis for key hash {sha256_hex}")
        return True

    def _has_loadable_candidate(self, sha256_hex: str) -> bool:
        """A nonempty disk entry this process has not failed to load: upstream's skip condition
        in _try_export_to_storage (keep the two in sync on rebase)."""
        obj_path = self.cache_path / f"{sha256_hex}.o"
        return (
            sha256_hex not in self._unloadable and obj_path.exists() and obj_path.stat().st_size > 0
        )


def get_jit_cache(name: str | None = None) -> JITCache:
    """
    JIT cache factory.
    `name` is an optional identifier to create subdirectories to manage cache.

    When persistent caching is enabled, artifacts are namespaced under a
    source fingerprint directory so that code or dependency changes
    automatically invalidate stale entries.
    """
    if CUTE_DSL_CACHE_ENABLED:
        path = get_cache_path() / _compute_source_fingerprint()
        if name:
            path = path / name
        fa_log(1, f"Creating persistent JIT cache at {path}")
        if _REDIS_L2 is not None:
            key_prefix = f"fa4:{_compute_source_fingerprint()}:{name or ''}"
            return RedisJITPersistentCache(path, _REDIS_L2, key_prefix)
        return JITPersistentCache(path)
    else:
        fa_log(1, "Persistent cache disabled, using in-memory JIT cache")
        return JITCache()
