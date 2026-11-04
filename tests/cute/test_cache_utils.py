import fcntl
import hashlib
import logging
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

import flash_attn.cute.cache_utils as cache_utils
from flash_attn.cute import fa_logging


def test_persistent_cache_hit_logs_at_host_level_only(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="flash_attn")
    original_level = fa_logging.get_fa_log_level()
    key = ("test-key",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    obj_path.write_bytes(b"cache-hit")
    monkeypatch.setattr(
        cache_utils.cute.runtime,
        "load_module",
        lambda *_args, **_kwargs: SimpleNamespace(func=object()),
    )
    try:
        monkeypatch.setattr(fa_logging, "_fa_log_level", 0)
        assert cache_utils.JITPersistentCache(tmp_path)._try_load_from_storage(key)
        assert "Loading compiled function from disk" not in caplog.text

        caplog.clear()
        monkeypatch.setattr(fa_logging, "_fa_log_level", 1)
        assert cache_utils.JITPersistentCache(tmp_path)._try_load_from_storage(key)
        assert "Loading compiled function from disk" in caplog.text
    finally:
        monkeypatch.setattr(fa_logging, "_fa_log_level", original_level)


def test_persistent_cache_unloadable_entry_is_a_miss(tmp_path, monkeypatch):
    """A bad entry (e.g. a zero-byte object left by an export that hit a full disk) must read
    as a miss, so the caller recompiles, instead of failing every later run."""
    key = ("bad-entry",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    (tmp_path / f"{cache._key_to_hash(key)}.o").write_bytes(b"")

    def failing_load(*_args, **_kwargs):
        raise RuntimeError("Failed to lookup function '__tvm_ffi_func'")

    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", failing_load)
    assert not cache._try_load_from_storage(key)


def test_persistent_cache_export_is_atomic(tmp_path):
    """A failed export leaves no entry behind; a successful one lands at the final path."""
    cache = cache_utils.JITPersistentCache(tmp_path)

    class Failing:
        def export_to_c(self, object_file_path, function_name):
            open(object_file_path, "wb").close()  # created, then the write fails
            raise OSError(28, "No space left on device")

    class Good:
        def export_to_c(self, object_file_path, function_name):
            with open(object_file_path, "wb") as f:
                f.write(b"object")

    cache._try_export_to_storage(("k1",), Failing())
    assert not list(tmp_path.glob("*.o")), "a failed export left a file behind"
    cache._try_export_to_storage(("k2",), Good())
    assert (tmp_path / f"{cache._key_to_hash(('k2',))}.o").read_bytes() == b"object"
    assert not list(tmp_path.glob(".*.tmp.o"))


def test_persistent_cache_replaces_nonempty_unloadable_entry(tmp_path, monkeypatch):
    """A truncated but nonempty entry fails to load; the recompiled function's export must
    replace it (size alone does not make an entry valid), so the next process loads it
    instead of recompiling forever."""
    key = ("truncated-entry",)
    cache = cache_utils.JITPersistentCache(tmp_path)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    obj_path.write_bytes(b"trunc")

    def load_module(path, **_kwargs):
        if open(path, "rb").read() != b"object":
            raise RuntimeError("Failed to lookup function '__tvm_ffi_func'")
        return SimpleNamespace(func=object())

    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", load_module)

    class Good:
        def export_to_c(self, object_file_path, function_name):
            with open(object_file_path, "wb") as f:
                f.write(b"object")

    assert not cache._try_load_from_storage(key)
    cache[key] = Good()  # recompiled: stored in memory and exported
    assert obj_path.read_bytes() == b"object"
    assert not list(tmp_path.glob(".*.tmp.o"))
    # a fresh process loads the replaced entry
    assert cache_utils.JITPersistentCache(tmp_path)._try_load_from_storage(key)

    # a nonempty entry this process loaded (or never tried) is still left alone
    other = ("other",)
    other_path = tmp_path / f"{cache._key_to_hash(other)}.o"
    other_path.write_bytes(b"from another process")
    cache[other] = Good()
    assert other_path.read_bytes() == b"from another process"


class DictRedis:
    """In-process stand-in for a redis.Redis client."""

    def __init__(self):
        self.data = {}
        self.ttl = {}
        self.client_options = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.data:
            return None
        self.data[key] = value
        self.ttl[key] = ex
        return True


SECRET_URL = "redis://:s3cret@redis:6379"


class DownRedis(DictRedis):
    """Fails like redis-py does when the server is unreachable, on GET or only on SET."""

    def __init__(self, fail_get=True):
        super().__init__()
        self.fail_get = fail_get

    def get(self, key):
        if self.fail_get:
            raise ConnectionError(f"Error 111 connecting to {SECRET_URL}. Connection refused.")
        return super().get(key)

    def set(self, key, value, nx=False, ex=None):
        raise ConnectionError(f"Error 111 connecting to {SECRET_URL}. Connection refused.")


class Good:
    def export_to_c(self, object_file_path, function_name):
        with open(object_file_path, "wb") as f:
            f.write(b"object")


class Failing:
    def export_to_c(self, object_file_path, function_name):
        raise OSError(28, "No space left on device")


def _stub_redis_module(monkeypatch, client):
    def from_url(_url, **options):
        client.client_options = options
        return client

    monkeypatch.setitem(
        sys.modules, "redis", SimpleNamespace(Redis=SimpleNamespace(from_url=from_url))
    )


def _redis_cache(tmp_path, monkeypatch, client):
    _stub_redis_module(monkeypatch, client)
    return cache_utils.RedisJITPersistentCache(
        tmp_path, cache_utils.RedisL2(SECRET_URL), "fa4:fp:fwd"
    )


def _key_of(cache, key):
    return cache_utils._redis_key(cache.key_prefix, cache._key_to_hash(key))


@pytest.fixture
def load_module_of_good_objects(monkeypatch):
    loaded = SimpleNamespace(func=object())

    def load_module(path, **_kwargs):
        if open(path, "rb").read() != b"object":
            raise RuntimeError("Failed to lookup function '__tvm_ffi_func'")
        return loaded

    monkeypatch.setattr(cache_utils.cute.runtime, "load_module", load_module)
    return loaded.func


def test_redis_export_is_shared_with_an_empty_disk_cache(
    tmp_path, monkeypatch, load_module_of_good_objects
):
    """A function compiled on one machine is uploaded after its export; another machine's disk
    miss fetches it into its own disk cache and loads it instead of compiling."""
    key = ("shared",)
    redis = DictRedis()
    _redis_cache(tmp_path / "node0", monkeypatch, redis)[key] = Good()

    other = _redis_cache(tmp_path / "node1", monkeypatch, redis)
    assert other[key] is load_module_of_good_objects
    assert (tmp_path / "node1" / f"{other._key_to_hash(key)}.o").read_bytes() == b"object"
    assert not list((tmp_path / "node1").glob(".*.tmp.o"))
    # entries age out of the Redis shared with other caches, and a stalled server cannot hang
    assert all(0 < ttl < float("inf") for ttl in redis.ttl.values())
    assert redis.client_options["socket_connect_timeout"] > 0
    assert redis.client_options["socket_timeout"] > 0


def test_redis_heals_an_unloadable_disk_entry(tmp_path, monkeypatch, load_module_of_good_objects):
    """A truncated local entry is replaced from Redis instead of recompiled."""
    key = ("healed",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    obj_path.write_bytes(b"trunc")
    redis.data[_key_of(cache, key)] = b"object"
    assert key in cache
    assert obj_path.read_bytes() == b"object"


def test_redis_upload_keeps_the_first_entry(tmp_path, monkeypatch):
    """Concurrent cold starts all export; the first upload wins and later ones leave it."""
    key = ("raced",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    redis.data[_key_of(cache, key)] = b"first"
    cache[key] = Good()
    assert redis.data == {_key_of(cache, key): b"first"}


def test_unloadable_redis_entry_is_replaced(tmp_path, monkeypatch, load_module_of_good_objects):
    """A bad Redis entry would make every machine recompile forever: the recompiled export
    replaces it."""
    key = ("bad-remote",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path / "node0", monkeypatch, redis)
    redis.data[_key_of(cache, key)] = b"trunc"
    assert key not in cache
    cache[key] = Good()
    assert list(redis.data.values()) == [b"object"]
    assert key in _redis_cache(tmp_path / "node1", monkeypatch, redis)


def test_failed_export_never_uploads_the_unloadable_disk_entry(
    tmp_path, monkeypatch, load_module_of_good_objects
):
    """A truncated disk entry whose recompiled export fails (full disk) must not be uploaded:
    it would overwrite a good entry another machine put in Redis meanwhile."""
    key = ("disk-full",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    (tmp_path / f"{cache._key_to_hash(key)}.o").write_bytes(b"trunc")
    assert key not in cache
    redis.data[_key_of(cache, key)] = b"object"  # another machine's upload
    fn = Failing()
    cache[key] = fn
    assert cache[key] is fn
    assert redis.data == {_key_of(cache, key): b"object"}


def test_fetch_uses_an_entry_another_process_wrote_while_it_waited_for_the_lock(
    tmp_path, monkeypatch, load_module_of_good_objects
):
    """The processes of one machine fetch an entry once: whoever waited on the lock uses the
    entry already on disk instead of fetching (here, stale bytes) over it."""
    key = ("herd",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    obj_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    redis.data[_key_of(cache, key)] = b"stale"

    class OtherProcessWritesFirst(cache_utils.FileLock):
        def __enter__(self):
            if self.exclusive:
                obj_path.write_bytes(b"object")
            return super().__enter__()

    monkeypatch.setattr(cache_utils, "FileLock", OtherProcessWritesFirst)
    assert key in cache
    assert obj_path.read_bytes() == b"object"


def test_unusable_redis_reply_is_a_miss(tmp_path, monkeypatch, load_module_of_good_objects):
    """A reply that cannot be written to disk (a client built with decode_responses returns
    str) is a miss: the caller compiles instead of crashing."""
    key = ("decoded",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    redis.data[_key_of(cache, key)] = "object"
    assert key not in cache
    cache[key] = Good()
    assert key in cache_utils.JITPersistentCache(tmp_path)


def test_failed_disk_write_of_a_fetched_entry_is_a_miss_and_leaves_no_temp_file(
    tmp_path, monkeypatch, load_module_of_good_objects, caplog
):
    key = ("disk-error",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    redis.data[_key_of(cache, key)] = b"object"

    def full_disk(*_args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cache_utils.os, "replace", full_disk)
    with caplog.at_level(logging.WARNING, logger="flash_attn"):
        assert key not in cache
        redis.data[_key_of(cache, ("disk-error-2",))] = b"object"
        assert ("disk-error-2",) not in cache
    assert not list(tmp_path.glob(".*.tmp.o"))
    # the first failure warns, later ones stay quiet; Redis itself stays on
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "No space left on device" in caplog.text
    assert cache.redis_l2.hits == 2


def test_fetch_waiting_out_the_lock_timeout_is_a_miss(
    tmp_path, monkeypatch, load_module_of_good_objects, caplog
):
    """A process stuck holding the entry's lock makes the fetch time out: a warned miss that
    compiles locally, never an exception from the lock."""
    key = ("lock-held",)
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    redis.data[_key_of(cache, key)] = b"object"
    monkeypatch.setattr(cache, "LOCK_TIMEOUT_SECONDS", 0.2)

    class LockTakenAfterUpstreamLoad(cache_utils.FileLock):
        def __enter__(self):
            if self.exclusive:  # the fetch's lock; upstream's disk load takes a shared one
                holder = os.open(self.lock_path, os.O_RDWR | os.O_CREAT)
                fcntl.flock(holder, fcntl.LOCK_EX)
                held.append(holder)
            return super().__enter__()

    held = []
    monkeypatch.setattr(cache_utils, "FileLock", LockTakenAfterUpstreamLoad)
    try:
        with caplog.at_level(logging.WARNING, logger="flash_attn"):
            assert key not in cache
    finally:
        for fd in held:
            os.close(fd)
    assert held
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    assert not (tmp_path / f"{cache._key_to_hash(key)}.o").exists()


def test_unknown_compile_target_disables_sharing_not_compiling(
    tmp_path, monkeypatch, load_module_of_good_objects, caplog
):
    """If the compile target cannot be read (a cutlass API rename), Redis turns off with the
    same single warning as an outage: nothing is shared and nothing raises."""
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)

    def unknown_target():
        raise ImportError("cannot import name 'detect_gpu_arch'")

    monkeypatch.setattr(cache_utils, "_compile_target", unknown_target)
    with caplog.at_level(logging.WARNING, logger="flash_attn"):
        for key in [("no-target",), ("no-target-2",)]:
            assert key not in cache
            fn = Good()
            cache[key] = fn
            assert cache[key] is fn
    assert redis.data == {}
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_entry_vanishing_after_export_never_fails_the_compile(tmp_path, monkeypatch):
    """The function is compiled and cached in memory; checking the disk entry before upload is
    best effort (another process may clear the cache between exists() and stat())."""
    cache = _redis_cache(tmp_path, monkeypatch, DictRedis())

    def vanished(_sha256_hex):
        raise FileNotFoundError("entry removed between exists() and stat()")

    monkeypatch.setattr(cache, "_has_loadable_candidate", vanished)
    key = ("vanished",)
    fn = Good()
    cache[key] = fn
    assert cache[key] is fn


def test_entry_unreadable_after_export_never_fails_the_compile(tmp_path, monkeypatch):
    """The entry can also vanish between the check and the read for upload: nothing is uploaded
    and the compiled function is still returned."""
    redis = DictRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    read_bytes = cache_utils.Path.read_bytes

    def vanished_before_read(path):
        if path.parent == tmp_path and path.suffix == ".o":
            raise FileNotFoundError(f"{path} removed before the upload read it")
        return read_bytes(path)

    monkeypatch.setattr(cache_utils.Path, "read_bytes", vanished_before_read)
    key = ("unreadable",)
    fn = Good()
    cache[key] = fn
    assert cache[key] is fn
    assert redis.data == {}


def test_redis_get_runs_with_the_entry_lock_free(
    tmp_path, monkeypatch, load_module_of_good_objects
):
    """A slow GET must not hold the entry's lock: upstream's load and export wait on it and
    raise when their lock times out."""
    key = ("lock-free-get",)

    class LockCheckingRedis(DictRedis):
        def get(self, redis_key):
            fd = os.open(cache._lock_path(cache._key_to_hash(key)), os.O_RDWR | os.O_CREAT)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if anyone holds it
            finally:
                os.close(fd)
            return super().get(redis_key)

    redis = LockCheckingRedis()
    cache = _redis_cache(tmp_path, monkeypatch, redis)
    redis.data[_key_of(cache, key)] = b"object"
    assert key in cache


def test_summary_counts_hits_misses_and_uploads(
    tmp_path, monkeypatch, load_module_of_good_objects, caplog
):
    redis = DictRedis()
    _stub_redis_module(monkeypatch, redis)
    redis_l2 = cache_utils.RedisL2(SECRET_URL)
    key = ("counted",)
    node0 = cache_utils.RedisJITPersistentCache(tmp_path / "node0", redis_l2, "fa4:fp:fwd")
    assert key not in node0
    node0[key] = Good()
    node1 = cache_utils.RedisJITPersistentCache(tmp_path / "node1", redis_l2, "fa4:fp:fwd")
    assert key in node1
    node1[key] = Good()  # a second upload of the same entry is rejected by NX, not counted
    with caplog.at_level(logging.INFO, logger="flash_attn"):
        redis_l2.log_summary()
    assert "1 hits, 1 misses, 1 uploads" in caplog.text


@pytest.mark.parametrize("fail_get", [True, False])
def test_summary_never_counts_failed_calls(
    tmp_path, monkeypatch, load_module_of_good_objects, caplog, fail_get
):
    """A Redis outage must not read as a cold cache: failed and skipped calls count as nothing."""
    _stub_redis_module(monkeypatch, DownRedis(fail_get=fail_get))
    redis_l2 = cache_utils.RedisL2(SECRET_URL)
    cache = cache_utils.RedisJITPersistentCache(tmp_path, redis_l2, "fa4:fp:fwd")
    for key in [("down",), ("down-2",)]:
        assert key not in cache
        cache[key] = Good()
    with caplog.at_level(logging.INFO, logger="flash_attn"):
        redis_l2.log_summary()
    misses = 0 if fail_get else 1  # a GET answered before the SET failed is a real miss
    expected = f"0 hits, {misses} misses, 0 uploads (disabled after an error)"
    assert expected in caplog.text


def test_unreachable_redis_server_falls_back_to_compiling(
    tmp_path, monkeypatch, load_module_of_good_objects, caplog
):
    """The real client against a refused port: redis-py's own exception types must be caught."""
    pytest.importorskip("redis")
    cache = cache_utils.RedisJITPersistentCache(
        tmp_path, cache_utils.RedisL2("redis://:s3cret@127.0.0.1:1/0"), "fa4:fp:fwd"
    )
    key = ("refused",)
    with caplog.at_level(logging.WARNING, logger="flash_attn"):
        assert key not in cache
        fn = Good()
        cache[key] = fn
        assert cache[key] is fn
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    assert "s3cret" not in caplog.text


REDIS_LOGGER = "logging.getLogger('flash_attn.redis_cache')"


@pytest.mark.parametrize(
    "fa_log_level, app_logging",
    [
        ("0", ""),
        ("0", "logging.basicConfig(level=logging.INFO); "),
        ("1", ""),
        ("0", f"{REDIS_LOGGER}.addHandler(logging.StreamHandler()); "),
        ("0", f"logging.basicConfig(level=logging.INFO); {REDIS_LOGGER}.propagate = False; "),
    ],
    ids=["no-logging-config", "app-logging", "fa-log-level-1", "direct-handler", "no-propagate"],
)
def test_environment_turns_on_the_redis_cache_at_import(tmp_path, fa_log_level, app_logging):
    """The variables a launcher exports reach get_jit_cache through a fresh import, which reports
    the L2 once at startup, warns once when Redis fails and summarizes once at exit: visible in an
    application that never configures logging (smol-gen), and never doubled or lost whatever
    handlers the application or FA_LOG_LEVEL add."""
    env = dict(
        os.environ,
        FA_LOG_LEVEL=fa_log_level,
        FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="1",
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(tmp_path),
        FLASH_ATTENTION_CUTE_DSL_CACHE_REDIS_URL="redis://:s3cret@127.0.0.1:1/0",
    )
    probe = (
        "import logging; " + app_logging + "import flash_attn.cute.cache_utils as c; "
        "cache = c.get_jit_cache('fwd'); print(type(cache).__name__); "
        "cache.redis_l2.get('fa4:fp:fwd', '0' * 64)"  # the URL's port refuses: Redis turns off
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "RedisJITPersistentCache" in result.stdout.splitlines()
    # one line each: the L2 is on, it was disabled, the summary at exit; none shows the URL
    output = result.stdout + result.stderr
    summary = "FA4 Redis compile cache: 0 hits, 0 misses, 0 uploads (disabled after an error)"
    for line in ["FA4 Redis compile cache on", "disabled for this process", summary]:
        assert output.count(line) == 1, line
        if fa_log_level == "0":
            assert line in result.stderr, line
    assert "s3cret" not in output


@pytest.mark.parametrize("failure", ["no-redis-package", "get-fails", "set-fails"])
def test_redis_failure_falls_back_to_compiling_with_one_warning(
    tmp_path, monkeypatch, load_module_of_good_objects, caplog, failure
):
    """Every cache in the process shares one client: the first failure warns once, never
    leaks the URL, and every cache keeps compiling and exporting to disk."""
    _stub_redis_module(monkeypatch, DownRedis(fail_get=failure == "get-fails"))
    if failure == "no-redis-package":
        monkeypatch.setitem(sys.modules, "redis", None)  # import redis raises ImportError
    redis_l2 = cache_utils.RedisL2(SECRET_URL)
    caches = [
        cache_utils.RedisJITPersistentCache(tmp_path / name, redis_l2, f"fa4:fp:{name}")
        for name in ("fwd", "bwd")
    ]
    key = ("offline",)
    with caplog.at_level(logging.WARNING, logger="flash_attn"):
        for cache in caches:
            assert key not in cache
            fn = Good()
            cache[key] = fn
            assert cache[key] is fn
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "s3cret" not in caplog.text
    assert all(key in cache_utils.JITPersistentCache(tmp_path / n) for n in ("fwd", "bwd"))


@pytest.mark.parametrize(
    "enabled, url, shared",
    [(True, SECRET_URL, True), (True, None, False), (False, SECRET_URL, False)],
)
def test_get_jit_cache_shares_through_redis_only_with_cache_enabled_and_url(
    tmp_path, monkeypatch, enabled, url, shared
):
    """Redis is used only when both settings are on, with keys namespaced by FA4 source
    fingerprint, cache name and compile target so other versions, kernels and GPUs never
    collide."""
    redis = DictRedis()
    _stub_redis_module(monkeypatch, redis)
    monkeypatch.setattr(cache_utils, "CUTE_DSL_CACHE_ENABLED", enabled)
    monkeypatch.setattr(cache_utils, "CUTE_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_utils, "_REDIS_L2", cache_utils.RedisL2(url) if url else None)
    key = ("same-compile-key",)
    for name in ("fwd", "bwd"):
        cache_utils.get_jit_cache(name)[key] = Good()
    fingerprint = cache_utils._compute_source_fingerprint()
    target = cache_utils._compile_target()
    sha = hashlib.sha256(pickle.dumps(key)).hexdigest()
    expected = (
        {f"fa4:{fingerprint}:{n}:{target}:{sha}" for n in ("fwd", "bwd")} if shared else set()
    )
    assert set(redis.data) == expected
