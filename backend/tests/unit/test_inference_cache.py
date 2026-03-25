"""Tests for the inference result cache.

Covers key generation (normalisation, model scoping, extra kwargs),
LRU eviction, TTL expiry, invalidation, clear, stats, and thread safety.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.ml.utils.inference_cache import (
    InferenceCache,
    _normalise,
    make_cache_key,
)


# ── Normalisation ────────────────────────────────────────────────────────


class TestNormalise:
    """Input text normalisation for consistent hashing."""

    def test_lowercases(self) -> None:
        assert _normalise("Patient Has HTN") == "patient has htn"

    def test_collapses_whitespace(self) -> None:
        assert _normalise("chest   pain\ton exertion") == "chest pain on exertion"

    def test_strips_surrounding_whitespace(self) -> None:
        assert _normalise("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert _normalise("") == ""

    def test_newlines_collapsed(self) -> None:
        assert _normalise("line1\n\nline2") == "line1 line2"


# ── Key generation ───────────────────────────────────────────────────────


class TestMakeCacheKey:
    """Deterministic SHA-256 key generation."""

    def test_same_input_same_key(self) -> None:
        k1 = make_cache_key("ner", "chest pain")
        k2 = make_cache_key("ner", "chest pain")
        assert k1 == k2

    def test_different_model_different_key(self) -> None:
        k1 = make_cache_key("ner", "chest pain")
        k2 = make_cache_key("icd", "chest pain")
        assert k1 != k2

    def test_whitespace_normalised(self) -> None:
        k1 = make_cache_key("ner", "chest  pain")
        k2 = make_cache_key("ner", "chest pain")
        assert k1 == k2

    def test_case_normalised(self) -> None:
        k1 = make_cache_key("ner", "Chest Pain")
        k2 = make_cache_key("ner", "chest pain")
        assert k1 == k2

    def test_kwargs_affect_key(self) -> None:
        k1 = make_cache_key("icd", "chest pain", top_k=5)
        k2 = make_cache_key("icd", "chest pain", top_k=10)
        assert k1 != k2

    def test_kwargs_order_independent(self) -> None:
        k1 = make_cache_key("icd", "chest pain", top_k=5, detail="brief")
        k2 = make_cache_key("icd", "chest pain", detail="brief", top_k=5)
        assert k1 == k2

    def test_no_kwargs_differs_from_empty_kwargs(self) -> None:
        k1 = make_cache_key("ner", "text")
        k2 = make_cache_key("ner", "text", top_k=5)
        assert k1 != k2

    def test_key_is_hex_sha256(self) -> None:
        key = make_cache_key("ner", "hello")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ── Cache get / put ──────────────────────────────────────────────────────


class TestCacheGetPut:
    """Basic cache operations."""

    def test_miss_returns_none(self) -> None:
        cache = InferenceCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_put_then_get(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("k1", {"entities": ["pain"]})
        assert cache.get("k1") == {"entities": ["pain"]}

    def test_overwrite_existing_key(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("k1", "old")
        cache.put("k1", "new")
        assert cache.get("k1") == "new"

    def test_stores_various_types(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("list", [1, 2, 3])
        cache.put("dict", {"a": 1})
        cache.put("str", "hello")
        cache.put("none", None)
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}
        assert cache.get("str") == "hello"
        # None is a valid cached value — but get() also returns None on miss,
        # so we rely on stats to confirm the hit.
        cache.get("none")
        assert cache.stats()["hits"] >= 4


# ── LRU eviction ─────────────────────────────────────────────────────────


class TestLRUEviction:
    """Least-recently-used eviction when max_size is reached."""

    def test_evicts_lru_on_overflow(self) -> None:
        cache = InferenceCache(max_size=2, default_ttl=3600)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_refreshes_lru_order(self) -> None:
        cache = InferenceCache(max_size=2, default_ttl=3600)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # Touch "a" — now "b" is LRU.
        cache.put("c", 3)  # Should evict "b".

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_overwrite_refreshes_lru_order(self) -> None:
        cache = InferenceCache(max_size=2, default_ttl=3600)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 10)  # Overwrite refreshes "a".
        cache.put("c", 3)  # Should evict "b".

        assert cache.get("a") == 10
        assert cache.get("b") is None


# ── TTL expiry ───────────────────────────────────────────────────────────


class TestTTLExpiry:
    """Entries expire after their TTL."""

    def test_expired_entry_returns_none(self) -> None:
        cache = InferenceCache(max_size=10, default_ttl=0.1)
        cache.put("k", "value")
        time.sleep(0.15)
        assert cache.get("k") is None

    def test_non_expired_entry_returned(self) -> None:
        cache = InferenceCache(max_size=10, default_ttl=10)
        cache.put("k", "value")
        assert cache.get("k") == "value"

    def test_custom_ttl_override(self) -> None:
        cache = InferenceCache(max_size=10, default_ttl=10)
        cache.put("fast", "gone soon", ttl=0.1)
        time.sleep(0.15)
        assert cache.get("fast") is None

    def test_expired_entry_evicted_from_store(self) -> None:
        cache = InferenceCache(max_size=10, default_ttl=0.1)
        cache.put("k", "value")
        time.sleep(0.15)
        cache.get("k")  # Triggers eviction.
        assert cache.stats()["size"] == 0


# ── Invalidation ─────────────────────────────────────────────────────────


class TestInvalidation:
    """Explicit key removal."""

    def test_invalidate_existing(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("k", "value")
        assert cache.invalidate("k") is True
        assert cache.get("k") is None

    def test_invalidate_nonexistent(self) -> None:
        cache = InferenceCache(max_size=10)
        assert cache.invalidate("nope") is False


# ── Clear ────────────────────────────────────────────────────────────────


class TestClear:
    """Bulk removal of all entries."""

    def test_clear_returns_count(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.clear() == 2

    def test_clear_empties_cache(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("a", 1)
        cache.clear()
        assert cache.get("a") is None
        assert cache.stats()["size"] == 0

    def test_clear_resets_stats(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("a", 1)
        cache.get("a")
        cache.get("miss")
        cache.clear()
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# ── Stats ────────────────────────────────────────────────────────────────


class TestStats:
    """Cache performance statistics."""

    def test_initial_stats(self) -> None:
        cache = InferenceCache(max_size=100, default_ttl=60)
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 100
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["default_ttl"] == 60

    def test_hit_rate_calculation(self) -> None:
        cache = InferenceCache(max_size=10)
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        cache.get("a")  # hit
        cache.get("c")  # miss
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(0.5)


# ── Static make_key method ──────────────────────────────────────────────


class TestStaticMakeKey:
    """InferenceCache.make_key delegates to module-level make_cache_key."""

    def test_delegates_correctly(self) -> None:
        k1 = InferenceCache.make_key("ner", "hello")
        k2 = make_cache_key("ner", "hello")
        assert k1 == k2


# ── Thread safety ────────────────────────────────────────────────────────


class TestThreadSafety:
    """Concurrent reads and writes don't corrupt the cache."""

    def test_concurrent_put_get(self) -> None:
        cache = InferenceCache(max_size=100, default_ttl=10)
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def writer(prefix: str) -> None:
            barrier.wait()
            for i in range(50):
                cache.put(f"{prefix}_{i}", i)

        def reader(prefix: str) -> None:
            barrier.wait()
            for i in range(50):
                try:
                    val = cache.get(f"{prefix}_{i}")
                    if val is not None:
                        assert val == i
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("w1",)),
            threading.Thread(target=writer, args=("w2",)),
            threading.Thread(target=reader, args=("w1",)),
            threading.Thread(target=reader, args=("w2",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_concurrent_eviction_no_crash(self) -> None:
        """Tiny cache under heavy contention doesn't raise."""
        cache = InferenceCache(max_size=5, default_ttl=10)
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            for i in range(100):
                try:
                    cache.put(f"t{tid}_{i}", i)
                    cache.get(f"t{tid}_{i}")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert cache.stats()["size"] <= 5
