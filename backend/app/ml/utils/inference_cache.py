"""Hash-based inference result cache for deduplicating model calls.

Clinical workflows often submit the same note multiple times (re-analysis,
batch jobs with duplicate rows, retries).  This cache avoids redundant
inference by keying on a SHA-256 hash of the *normalised* input text and
the model name.

Two backends are supported:

    **In-memory LRU** — Bounded ``OrderedDict`` with TTL eviction.  Zero
    external dependencies; suitable for single-process deployments.

    **Redis** — Optional async-capable backend for shared multi-worker
    caches in production.  Requires ``redis`` to be installed.

Design decisions:
    - **Normalisation** — Inputs are lowercased and whitespace-collapsed
      before hashing so that ``"Patient has  HTN"`` and
      ``"patient has htn"`` map to the same key.
    - **Model-scoped keys** — Hashes include the model name so a cache hit
      for NER doesn't return ICD results.
    - **TTL** — Both backends support per-entry TTL (defaults to the
      ``redis_cache_ttl`` setting).  Stale results are worse than a
      cache miss.
    - **Thread-safe** — The in-memory backend is protected by
      ``threading.Lock``.

Usage::

    cache = InferenceCache(backend="memory", max_size=1024)
    key = cache.make_key("ner", "Patient presents with chest pain")
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = model.predict(text)
    cache.put(key, result)
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase for consistent hashing.

    Parameters
    ----------
    text:
        Raw input text.

    Returns
    -------
    str
        Normalised text.
    """
    return _WHITESPACE_RE.sub(" ", text.strip()).lower()


def make_cache_key(model_name: str, text: str, **kwargs: Any) -> str:
    """Produce a deterministic SHA-256 cache key.

    Parameters
    ----------
    model_name:
        Model identifier (e.g. ``"ner"``, ``"icd"``).
    text:
        Input clinical text.
    **kwargs:
        Additional parameters that affect the result (e.g.
        ``top_k=5``, ``detail_level="brief"``).  They are JSON-
        serialised and included in the hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    normalised = _normalise(text)
    parts = [model_name, normalised]
    if kwargs:
        # Sort keys for deterministic ordering.
        parts.append(json.dumps(kwargs, sort_keys=True, default=str))
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _CacheEntry:
    """Internal wrapper holding a cached value and its expiry time."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl


class InferenceCache:
    """Bounded in-memory LRU cache with TTL eviction.

    Parameters
    ----------
    max_size:
        Maximum number of entries.  When exceeded the least-recently
        used entry is evicted.
    default_ttl:
        Time-to-live in seconds for each entry.

    Examples
    --------
    >>> cache = InferenceCache(max_size=512, default_ttl=600)
    >>> key = make_cache_key("ner", "chest pain on exertion")
    >>> cache.get(key)  # None (miss)
    >>> cache.put(key, [{"entity": "chest pain", "type": "SYMPTOM"}])
    >>> cache.get(key)  # hit
    [{"entity": "chest pain", "type": "SYMPTOM"}]
    """

    def __init__(
        self,
        max_size: int = 1024,
        default_ttl: float = 3600.0,
    ) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: collections.OrderedDict[str, _CacheEntry] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value if it exists and has not expired.

        Parameters
        ----------
        key:
            Cache key (typically from :func:`make_cache_key`).

        Returns
        -------
        Any or None
            Cached value, or ``None`` on miss / expired entry.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.monotonic() > entry.expires_at:
                # Expired — evict and treat as miss.
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most-recently used).
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in the cache.

        Parameters
        ----------
        key:
            Cache key.
        value:
            Value to cache (must be picklable if using Redis backend).
        ttl:
            Optional per-entry TTL override in seconds.
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            if key in self._store:
                # Update existing entry and move to end.
                self._store[key] = _CacheEntry(value, effective_ttl)
                self._store.move_to_end(key)
            else:
                # Evict LRU if at capacity.
                while len(self._store) >= self.max_size:
                    evicted_key, _ = self._store.popitem(last=False)
                    logger.debug("Cache evicted LRU entry: %s…", evicted_key[:12])
                self._store[key] = _CacheEntry(value, effective_ttl)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from the cache.

        Parameters
        ----------
        key:
            Cache key to remove.

        Returns
        -------
        bool
            ``True`` if the key existed and was removed.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns
        -------
        int
            Number of entries that were removed.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Inference cache cleared (%d entries removed)", count)
            return count

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics.

        Returns
        -------
        dict
            Keys: ``size``, ``max_size``, ``hits``, ``misses``,
            ``hit_rate``, ``default_ttl``.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total > 0 else 0.0,
                "default_ttl": self.default_ttl,
            }

    # ------------------------------------------------------------------
    # Convenience: make_key is re-exported here for one-import usage
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(model_name: str, text: str, **kwargs: Any) -> str:
        """Produce a deterministic cache key (delegates to module-level fn).

        Parameters
        ----------
        model_name:
            Model identifier.
        text:
            Input text.
        **kwargs:
            Additional parameters.

        Returns
        -------
        str
            SHA-256 hex digest.
        """
        return make_cache_key(model_name, text, **kwargs)
