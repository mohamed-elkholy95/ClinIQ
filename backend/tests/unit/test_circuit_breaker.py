"""Tests for the circuit breaker pattern implementation.

Covers all three states (CLOSED → OPEN → HALF_OPEN → CLOSED), the
decorator and context-manager interfaces, excluded exceptions, manual
reset, observability helpers, and thread-safety under contention.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from app.ml.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


# ── Helpers ──────────────────────────────────────────────────────────────


class _TransientError(Exception):
    """Simulated transient model failure."""


class _ValidationError(Exception):
    """Simulated input-validation error (should NOT trip the breaker)."""


def _make_breaker(**overrides) -> CircuitBreaker:
    """Create a breaker with short thresholds for fast tests."""
    defaults = {
        "name": "test",
        "failure_threshold": 3,
        "recovery_timeout": 0.2,
        "excluded_exceptions": (_ValidationError,),
    }
    defaults.update(overrides)
    return CircuitBreaker(**defaults)


# ── Basic state transitions ──────────────────────────────────────────────


class TestClosedState:
    """Breaker starts CLOSED and stays closed under normal conditions."""

    def test_initial_state_is_closed(self) -> None:
        cb = _make_breaker()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_successful_call_keeps_closed(self) -> None:
        cb = _make_breaker()

        @cb
        def ok() -> str:
            return "ok"

        assert ok() == "ok"
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_single_failure_stays_closed(self) -> None:
        cb = _make_breaker(failure_threshold=3)

        @cb
        def flaky() -> None:
            raise _TransientError("boom")

        with pytest.raises(_TransientError):
            flaky()

        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 1


class TestOpenState:
    """Breaker opens after *failure_threshold* consecutive failures."""

    def test_opens_after_threshold(self) -> None:
        cb = _make_breaker(failure_threshold=3)

        @cb
        def fail() -> None:
            raise _TransientError("boom")

        for _ in range(3):
            with pytest.raises(_TransientError):
                fail()

        assert cb.state is CircuitState.OPEN
        assert cb.failure_count == 3

    def test_rejects_calls_when_open(self) -> None:
        cb = _make_breaker(failure_threshold=1)

        @cb
        def fail() -> None:
            raise _TransientError("boom")

        with pytest.raises(_TransientError):
            fail()

        with pytest.raises(CircuitOpenError) as exc_info:
            fail()

        assert exc_info.value.name == "test"
        assert exc_info.value.retry_after >= 0

    def test_circuit_open_error_message(self) -> None:
        err = CircuitOpenError("ner", 5.3)
        assert "ner" in str(err)
        assert "OPEN" in str(err)
        assert "5.3" in str(err)

    def test_success_resets_failure_count(self) -> None:
        """A success before threshold resets the counter."""
        cb = _make_breaker(failure_threshold=3)
        call_count = 0

        @cb
        def sometimes_fail() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise _TransientError("boom")
            return "ok"

        with pytest.raises(_TransientError):
            sometimes_fail()
        with pytest.raises(_TransientError):
            sometimes_fail()

        assert cb.failure_count == 2
        # Third call succeeds → resets counter.
        assert sometimes_fail() == "ok"
        assert cb.failure_count == 0
        assert cb.state is CircuitState.CLOSED


class TestHalfOpenState:
    """After recovery_timeout, a probe call is allowed through."""

    def test_transitions_to_half_open(self) -> None:
        cb = _make_breaker(failure_threshold=1, recovery_timeout=0.1)

        @cb
        def fail() -> None:
            raise _TransientError("boom")

        with pytest.raises(_TransientError):
            fail()
        assert cb.state is CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state is CircuitState.HALF_OPEN

    def test_successful_probe_closes_breaker(self) -> None:
        cb = _make_breaker(failure_threshold=1, recovery_timeout=0.1)

        calls = 0

        @cb
        def recover() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _TransientError("first call fails")
            return "recovered"

        with pytest.raises(_TransientError):
            recover()
        time.sleep(0.15)

        assert recover() == "recovered"
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_failed_probe_reopens_breaker(self) -> None:
        cb = _make_breaker(failure_threshold=1, recovery_timeout=0.1)

        @cb
        def always_fail() -> None:
            raise _TransientError("still broken")

        with pytest.raises(_TransientError):
            always_fail()
        time.sleep(0.15)

        # Probe call — fails → re-opens.
        with pytest.raises(_TransientError):
            always_fail()
        assert cb.state is CircuitState.OPEN


# ── Excluded exceptions ──────────────────────────────────────────────────


class TestExcludedExceptions:
    """Excluded exceptions pass through without tripping the breaker."""

    def test_excluded_via_decorator(self) -> None:
        cb = _make_breaker(failure_threshold=1)

        @cb
        def validate() -> None:
            raise _ValidationError("bad input")

        with pytest.raises(_ValidationError):
            validate()

        assert cb.failure_count == 0
        assert cb.state is CircuitState.CLOSED

    def test_excluded_via_context_manager(self) -> None:
        cb = _make_breaker(failure_threshold=1)

        with pytest.raises(_ValidationError):
            with cb:
                raise _ValidationError("bad input")

        assert cb.failure_count == 0


# ── Context manager interface ────────────────────────────────────────────


class TestContextManager:
    """The ``with`` interface mirrors the decorator behaviour."""

    def test_success_records_success(self) -> None:
        cb = _make_breaker()
        with cb:
            result = 42
        assert result == 42
        assert cb.failure_count == 0

    def test_failure_records_failure(self) -> None:
        cb = _make_breaker(failure_threshold=5)
        with pytest.raises(_TransientError):
            with cb:
                raise _TransientError("boom")
        assert cb.failure_count == 1

    def test_rejects_when_open(self) -> None:
        cb = _make_breaker(failure_threshold=1)
        with pytest.raises(_TransientError):
            with cb:
                raise _TransientError("boom")

        with pytest.raises(CircuitOpenError):
            with cb:
                pass  # pragma: no cover — never reached


# ── Manual reset ─────────────────────────────────────────────────────────


class TestReset:
    """Manual reset forces the breaker to CLOSED."""

    def test_reset_from_open(self) -> None:
        cb = _make_breaker(failure_threshold=1)

        @cb
        def fail() -> None:
            raise _TransientError("boom")

        with pytest.raises(_TransientError):
            fail()
        assert cb.state is CircuitState.OPEN

        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None


# ── Observability ────────────────────────────────────────────────────────


class TestObservability:
    """to_dict() exposes state for health endpoints."""

    def test_to_dict_closed(self) -> None:
        cb = _make_breaker()
        d = cb.to_dict()
        assert d["name"] == "test"
        assert d["state"] == "closed"
        assert d["failure_count"] == 0
        assert d["failure_threshold"] == 3
        assert d["recovery_timeout"] == 0.2

    def test_to_dict_open(self) -> None:
        cb = _make_breaker(failure_threshold=1)

        @cb
        def fail() -> None:
            raise _TransientError("boom")

        with pytest.raises(_TransientError):
            fail()

        d = cb.to_dict()
        assert d["state"] == "open"
        assert d["failure_count"] == 1

    def test_last_failure_time_set(self) -> None:
        cb = _make_breaker()

        @cb
        def fail() -> None:
            raise _TransientError("boom")

        assert cb.last_failure_time is None
        with pytest.raises(_TransientError):
            fail()
        assert cb.last_failure_time is not None


# ── Thread safety ────────────────────────────────────────────────────────


class TestThreadSafety:
    """Concurrent calls don't corrupt state."""

    def test_concurrent_failures_converge(self) -> None:
        cb = _make_breaker(failure_threshold=5, recovery_timeout=60)
        barrier = threading.Barrier(4)
        errors: list[Exception] = []

        @cb
        def fail() -> None:
            raise _TransientError("boom")

        def worker() -> None:
            barrier.wait()
            for _ in range(3):
                try:
                    fail()
                except (_TransientError, CircuitOpenError):
                    pass
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        # After 4 threads × 3 failures each, breaker must be open.
        assert cb.state is CircuitState.OPEN

    def test_concurrent_success_and_failure(self) -> None:
        """Mixed success/failure under contention doesn't crash."""
        cb = _make_breaker(failure_threshold=100, recovery_timeout=60)
        counter = {"calls": 0}
        lock = threading.Lock()

        @cb
        def mixed() -> str:
            with lock:
                counter["calls"] += 1
                n = counter["calls"]
            if n % 3 == 0:
                raise _TransientError("boom")
            return "ok"

        def worker() -> None:
            for _ in range(20):
                try:
                    mixed()
                except (_TransientError, CircuitOpenError):
                    pass

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Should not have opened (threshold=100).
        assert cb.state is CircuitState.CLOSED
