"""Circuit breaker for resilient ML model inference.

Implements the circuit breaker pattern to prevent cascading failures when
a model is repeatedly failing.  After a configurable number of consecutive
failures the breaker "opens" and immediately rejects new requests for a
cool-down period, giving the model (or its dependency) time to recover.

States:
    **CLOSED** — Normal operation.  Failures are counted; after
    *failure_threshold* consecutive failures the breaker opens.

    **OPEN** — All calls are rejected with ``CircuitOpenError`` without
    invoking the model.  After *recovery_timeout* seconds the breaker
    transitions to half-open.

    **HALF_OPEN** — A single probe call is allowed through.  If it
    succeeds the breaker closes; if it fails the breaker re-opens.

Design decisions:
    - **Thread-safe** via ``threading.Lock`` — safe for FastAPI's
      default threadpool executor.
    - **Per-model granularity** — instantiate one breaker per model so
      a flaky NER model doesn't block the healthy ICD classifier.
    - **Decorator and context-manager API** — the decorator suits model
      methods; the context manager suits ad-hoc inference blocks.
    - **Observable** — exposes ``state``, ``failure_count``, and
      ``last_failure_time`` for health-check endpoints.

Usage::

    breaker = CircuitBreaker(name="ner", failure_threshold=5)

    @breaker
    def predict(text: str) -> list[Entity]:
        return ner_model.extract(text)

    # Or as a context manager:
    with breaker:
        result = ner_model.extract(text)
"""

from __future__ import annotations

import enum
import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(enum.Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open.

    Parameters
    ----------
    name:
        Identifier of the breaker that rejected the call.
    retry_after:
        Seconds until the breaker will transition to half-open.
    """

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN — retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """Thread-safe circuit breaker for wrapping model inference calls.

    Parameters
    ----------
    name:
        Human-readable identifier (e.g. ``"ner"``, ``"icd"``).
    failure_threshold:
        Number of consecutive failures before the breaker opens.
    recovery_timeout:
        Seconds to wait in the OPEN state before allowing a probe.
    excluded_exceptions:
        Exception types that should **not** count as failures (e.g.
        validation errors caused by bad input, not model faults).

    Examples
    --------
    >>> cb = CircuitBreaker("ner", failure_threshold=3, recovery_timeout=30)
    >>> @cb
    ... def extract(text):
    ...     return model.run(text)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        excluded_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.excluded_exceptions = excluded_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Read-only properties for observability
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current state of the breaker (may auto-transition to HALF_OPEN)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures since last success."""
        return self._failure_count

    @property
    def last_failure_time(self) -> float | None:
        """Monotonic timestamp of the most recent recorded failure."""
        return self._last_failure_time

    # ------------------------------------------------------------------
    # Internal state machine helpers (caller must hold ``_lock``)
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """Transition from OPEN → HALF_OPEN if the recovery timeout elapsed."""
        if self._state is CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    "Circuit breaker '%s': OPEN → HALF_OPEN (%.1fs elapsed)",
                    self.name,
                    elapsed,
                )
                self._state = CircuitState.HALF_OPEN

    def _record_success(self) -> None:
        """Reset counters after a successful call."""
        with self._lock:
            prev = self._state
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            if prev is not CircuitState.CLOSED:
                logger.info(
                    "Circuit breaker '%s': %s → CLOSED (success)",
                    self.name,
                    prev.value,
                )

    def _record_failure(self) -> None:
        """Increment the failure counter, potentially opening the breaker."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state is CircuitState.HALF_OPEN:
                # Probe failed — re-open.
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker '%s': HALF_OPEN → OPEN (probe failed, "
                    "failures=%d)",
                    self.name,
                    self._failure_count,
                )
            elif (
                self._state is CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker '%s': CLOSED → OPEN (threshold %d reached)",
                    self.name,
                    self.failure_threshold,
                )

    def _check_state(self) -> None:
        """Raise ``CircuitOpenError`` if the breaker is currently open."""
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state is CircuitState.OPEN:
                # Calculate how long until half-open.
                elapsed = (
                    time.monotonic() - self._last_failure_time
                    if self._last_failure_time
                    else 0.0
                )
                retry_after = max(0.0, self.recovery_timeout - elapsed)
                raise CircuitOpenError(self.name, retry_after)

    # ------------------------------------------------------------------
    # Decorator interface
    # ------------------------------------------------------------------

    def __call__(self, func: F) -> F:
        """Decorate *func* to be protected by this circuit breaker.

        Parameters
        ----------
        func:
            The function to wrap.

        Returns
        -------
        F
            Decorated function that raises ``CircuitOpenError`` when the
            breaker is open.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute the wrapped function under circuit breaker protection."""
            self._check_state()
            try:
                result = func(*args, **kwargs)
            except self.excluded_exceptions:
                # Input-validation errors aren't model faults — let them
                # pass through without tripping the breaker.
                raise
            except Exception:
                self._record_failure()
                raise
            else:
                self._record_success()
                return result

        return wrapper  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> CircuitBreaker:
        """Enter the breaker context, raising if the circuit is open."""
        self._check_state()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Record success or failure on context exit.

        Parameters
        ----------
        exc_type:
            Exception class, if any.
        exc_val:
            Exception instance, if any.
        exc_tb:
            Traceback, if any.

        Returns
        -------
        bool
            Always ``False`` — exceptions are never suppressed.
        """
        if exc_type is None:
            self._record_success()
        elif self.excluded_exceptions and issubclass(
            exc_type, self.excluded_exceptions
        ):
            pass  # Excluded — don't trip the breaker.
        else:
            self._record_failure()
        return False

    # ------------------------------------------------------------------
    # Manual reset (for tests and admin endpoints)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Force the breaker back to CLOSED with zero failures.

        Useful in tests and for admin override endpoints.
        """
        with self._lock:
            prev = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(
                "Circuit breaker '%s': %s → CLOSED (manual reset)",
                self.name,
                prev.value,
            )

    # ------------------------------------------------------------------
    # Serialisation for health endpoints
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the breaker state.

        Returns
        -------
        dict
            Keys: ``name``, ``state``, ``failure_count``,
            ``failure_threshold``, ``recovery_timeout``.
        """
        return {
            "name": self.name,
            "state": self.state.value,  # triggers auto-transition check
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }
