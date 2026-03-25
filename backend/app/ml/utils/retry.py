"""Retry utilities for resilient ML inference.

Provides a configurable retry decorator with exponential back-off,
jitter, and per-exception filtering — designed for wrapping model
inference calls that may fail transiently (OOM spikes, temporary file
locks, intermittent GPU errors).

Usage
-----
>>> from app.ml.utils.retry import retry
>>>
>>> @retry(max_attempts=3, backoff_base=0.5)
... def predict(text: str) -> dict:
...     return model.run(text)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app.core.exceptions import InferenceError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 10.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[BaseException], ...] = (
        InferenceError,
        OSError,
        RuntimeError,
    ),
) -> Callable[[F], F]:
    """Decorator that retries a function on transient failures.

    Parameters
    ----------
    max_attempts:
        Total number of attempts (including the first call).
    backoff_base:
        Base delay in seconds; doubles after each failed attempt.
    backoff_max:
        Cap on the computed delay to prevent excessive waits.
    jitter:
        When *True*, add uniform random jitter (0 – delay) to spread
        out retries across concurrent callers.
    retryable_exceptions:
        Tuple of exception types that should trigger a retry.  Any
        exception **not** in this tuple is raised immediately.

    Returns
    -------
    Callable
        Wrapped function with retry logic.
    """

    def decorator(func: F) -> F:
        """Wrap *func* with retry logic and return the decorated callable."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute the wrapped function, retrying on transient failures."""
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "All %d attempts exhausted for %s: %s",
                            max_attempts,
                            func.__qualname__,
                            exc,
                        )
                        raise
                    delay = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                    if jitter:
                        delay = random.uniform(0, delay)  # noqa: S311
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s); retrying in %.2fs",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            # Should be unreachable, but satisfies type-checkers.
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
