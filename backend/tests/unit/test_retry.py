"""Tests for the retry decorator utility."""

import pytest

from app.core.exceptions import InferenceError
from app.ml.utils.retry import retry


class TestRetryDecorator:
    """Verify retry behaviour under various failure scenarios."""

    def test_succeeds_on_first_attempt(self) -> None:
        """Function that succeeds immediately should not retry."""
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_transient_error(self) -> None:
        """Function should be retried on retryable exceptions."""
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01)
        def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise InferenceError("model", "transient failure")
            return "recovered"

        assert fail_then_succeed() == "recovered"
        assert call_count == 3

    def test_raises_after_max_attempts(self) -> None:
        """After exhausting all attempts the last exception propagates."""
        call_count = 0

        @retry(max_attempts=2, backoff_base=0.01)
        def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise InferenceError("model", "permanent failure")

        with pytest.raises(InferenceError, match="permanent failure"):
            always_fail()

        assert call_count == 2

    def test_non_retryable_exception_propagates_immediately(self) -> None:
        """Exceptions not in retryable_exceptions should not trigger retries."""
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01)
        def raise_value_error() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            raise_value_error()

        assert call_count == 1

    def test_custom_retryable_exceptions(self) -> None:
        """Custom retryable exception tuple should be respected."""
        call_count = 0

        @retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_exceptions=(ValueError,),
        )
        def fail_with_value_error() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("retry me")
            return "done"

        assert fail_with_value_error() == "done"
        assert call_count == 2

    def test_preserves_function_metadata(self) -> None:
        """Decorated function should retain its name and docstring."""

        @retry(max_attempts=2, backoff_base=0.01)
        def my_function() -> None:
            """My docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
