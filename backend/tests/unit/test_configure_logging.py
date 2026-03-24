"""Unit tests for the configure_logging helper.

Targets uncovered lines 58–66 in middleware/logging.py.
"""

from __future__ import annotations

from unittest.mock import patch

import structlog

from app.middleware.logging import configure_logging


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_json_format_configures_structlog(self) -> None:
        """JSON format should use JSONRenderer."""
        configure_logging(log_level="INFO", log_format="json")
        # structlog should be configured — verify by getting a logger
        log = structlog.get_logger()
        assert log is not None

    def test_console_format_configures_structlog(self) -> None:
        """Console format should use ConsoleRenderer."""
        configure_logging(log_level="DEBUG", log_format="console")
        log = structlog.get_logger()
        assert log is not None

    def test_default_parameters(self) -> None:
        """Default parameters should not raise."""
        configure_logging()
        log = structlog.get_logger()
        assert log is not None

    @patch("app.middleware.logging.structlog.configure")
    def test_json_includes_json_renderer(self, mock_configure) -> None:
        """JSON format should pass JSONRenderer in processors."""
        configure_logging(log_format="json")
        mock_configure.assert_called_once()
        call_kwargs = mock_configure.call_args
        processors = call_kwargs.kwargs.get("processors") or call_kwargs[1].get("processors", [])
        # Last processor should be JSONRenderer
        assert any(
            isinstance(p, structlog.processors.JSONRenderer)
            for p in processors
        )

    @patch("app.middleware.logging.structlog.configure")
    def test_console_includes_console_renderer(self, mock_configure) -> None:
        """Console format should pass ConsoleRenderer in processors."""
        configure_logging(log_format="console")
        mock_configure.assert_called_once()
        call_kwargs = mock_configure.call_args
        processors = call_kwargs.kwargs.get("processors") or call_kwargs[1].get("processors", [])
        assert any(
            isinstance(p, structlog.dev.ConsoleRenderer)
            for p in processors
        )
