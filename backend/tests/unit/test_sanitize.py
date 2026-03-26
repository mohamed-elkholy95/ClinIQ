"""Tests for the input sanitization middleware and utility functions.

Validates that clinical text containing encoding artifacts, control
characters, null bytes, and byte-order marks is correctly cleaned
before reaching ML inference modules or database storage.
"""

import pytest

from app.middleware.sanitize import (
    InputSanitizationMiddleware,
    sanitize_text,
    sanitize_dict,
    _sanitize_list,
    _DEFAULT_MAX_TEXT_CHARS,
)


# ─── sanitize_text ────────────────────────────────────────────


class TestSanitizeText:
    """Tests for the core sanitize_text function."""

    def test_removes_null_bytes(self) -> None:
        """Null bytes from binary protocols are stripped."""
        assert sanitize_text("Patient\x00 is stable") == "Patient is stable"

    def test_removes_multiple_null_bytes(self) -> None:
        """Multiple scattered null bytes are all removed."""
        assert sanitize_text("\x00Hello\x00World\x00") == "HelloWorld"

    def test_removes_bom(self) -> None:
        """UTF-8 byte-order mark at start of text is removed."""
        assert sanitize_text("\ufeffDiagnosis: HTN") == "Diagnosis: HTN"

    def test_removes_bom_mid_text(self) -> None:
        """BOM characters appearing mid-text are also removed."""
        assert sanitize_text("A\ufeffB") == "AB"

    def test_strips_control_characters(self) -> None:
        """C0 control characters (except tab, LF, CR) are stripped."""
        # \x01 (SOH), \x02 (STX), \x07 (BEL), \x08 (BS)
        assert sanitize_text("A\x01B\x02C\x07D\x08E") == "ABCDE"

    def test_preserves_tabs(self) -> None:
        """Tab characters carry structural meaning and must be kept."""
        text = "Medication:\tLisinopril 10mg"
        assert sanitize_text(text) == text

    def test_preserves_newlines(self) -> None:
        """Newlines delimit sections and list items — must survive."""
        text = "Line 1\nLine 2\nLine 3"
        assert sanitize_text(text) == text

    def test_preserves_carriage_returns(self) -> None:
        """Windows-style \\r\\n line endings must be preserved."""
        text = "Line 1\r\nLine 2"
        assert sanitize_text(text) == text

    def test_strips_del_character(self) -> None:
        """DEL (\\x7f) is a control character and should be removed."""
        assert sanitize_text("ABC\x7fDEF") == "ABCDEF"

    def test_strips_vertical_tab(self) -> None:
        """Vertical tab (\\x0b) is stripped."""
        assert sanitize_text("A\x0bB") == "AB"

    def test_strips_escape_sequences(self) -> None:
        """Characters in \\x0e–\\x1f range are stripped."""
        assert sanitize_text("A\x0eB\x1fC") == "ABC"

    def test_truncates_overlong_text(self) -> None:
        """Text exceeding max_chars is truncated."""
        text = "A" * 100
        result = sanitize_text(text, max_chars=50)
        assert len(result) == 50
        assert result == "A" * 50

    def test_does_not_truncate_short_text(self) -> None:
        """Text within limits is returned unchanged."""
        text = "Short clinical note."
        assert sanitize_text(text, max_chars=1000) == text

    def test_empty_string_passthrough(self) -> None:
        """Empty string returns empty string."""
        assert sanitize_text("") == ""

    def test_none_like_empty(self) -> None:
        """Empty-ish strings are handled."""
        assert sanitize_text("") == ""

    def test_preserves_unicode_medical_symbols(self) -> None:
        """Medical symbols (°, µ, ±) must survive sanitization."""
        text = "Temp: 37.2°C, pH 7.4 ± 0.02, µg/dL"
        assert sanitize_text(text) == text

    def test_preserves_replacement_character(self) -> None:
        """Unicode replacement character (U+FFFD) is kept as a marker."""
        text = "Data\ufffdcorrupt"
        assert sanitize_text(text) == text

    def test_combined_artifacts(self) -> None:
        """Multiple artifact types in a single string are all handled."""
        text = "\ufeff\x00Patient\x01 has\x07 HTN\x00"
        assert sanitize_text(text) == "Patient has HTN"


# ─── sanitize_dict ────────────────────────────────────────────


class TestSanitizeDict:
    """Tests for recursive dictionary sanitization."""

    def test_sanitizes_string_values(self) -> None:
        """String values in a flat dict are sanitized."""
        data = {"text": "Hello\x00World", "count": 5}
        result = sanitize_dict(data)
        assert result == {"text": "HelloWorld", "count": 5}

    def test_preserves_non_string_values(self) -> None:
        """Integers, floats, booleans, and None pass through."""
        data = {"a": 42, "b": 3.14, "c": True, "d": None}
        result = sanitize_dict(data)
        assert result == data

    def test_nested_dict_sanitization(self) -> None:
        """Nested dictionaries are recursively sanitized."""
        data = {"outer": {"inner": "A\x00B"}}
        result = sanitize_dict(data)
        assert result == {"outer": {"inner": "AB"}}

    def test_list_values_sanitized(self) -> None:
        """Lists of strings within a dict are sanitized."""
        data = {"items": ["A\x00", "B\x01C"]}
        result = sanitize_dict(data)
        assert result == {"items": ["A", "BC"]}

    def test_deeply_nested_structure(self) -> None:
        """Three levels of nesting are handled."""
        data = {"a": {"b": [{"c": "X\x00Y"}]}}
        result = sanitize_dict(data)
        assert result == {"a": {"b": [{"c": "XY"}]}}

    def test_empty_dict(self) -> None:
        """Empty dict returns empty dict."""
        assert sanitize_dict({}) == {}

    def test_max_chars_applied(self) -> None:
        """max_text_chars is propagated to nested string values."""
        data = {"text": "A" * 100}
        result = sanitize_dict(data, max_text_chars=50)
        assert len(result["text"]) == 50


# ─── _sanitize_list ───────────────────────────────────────────


class TestSanitizeList:
    """Tests for the list sanitization helper."""

    def test_sanitizes_string_items(self) -> None:
        """String items in a list are sanitized."""
        items = ["A\x00B", "C\x01D"]
        result = _sanitize_list(items, _DEFAULT_MAX_TEXT_CHARS)
        assert result == ["AB", "CD"]

    def test_preserves_non_string_items(self) -> None:
        """Non-string items pass through unchanged."""
        items = [1, 2.5, True, None]
        result = _sanitize_list(items, _DEFAULT_MAX_TEXT_CHARS)
        assert result == items

    def test_nested_dicts_in_list(self) -> None:
        """Dicts inside lists are recursively sanitized."""
        items = [{"text": "A\x00B"}]
        result = _sanitize_list(items, _DEFAULT_MAX_TEXT_CHARS)
        assert result == [{"text": "AB"}]

    def test_nested_lists(self) -> None:
        """Nested lists are recursively sanitized."""
        items = [["A\x00B", "C"]]
        result = _sanitize_list(items, _DEFAULT_MAX_TEXT_CHARS)
        assert result == [["AB", "C"]]

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert _sanitize_list([], _DEFAULT_MAX_TEXT_CHARS) == []


# ─── InputSanitizationMiddleware ──────────────────────────────


class TestInputSanitizationMiddleware:
    """Tests for middleware initialization and configuration."""

    def test_default_configuration(self) -> None:
        """Middleware initializes with default limits."""
        middleware = InputSanitizationMiddleware(app=None)
        assert middleware.max_body_bytes == 2 * 1024 * 1024
        assert middleware.max_text_chars == 500_000

    def test_custom_configuration(self) -> None:
        """Middleware accepts custom limits."""
        middleware = InputSanitizationMiddleware(
            app=None,
            max_body_bytes=1024,
            max_text_chars=100,
        )
        assert middleware.max_body_bytes == 1024
        assert middleware.max_text_chars == 100
