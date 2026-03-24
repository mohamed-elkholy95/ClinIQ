"""Input validation utilities for clinical text processing.

Provides pre-inference checks to reject or flag problematic inputs
early, before they consume pipeline resources.  Validation covers
length limits, encoding issues, and basic content-quality heuristics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Upper byte-length limit before any processing (10 MB).
_MAX_RAW_BYTES = 10 * 1024 * 1024

# Minimum meaningful clinical text length (characters).
_MIN_CLINICAL_LENGTH = 20

# Regex for detecting high ratios of non-alphanumeric characters —
# a sign of binary or garbled input.
_NOISE_PATTERN = re.compile(r"[^\w\s.,;:!?\-/()']", re.UNICODE)


@dataclass
class ValidationResult:
    """Outcome of a text validation check.

    Attributes
    ----------
    is_valid:
        Whether the text passed all checks.
    errors:
        Hard failures — the text should be rejected.
    warnings:
        Soft issues — the text can proceed but results may be degraded.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_clinical_text(
    text: str,
    *,
    max_length: int = 100_000,
    min_length: int = _MIN_CLINICAL_LENGTH,
    max_raw_bytes: int = _MAX_RAW_BYTES,
) -> ValidationResult:
    """Validate raw clinical text before pipeline processing.

    Parameters
    ----------
    text:
        The raw text to validate.
    max_length:
        Maximum allowed character count.
    min_length:
        Minimum character count for meaningful clinical content.
    max_raw_bytes:
        Maximum byte size of the encoded text.

    Returns
    -------
    ValidationResult
        Aggregated validation outcome.
    """
    result = ValidationResult()

    # ── Type check ────────────────────────────────────────────
    if not isinstance(text, str):
        result.is_valid = False
        result.errors.append(f"Expected str, got {type(text).__name__}")
        return result

    # ── Byte-size gate (guards against oversized uploads) ─────
    byte_len = len(text.encode("utf-8", errors="replace"))
    if byte_len > max_raw_bytes:
        result.is_valid = False
        result.errors.append(
            f"Text exceeds maximum byte size ({byte_len:,} > {max_raw_bytes:,})"
        )
        return result

    stripped = text.strip()

    # ── Emptiness ─────────────────────────────────────────────
    if not stripped:
        result.is_valid = False
        result.errors.append("Text is empty or whitespace-only")
        return result

    # ── Length bounds ──────────────────────────────────────────
    if len(stripped) < min_length:
        result.is_valid = False
        result.errors.append(
            f"Text too short for meaningful analysis "
            f"({len(stripped)} < {min_length} chars)"
        )
        return result

    if len(stripped) > max_length:
        result.warnings.append(
            f"Text exceeds recommended length ({len(stripped):,} > {max_length:,}); "
            "it will be truncated during preprocessing"
        )

    # ── Noise ratio ───────────────────────────────────────────
    noise_chars = len(_NOISE_PATTERN.findall(stripped))
    noise_ratio = noise_chars / len(stripped) if stripped else 0.0
    if noise_ratio > 0.30:
        result.warnings.append(
            f"High noise ratio ({noise_ratio:.0%}) — text may contain "
            "binary data or encoding artefacts"
        )

    # ── NULL bytes ────────────────────────────────────────────
    if "\x00" in text:
        result.is_valid = False
        result.errors.append("Text contains null bytes — likely binary data")

    # ── Minimal word count ────────────────────────────────────
    word_count = len(stripped.split())
    if word_count < 3:
        result.warnings.append(
            f"Very few words ({word_count}) — analysis quality may be low"
        )

    if not result.errors:
        result.is_valid = True

    return result
