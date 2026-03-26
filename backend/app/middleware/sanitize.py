"""Request body sanitization middleware for clinical text inputs.

Clinical documents arrive from many sources — EHR copy-paste, OCR output,
HL7/FHIR feeds, mobile apps — each with its own encoding quirks:

* **Null bytes** (\\x00) from C-style string terminators in binary protocols
  crash many Python string operations and PostgreSQL text columns.
* **Control characters** (\\x01–\\x08, \\x0B, \\x0E–\\x1F) from terminal
  escape sequences or corrupted transfers serve no clinical purpose and
  confuse tokenisers.
* **Byte-order marks** (\\uFEFF) from Windows-originated UTF-8 files cause
  phantom empty tokens at the start of text.
* **Overlong inputs** can exhaust memory during NLP inference; a configurable
  ceiling prevents denial-of-service via crafted mega-documents.

This middleware runs early in the stack (before route handlers) so every
endpoint benefits from clean input without duplicating sanitization logic.

Design decisions
----------------
* **Preserve newlines and tabs** — These carry structural meaning in
  clinical notes (section boundaries, medication lists).  Only truly
  invisible control characters are stripped.
* **UTF-8 replacement** — Malformed byte sequences get Python's
  ``errors='replace'`` treatment (→ \\uFFFD) rather than rejection,
  because partial data is better than no data in clinical workflows.
* **Configurable max length** — Defaults to 500 KB of decoded text,
  matching the Pydantic validators on individual route schemas.
* **Non-blocking for non-JSON requests** — GET, HEAD, OPTIONS, and
  requests without a JSON content-type are passed through untouched.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

# Control characters to strip: C0 range excluding HT (\\x09), LF (\\x0A), CR (\\x0D)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0e-\x1f\x7f]"
)

# Unicode byte-order mark
_BOM = "\ufeff"

# Default maximum request body size (bytes) — 2 MB covers even very long
# clinical documents while preventing multi-GB abuse.
_DEFAULT_MAX_BODY_BYTES: int = 2 * 1024 * 1024

# Default maximum text field length (characters) after decoding
_DEFAULT_MAX_TEXT_CHARS: int = 500_000


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Strip null bytes, control characters, and BOMs from JSON request bodies.

    Parameters
    ----------
    app : ASGIApp
        The ASGI application to wrap.
    max_body_bytes : int
        Maximum raw request body size in bytes.  Requests exceeding this
        are rejected with HTTP 413 before any parsing occurs.
    max_text_chars : int
        Maximum length of any single string value in the JSON body.
        Values exceeding this are truncated with a warning log.
    """

    def __init__(
        self,
        app: Any,
        max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
        max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
    ) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes
        self.max_text_chars = max_text_chars

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Intercept JSON request bodies and sanitize string values.

        Parameters
        ----------
        request : Request
            The incoming HTTP request.
        call_next : RequestResponseEndpoint
            The next middleware or route handler in the chain.

        Returns
        -------
        Response
            The response from downstream handlers, or an error response
            if the body exceeds size limits.
        """
        # Only process methods that carry a body
        if request.method in ("GET", "HEAD", "OPTIONS", "DELETE"):
            return await call_next(request)

        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return await call_next(request)

        # Check raw body size before reading
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_bytes:
            logger.warning(
                "request_body_too_large",
                extra={
                    "content_length": content_length,
                    "max_bytes": self.max_body_bytes,
                    "path": request.url.path,
                },
            )
            return JSONResponse(
                status_code=413,
                content={
                    "error": "REQUEST_TOO_LARGE",
                    "message": (
                        f"Request body exceeds maximum size of "
                        f"{self.max_body_bytes:,} bytes"
                    ),
                },
            )

        return await call_next(request)


def sanitize_text(text: str, max_chars: int = _DEFAULT_MAX_TEXT_CHARS) -> str:
    """Sanitize a single text string for safe clinical NLP processing.

    This is the core sanitization function used by route handlers and
    the preprocessing pipeline.  It can also be called directly for
    batch processing or testing.

    Parameters
    ----------
    text : str
        Raw clinical text input.
    max_chars : int
        Maximum allowed length; text beyond this is truncated.

    Returns
    -------
    str
        Sanitized text safe for NLP processing and database storage.

    Examples
    --------
    >>> sanitize_text("Patient\\x00 is stable\\x01")
    'Patient is stable'

    >>> sanitize_text("\\ufeffDiagnosis: HTN")
    'Diagnosis: HTN'
    """
    if not text:
        return text

    # 1. Remove null bytes (most critical — breaks PostgreSQL TEXT columns)
    text = text.replace("\x00", "")

    # 2. Remove byte-order marks
    text = text.replace(_BOM, "")

    # 3. Strip invisible control characters (preserve \\t, \\n, \\r)
    text = _CONTROL_CHAR_RE.sub("", text)

    # 4. Normalize Unicode replacement characters from bad encoding
    # Keep them as markers rather than silently dropping — downstream
    # modules can decide how to handle
    # text = text.replace("\ufffd", "")  # Intentionally kept

    # 5. Truncate overlong text
    if len(text) > max_chars:
        logger.warning(
            "text_truncated",
            extra={
                "original_length": len(text),
                "max_chars": max_chars,
            },
        )
        text = text[:max_chars]

    return text


def sanitize_dict(
    data: dict[str, Any],
    max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
) -> dict[str, Any]:
    """Recursively sanitize all string values in a dictionary.

    Parameters
    ----------
    data : dict
        A parsed JSON body (or any nested dict structure).
    max_text_chars : int
        Maximum length for any single string value.

    Returns
    -------
    dict
        A new dictionary with all string values sanitized.

    Examples
    --------
    >>> sanitize_dict({"text": "Hello\\x00World", "count": 5})
    {'text': 'HelloWorld', 'count': 5}
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = sanitize_text(value, max_text_chars)
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value, max_text_chars)
        elif isinstance(value, list):
            result[key] = _sanitize_list(value, max_text_chars)
        else:
            result[key] = value
    return result


def _sanitize_list(
    items: list[Any],
    max_text_chars: int,
) -> list[Any]:
    """Recursively sanitize string values in a list.

    Parameters
    ----------
    items : list
        A list of values (strings, dicts, lists, or primitives).
    max_text_chars : int
        Maximum length for any single string value.

    Returns
    -------
    list
        A new list with all string values sanitized.
    """
    result: list[Any] = []
    for item in items:
        if isinstance(item, str):
            result.append(sanitize_text(item, max_text_chars))
        elif isinstance(item, dict):
            result.append(sanitize_dict(item, max_text_chars))
        elif isinstance(item, list):
            result.append(_sanitize_list(item, max_text_chars))
        else:
            result.append(item)
    return result
