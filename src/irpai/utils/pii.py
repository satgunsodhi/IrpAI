"""
PII Redactor

Regex-based detection and redaction of personally identifiable
information from extracted text.  Runs entirely in pure Python —
no external dependencies required.

Supported PII types:
    - Email addresses
    - Phone numbers (US format)
    - Social Security Numbers (US)
    - Credit card numbers
    - IP addresses
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# -- patterns ---------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


# -- public API -------------------------------------------------------------


def redact_text(text: str) -> str:
    """
    Replace all PII occurrences in *text* with ``[REDACTED_<TYPE>]`` tokens.

    >>> redact_text("Email me at alice@example.com")
    'Email me at [REDACTED_EMAIL]'
    """
    for pii_type, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
    return text


def detect_pii(text: str) -> list[dict[str, object]]:
    """
    Detect PII in *text* without redacting.

    Returns a list of ``{"type": str, "count": int}`` dicts, one per
    PII category found.
    """
    found: list[dict[str, object]] = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found.append({"type": pii_type, "count": len(matches)})
    return found
