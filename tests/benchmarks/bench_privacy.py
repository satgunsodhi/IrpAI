"""
Privacy Benchmark
=================

Tests IrpAI's PII detection and redaction capabilities using
the ``irpai.utils.pii`` module:
- Detection accuracy (precision & recall)
- Redaction completeness
- Per-PII-type coverage
- False positive rate
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irpai.utils.pii import detect_pii, redact_text


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PIICategoryResult:
    """Results for a single PII category."""
    pii_type: str
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    precision: float = 0.0
    recall: float = 0.0


@dataclass
class PrivacyBenchmarkResults:
    category_results: list[PIICategoryResult] = field(default_factory=list)
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    redaction_completeness: float = 0.0
    false_positive_rate: float = 0.0
    total_test_cases: int = 0


# ---------------------------------------------------------------------------
# Test data with known labels
# ---------------------------------------------------------------------------

# (input_text, expected_pii_types)
PII_TEST_CASES: list[tuple[str, list[str]]] = [
    # Email addresses
    ("Contact alice@example.com for support", ["email"]),
    ("Send to user.name@company.co.uk", ["email"]),
    ("Email john.doe123@gmail.com", ["email"]),

    # Phone numbers
    ("Call 555-123-4567 for info", ["phone"]),
    ("Phone: 800.555.1234", ["phone"]),
    ("Reach us at 5551234567", ["phone"]),

    # SSN
    ("SSN: 123-45-6789", ["ssn"]),
    ("Social: 987-65-4321", ["ssn"]),

    # Credit cards
    ("Card: 4111-1111-1111-1111", ["credit_card"]),
    ("CC 4111 1111 1111 1111", ["credit_card"]),
    ("Pay with 5500000000000004", ["credit_card"]),

    # IP addresses
    ("Server at 192.168.1.1", ["ip_address"]),
    ("Connect to 10.0.0.255", ["ip_address"]),
    ("DNS: 8.8.8.8", ["ip_address"]),

    # Multiple PII in one string
    (
        "User john@corp.com logged in from 10.0.0.1, phone 555-000-1234",
        ["email", "ip_address", "phone"],
    ),
    (
        "SSN 123-45-6789, card 4111-1111-1111-1111",
        ["ssn", "credit_card"],
    ),

    # No PII (should produce no detections)
    ("The weather is nice today", []),
    ("IrpAI pipeline completed in 287ms", []),
    ("Total regions: 12, confidence: 0.95", []),
    ("Meeting at 3pm in room 204", []),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_privacy_benchmark() -> PrivacyBenchmarkResults:
    """Execute the privacy/PII benchmark."""
    results = PrivacyBenchmarkResults(total_test_cases=len(PII_TEST_CASES))

    # Per-category tracking
    categories: dict[str, PIICategoryResult] = {}
    for pii_type in ["email", "phone", "ssn", "credit_card", "ip_address"]:
        categories[pii_type] = PIICategoryResult(pii_type=pii_type)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_non_pii = 0
    non_pii_flagged = 0

    redaction_passes = 0
    redaction_total = 0

    for text, expected_types in PII_TEST_CASES:
        # Detection
        found = detect_pii(text)
        found_types = {item["type"] for item in found}
        expected_set = set(expected_types)

        for pii_type in expected_set:
            if pii_type in found_types:
                categories[pii_type].true_positives += 1
                total_tp += 1
            else:
                categories[pii_type].false_negatives += 1
                total_fn += 1

        for pii_type in found_types:
            if pii_type not in expected_set:
                if pii_type in categories:
                    categories[pii_type].false_positives += 1
                total_fp += 1

        # Track false positives on clean text
        if not expected_types:
            total_non_pii += 1
            if found:
                non_pii_flagged += 1

        # Redaction completeness
        if expected_types:
            redacted = redact_text(text)
            redaction_total += 1
            # Check that all original PII strings are gone
            all_redacted = True
            for pii_type in expected_types:
                marker = f"[REDACTED_{pii_type.upper()}]"
                if marker not in redacted:
                    all_redacted = False
            if all_redacted:
                redaction_passes += 1

    # Compute per-category precision/recall
    for cat in categories.values():
        detected = cat.true_positives + cat.false_positives
        if detected > 0:
            cat.precision = round(cat.true_positives / detected, 3)
        expected = cat.true_positives + cat.false_negatives
        if expected > 0:
            cat.recall = round(cat.true_positives / expected, 3)

    results.category_results = list(categories.values())

    # Overall metrics
    total_detected = total_tp + total_fp
    results.overall_precision = round(total_tp / max(total_detected, 1), 3)
    results.overall_recall = round(total_tp / max(total_tp + total_fn, 1), 3)
    results.redaction_completeness = round(
        redaction_passes / max(redaction_total, 1), 3
    )
    results.false_positive_rate = round(
        non_pii_flagged / max(total_non_pii, 1), 3
    )

    return results
