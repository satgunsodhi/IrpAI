"""
Shared fixtures for IrpAI research benchmarks.

Provides synthetic screenshots, pre-configured pipeline instances,
and helper utilities shared across all benchmark suites.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import pytest

from irpai.config import IrpAIConfig, Tier
from irpai.pipeline import IrpAIPipeline

from tests.research.synthetic_scenes import (
    SyntheticScene,
    build_all_scenes,
    make_browser_scene,
    make_chat_scene,
    make_dashboard_scene,
    make_ide_scene,
    make_mobile_scene,
)


# ---------------------------------------------------------------------------
# Scene fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ide_scene() -> SyntheticScene:
    return make_ide_scene()


@pytest.fixture(scope="session")
def browser_scene() -> SyntheticScene:
    return make_browser_scene()


@pytest.fixture(scope="session")
def mobile_scene() -> SyntheticScene:
    return make_mobile_scene()


@pytest.fixture(scope="session")
def chat_scene() -> SyntheticScene:
    return make_chat_scene()


@pytest.fixture(scope="session")
def dashboard_scene() -> SyntheticScene:
    return make_dashboard_scene()


@pytest.fixture(scope="session")
def all_scenes() -> list[SyntheticScene]:
    return build_all_scenes()


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def lite_pipeline() -> IrpAIPipeline:
    return IrpAIPipeline(IrpAIConfig(tier=Tier.LITE))


@pytest.fixture(scope="session")
def standard_pipeline() -> IrpAIPipeline:
    return IrpAIPipeline(IrpAIConfig(tier=Tier.STANDARD))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Result of a timed execution."""

    elapsed_ms: float = 0.0
    label: str = ""


@contextmanager
def timed(label: str = "") -> Generator[TimingResult, None, None]:
    """Context manager that records wall-clock time in milliseconds."""
    result = TimingResult(label=label)
    t0 = time.perf_counter()
    yield result
    result.elapsed_ms = (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------


@dataclass
class MetricAccumulator:
    """Collects named numeric metrics for report generation."""

    data: dict[str, list[float]] = field(default_factory=dict)

    def record(self, name: str, value: float) -> None:
        self.data.setdefault(name, []).append(value)

    def mean(self, name: str) -> float:
        values = self.data.get(name, [])
        return sum(values) / len(values) if values else 0.0

    def percentile(self, name: str, pct: float) -> float:
        values = sorted(self.data.get(name, []))
        if not values:
            return 0.0
        idx = int(len(values) * pct / 100)
        return values[min(idx, len(values) - 1)]

    def total(self, name: str) -> float:
        return sum(self.data.get(name, []))

    def count(self, name: str) -> int:
        return len(self.data.get(name, []))
