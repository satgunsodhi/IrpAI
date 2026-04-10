"""
Latency Profiling Benchmark
============================

Per-layer and end-to-end timing analysis:
- Layer breakdown (preprocessing, detection, layout, assembly)
- End-to-end latency across all scenes
- Warmup cost (first-run vs steady-state)
- Scaling profile (latency vs image resolution)
- Throughput estimate (images/second)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image

from irpai.config import IrpAIConfig, Tier
from irpai.pipeline import IrpAIPipeline

from tests.research.synthetic_scenes import SyntheticScene, build_all_scenes


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LayerTiming:
    """Timing for a single pipeline layer."""
    layer: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, pct: float) -> float:
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        idx = int(len(s) * pct / 100)
        return s[min(idx, len(s) - 1)]


@dataclass
class ScaleProfile:
    """Latency at a specific resolution."""
    resolution: str
    width: int
    height: int
    latency_ms: float = 0.0


@dataclass
class LatencyBenchmarkResults:
    layer_timings: dict[str, LayerTiming] = field(default_factory=dict)
    e2e_times_ms: list[float] = field(default_factory=list)
    warmup_first_ms: float = 0.0
    warmup_steady_ms: float = 0.0
    scale_profiles: list[ScaleProfile] = field(default_factory=list)
    throughput_ips: float = 0.0  # images per second
    tier: str = ""


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

NUM_WARMUP_RUNS = 2
NUM_STEADY_RUNS = 5

SCALE_RESOLUTIONS = [
    (640, 480),
    (1024, 768),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]


def run_latency_benchmark() -> list[LatencyBenchmarkResults]:
    """Execute latency profiling for each available tier."""
    scenes = build_all_scenes()
    all_results: list[LatencyBenchmarkResults] = []

    for tier in [Tier.LITE, Tier.STANDARD]:
        try:
            pipeline = IrpAIPipeline(IrpAIConfig(tier=tier))
        except Exception:
            continue

        results = LatencyBenchmarkResults(tier=tier.value)

        # Initialize layer timing accumulators
        for layer in ["preprocessing", "detection", "layout", "assembly", "total"]:
            results.layer_timings[layer] = LayerTiming(layer=layer)

        # Warmup runs
        warmup_times = []
        for _ in range(NUM_WARMUP_RUNS):
            t0 = time.perf_counter()
            try:
                pipeline.run(scenes[0].image)
            except Exception:
                pass
            warmup_times.append((time.perf_counter() - t0) * 1000)

        if warmup_times:
            results.warmup_first_ms = round(warmup_times[0], 1)

        # Steady-state benchmark across all scenes
        for scene in scenes:
            for _ in range(NUM_STEADY_RUNS):
                timings = _profile_run(pipeline, scene.image)
                if timings:
                    for layer, ms in timings.items():
                        if layer in results.layer_timings:
                            results.layer_timings[layer].times_ms.append(ms)
                    results.e2e_times_ms.append(timings.get("total", 0))

        if results.e2e_times_ms:
            results.warmup_steady_ms = round(
                sum(results.e2e_times_ms) / len(results.e2e_times_ms), 1
            )

        # Scaling profile
        for w, h in SCALE_RESOLUTIONS:
            scaled = scenes[0].image.resize((w, h), Image.Resampling.LANCZOS)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                try:
                    pipeline.run(scaled)
                except Exception:
                    pass
                times.append((time.perf_counter() - t0) * 1000)
            avg = sum(times) / len(times) if times else 0
            results.scale_profiles.append(ScaleProfile(
                resolution=f"{w}×{h}",
                width=w, height=h,
                latency_ms=round(avg, 1),
            ))

        # Throughput
        if results.e2e_times_ms:
            avg_ms = sum(results.e2e_times_ms) / len(results.e2e_times_ms)
            results.throughput_ips = round(1000 / avg_ms, 2) if avg_ms > 0 else 0

        all_results.append(results)

    return all_results


def _profile_run(
    pipeline: IrpAIPipeline,
    image: Image.Image,
) -> dict[str, float]:
    """Run pipeline and capture per-layer timings."""
    timings: dict[str, float] = {}

    t_total = time.perf_counter()

    try:
        # Preprocessing
        t0 = time.perf_counter()
        normalized = pipeline._normalizer.normalize(image)
        timings["preprocessing"] = round((time.perf_counter() - t0) * 1000, 2)

        # Detection
        t0 = time.perf_counter()
        detections = pipeline._run_detection(normalized)
        timings["detection"] = round((time.perf_counter() - t0) * 1000, 2)

        # OCR — skip if engine not available, measure the attempt
        t0 = time.perf_counter()
        try:
            ocr_result = pipeline._run_ocr(normalized)
        except Exception:
            from irpai.ocr.base import OCRResult
            w, h = normalized.processed_size
            ocr_result = OCRResult(image_size=(w, h))
        timings["ocr"] = round((time.perf_counter() - t0) * 1000, 2)

        # Layout
        t0 = time.perf_counter()
        layout = pipeline._layout.analyze(ocr_result, detections)
        timings["layout"] = round((time.perf_counter() - t0) * 1000, 2)

        # Assembly
        t0 = time.perf_counter()
        pipeline._assembler.assemble(layout=layout, ocr=ocr_result, detections=detections)
        timings["assembly"] = round((time.perf_counter() - t0) * 1000, 2)

        timings["total"] = round((time.perf_counter() - t_total) * 1000, 2)
    except Exception:
        timings["total"] = round((time.perf_counter() - t_total) * 1000, 2)

    return timings
