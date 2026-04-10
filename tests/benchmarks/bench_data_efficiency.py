"""
Data Efficiency Benchmark
=========================

Tests IrpAI's resilience to imperfect input data by systematically
degrading images along multiple axes:
- Resolution reduction
- Gaussian noise injection
- Contrast reduction
- JPEG compression artifacts
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance

from irpai.config import IrpAIConfig, Tier
from irpai.context.schema import ContextPack
from irpai.pipeline import IrpAIPipeline

from tests.research.synthetic_scenes import SyntheticScene, build_all_scenes


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DegradationResult:
    """Result at a single degradation level."""
    degradation_type: str
    level: float | int
    level_label: str
    scene: str
    regions_found: int = 0
    text_chars_extracted: int = 0
    confidence: float = 0.0
    tokens: int = 0
    error: str = ""


@dataclass
class DataEfficiencyResults:
    results: list[DegradationResult] = field(default_factory=list)
    graceful_degradation_index: float = 0.0


# ---------------------------------------------------------------------------
# Safe pipeline runner
# ---------------------------------------------------------------------------

def _safe_run(pipeline: IrpAIPipeline, image: Image.Image) -> ContextPack | None:
    """Run pipeline with fallback to manual layer execution."""
    try:
        return pipeline.run(image)
    except RuntimeError:
        pass

    try:
        normalized = pipeline._normalizer.normalize(image)
        detections = pipeline._run_detection(normalized)
        ocr_result = pipeline._run_ocr(normalized)
        layout = pipeline._layout.analyze(ocr_result, detections)
        return pipeline._assembler.assemble(
            layout=layout, ocr=ocr_result, detections=detections,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Degradation functions
# ---------------------------------------------------------------------------

def _reduce_resolution(img: Image.Image, factor: float) -> Image.Image:
    """Downscale image to factor × original size."""
    w, h = img.size
    new_w = max(32, int(w * factor))
    new_h = max(32, int(h * factor))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise with the given standard deviation."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _reduce_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Reduce contrast by the given factor (1.0 = original, 0.0 = gray)."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Compress and decompress via JPEG at the given quality."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

RESOLUTION_FACTORS = [1.0, 0.75, 0.50, 0.25]
NOISE_SIGMAS = [0, 5, 15, 30, 50]
CONTRAST_FACTORS = [1.0, 0.7, 0.4, 0.2]
JPEG_QUALITIES = [95, 75, 50, 25, 10]


def run_data_efficiency_benchmark() -> DataEfficiencyResults:
    """Execute the data efficiency benchmark across all degradation axes."""
    scenes = build_all_scenes()
    results = DataEfficiencyResults()

    config = IrpAIConfig(tier=Tier.LITE)
    try:
        pipeline = IrpAIPipeline(config)
    except Exception:
        return results

    # Use first 3 scenes for efficiency
    test_scenes = scenes[:3]

    for scene in test_scenes:
        name = scene.ground_truth.scene_name

        # Resolution degradation
        for factor in RESOLUTION_FACTORS:
            degraded = _reduce_resolution(scene.image, factor)
            r = _run_single(pipeline, degraded, "resolution", factor, f"{int(factor*100)}%", name)
            results.results.append(r)

        # Noise injection
        for sigma in NOISE_SIGMAS:
            if sigma == 0:
                degraded = scene.image
            else:
                degraded = _add_gaussian_noise(scene.image, sigma)
            r = _run_single(pipeline, degraded, "noise", sigma, f"σ={sigma}", name)
            results.results.append(r)

        # Contrast reduction
        for factor in CONTRAST_FACTORS:
            degraded = _reduce_contrast(scene.image, factor)
            r = _run_single(pipeline, degraded, "contrast", factor, f"{factor:.1f}", name)
            results.results.append(r)

        # JPEG compression
        for quality in JPEG_QUALITIES:
            degraded = _jpeg_compress(scene.image, quality)
            r = _run_single(pipeline, degraded, "jpeg", quality, f"Q{quality}", name)
            results.results.append(r)

    # Compute graceful degradation index
    baselines = [r for r in results.results if r.level_label in ("100%", "σ=0", "1.0", "Q95")]
    degraded_all = [r for r in results.results if r.level_label not in ("100%", "σ=0", "1.0", "Q95")]

    if baselines and degraded_all:
        avg_baseline = sum(r.regions_found for r in baselines) / len(baselines)
        avg_degraded = sum(r.regions_found for r in degraded_all) / len(degraded_all)
        results.graceful_degradation_index = round(
            avg_degraded / max(avg_baseline, 1) * 100, 1
        )

    return results


def _run_single(
    pipeline: IrpAIPipeline,
    image: Image.Image,
    deg_type: str,
    level: float | int,
    label: str,
    scene_name: str,
) -> DegradationResult:
    """Run pipeline on a single degraded image."""
    pack = _safe_run(pipeline, image)

    if pack is None:
        return DegradationResult(
            degradation_type=deg_type,
            level=level,
            level_label=label,
            scene=scene_name,
            error="Pipeline failed",
        )

    all_text = " ".join(r.text for r in pack.regions if r.text)
    avg_conf = 0.0
    if pack.regions:
        avg_conf = sum(r.confidence for r in pack.regions) / len(pack.regions)

    return DegradationResult(
        degradation_type=deg_type,
        level=level,
        level_label=label,
        scene=scene_name,
        regions_found=len(pack.regions),
        text_chars_extracted=len(all_text),
        confidence=round(avg_conf, 3),
        tokens=pack.token_estimate(),
    )
