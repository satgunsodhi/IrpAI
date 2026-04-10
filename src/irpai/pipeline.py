"""
IrpAI Pipeline Orchestrator

The central class that wires all layers together and executes the
full screenshot → context-pack pipeline.

Usage::

    from irpai import IrpAIPipeline, IrpAIConfig, Tier

    pipeline = IrpAIPipeline(IrpAIConfig(tier=Tier.STANDARD))
    pack = pipeline.run("screenshot.png")
    print(pack.to_json())
"""


from __future__ import annotations


import time
from pathlib import Path

import numpy as np
from PIL import Image

from irpai.config import IrpAIConfig
from irpai.context.assembler import ContextAssembler
from irpai.context.formatter import format_for_llm
from irpai.context.schema import ContextPack, PipelineMetadata
from irpai.detection.base import BaseDetector, DetectionOutput
from irpai.detection.hybrid_detector import HybridDetector
from irpai.layout.analyzer import LayoutAnalyzer, LayoutResult
from irpai.ocr.base import BaseOCR, OCRResult
from irpai.ocr.engine import create_ocr_engine
from irpai.preprocessing.normalizer import NormalizedImage, Normalizer
from irpai.utils.image import load_image
from irpai.utils.logging import get_logger

log = get_logger("pipeline")


class IrpAIPipeline:
    """
    Main IrpAI pipeline.

    Orchestrates the processing chain::

        Input Image
            → Preprocessing (normalize, scale, denoise)
            → Detection (RT-DETR primary + rule fallback)
            → OCR (EasyOCR text extraction)
            → Layout Analysis (region grouping + hierarchy)
            → Context Assembly (structured JSON context pack)

    Parameters
    ----------
    config:
        Pipeline configuration.  Use ``Tier.LITE`` or ``Tier.STANDARD``
        for quick presets.
    """

    def __init__(self, config: IrpAIConfig | None = None) -> None:
        self.config = config or IrpAIConfig()
        log.info("Initialising IrpAI pipeline (tier=%s)", self.config.tier.value)

        # Layers are initialised lazily — they load models on first use
        self._normalizer = Normalizer(self.config.preprocessing)
        self._ocr: BaseOCR | None = None
        self._detector: BaseDetector = HybridDetector(
            use_ml=self.config.detection.use_ml,
            ml_model_id=self.config.detection.model_id,
            confidence_threshold=self.config.detection.confidence_threshold,
            structural_boost=self.config.detection.structural_boost,
            nms_iou_threshold=self.config.detection.nms_iou_threshold,
            max_detections=self.config.detection.max_detections,
            device=self.config.detection.device,
        )
        self._layout = LayoutAnalyzer(self.config.layout)
        self._assembler = ContextAssembler(self.config.output)

    # -- lazy layer init ----------------------------------------------------

    @property
    def ocr(self) -> BaseOCR:
        if self._ocr is None:
            self._ocr = create_ocr_engine(self.config.ocr)
        return self._ocr

    @property
    def detector(self) -> BaseDetector:
        return self._detector

    # -- main entry point ---------------------------------------------------

    def run(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        output_format: str = "json",
    ) -> ContextPack:
        """
        Run the full pipeline on an image.

        Parameters
        ----------
        image:
            Path to screenshot, PIL Image, or NumPy array.
        output_format:
            Desired output format hint (``'json'``, ``'markdown'``).

        Returns
        -------
        ContextPack
            Structured context ready for LLM consumption.
        """
        t0 = time.perf_counter()
        stage_timings: dict[str, float] = {}
        warnings: list[str] = []

        # -- Layer 1: Preprocessing -----------------------------------------
        log.info("▸ Layer 1: Preprocessing")
        t_stage = time.perf_counter()
        normalized = self._normalizer.normalize(str(image) if isinstance(image, Path) else image)
        stage_timings["preprocessing"] = round((time.perf_counter() - t_stage) * 1000, 1)

        # Low-resolution warning
        quality = normalized.quality_metrics
        if quality.get("is_low_resolution"):
            warnings.append(
                f"Low resolution image ({quality.get('resolution')}). "
                "Results may be less accurate."
            )

        # -- Layer 2: Detection ---------------------------------------------
        log.info("▸ Layer 2: Detection")
        t_stage = time.perf_counter()
        detections = self._run_detection(normalized)
        stage_timings["detection"] = round((time.perf_counter() - t_stage) * 1000, 1)

        # -- Layer 3: OCR ---------------------------------------------------
        log.info("▸ Layer 3: OCR")
        t_stage = time.perf_counter()
        ocr_result = self._run_ocr(normalized)
        stage_timings["ocr"] = round((time.perf_counter() - t_stage) * 1000, 1)

        # -- Layer 4: Layout Analysis ---------------------------------------
        log.info("▸ Layer 4: Layout Analysis")
        t_stage = time.perf_counter()
        layout = self._layout.analyze(ocr_result, detections)
        stage_timings["layout"] = round((time.perf_counter() - t_stage) * 1000, 1)

        # -- Layer 5: Context Assembly --------------------------------------
        log.info("▸ Layer 5: Context Assembly")
        t_stage = time.perf_counter()
        total_ms = (time.perf_counter() - t0) * 1000

        confidence = self._compute_confidence(detections, ocr_result)

        metadata = PipelineMetadata(
            irpai_version="0.1.0",
            tier=self.config.tier.value,
            ocr_engine=self.ocr.name,
            detection_model=self.detector.name,
            total_ms=round(total_ms, 1),
            image_size=normalized.processed_size,
        )

        pack = self._assembler.assemble(
            layout=layout,
            ocr=ocr_result,
            detections=detections,
            metadata=metadata,
        )
        stage_timings["assembly"] = round((time.perf_counter() - t_stage) * 1000, 1)

        log.info(
            "✓ Pipeline complete in %.0f ms — %d regions, "
            "confidence=%.2f, ~%d tokens  [%s]",
            total_ms,
            len(pack.regions),
            confidence,
            pack.token_estimate(),
            ", ".join(f"{k}={v:.0f}ms" for k, v in stage_timings.items()),
        )
        if warnings:
            for w in warnings:
                log.warning("⚠ %s", w)

        return pack


    def run_formatted(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        style: str = "json",
    ) -> str:
        """
        Run the pipeline and return the formatted string output.

        This is a convenience wrapper around :meth:`run` + :func:`format_for_llm`.
        """
        pack = self.run(image)
        return format_for_llm(pack, style=style)

    # -- layer runners ------------------------------------------------------

    def _run_detection(self, normalized: NormalizedImage) -> DetectionOutput:
        """Run UI element detection."""
        try:
            return self.detector.detect(normalized.color)
        except Exception:
            log.exception("Detection failed, using empty result")
            w, h = normalized.processed_size
            return DetectionOutput(image_size=(w, h))

    def _run_ocr(self, normalized: NormalizedImage) -> OCRResult:
        try:
            return self.ocr.recognize(normalized.ocr_ready)
        except Exception:
            log.exception("OCR failed")
            w, h = normalized.processed_size
            return OCRResult(image_size=(w, h))

    # -- utilities ----------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        detections: DetectionOutput, ocr_result: OCRResult,
    ) -> float:
        """
        Compute an overall pipeline confidence score (0.0–1.0).

        Aggregates detection confidence, OCR confidence, and applies
        a penalty for very few detections.
        """
        scores: list[float] = []

        # Detection confidence
        if detections.detections:
            avg_det = sum(d.confidence for d in detections.detections) / len(
                detections.detections
            )
            scores.append(avg_det)

        # OCR confidence
        if ocr_result.all_lines:
            scores.append(ocr_result.mean_confidence)

        # Penalty for sparse detections
        total = len(detections.detections)
        if total < 3:
            scores.append(0.3)
        elif total < 10:
            scores.append(0.6)
        else:
            scores.append(0.9)

        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def warmup(self) -> None:
        """
        Pre-load all models.  Call this once at startup to avoid
        latency on the first :meth:`run` call.
        """
        log.info("Warming up pipeline (tier=%s)", self.config.tier.value)
        _ = self.ocr
        log.info("Warmup complete")

