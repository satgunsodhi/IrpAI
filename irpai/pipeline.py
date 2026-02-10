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

from irpai.config import IrpAIConfig, Tier
from irpai.context.assembler import ContextAssembler
from irpai.context.formatter import format_for_llm
from irpai.context.schema import ContextPack, PipelineMetadata
from irpai.detection.base import BaseDetector, DetectionOutput
from irpai.detection.yolo_detector import FallbackGridDetector
from irpai.layout.analyzer import LayoutAnalyzer, LayoutResult
from irpai.ocr.base import BaseOCR, OCRResult
from irpai.ocr.engine import create_ocr_engine
from irpai.preprocessing.normalizer import NormalizedImage, Normalizer
from irpai.research.helper import ResearchHelper, ResearchOutput
from irpai.vlm.base import BaseVLM, VLMResult
from irpai.utils.image import load_image
from irpai.utils.logging import get_logger

log = get_logger("pipeline")


class IrpAIPipeline:
    """
    Main IrpAI pipeline.

    Orchestrates the 5-layer processing chain::

        Input Image
            → Preprocessing (normalize, scale, denoise)
            → Detection (YOLO UI element detection)
            → OCR (PaddleOCR / Tesseract text extraction)
            → Layout Analysis (region grouping + hierarchy)
            → Context Assembly (structured JSON context pack)
            → [Optional] VLM captioning (Florence-2)
            → [Optional] Research helper (web search)

    Parameters
    ----------
    config:
        Pipeline configuration.  Use ``Tier.LITE``, ``Tier.STANDARD``,
        or ``Tier.FULL`` for quick presets.
    """

    def __init__(self, config: IrpAIConfig | None = None) -> None:
        self.config = config or IrpAIConfig()
        log.info("Initialising IrpAI pipeline (tier=%s)", self.config.tier.value)

        # Layers are initialised lazily — they load models on first use
        self._normalizer = Normalizer(self.config.preprocessing)
        self._ocr: BaseOCR | None = None
        self._detector: BaseDetector | None = None
        self._vlm: BaseVLM | None = None
        self._layout = LayoutAnalyzer(self.config.layout)
        self._assembler = ContextAssembler(self.config.output)
        self._research = ResearchHelper(self.config.research)

    # -- lazy layer init ----------------------------------------------------

    @property
    def ocr(self) -> BaseOCR:
        if self._ocr is None:
            self._ocr = create_ocr_engine(self.config.ocr)
        return self._ocr

    @property
    def detector(self) -> BaseDetector:
        if self._detector is None:
            if self.config.tier == Tier.LITE:
                self._detector = FallbackGridDetector()
            else:
                try:
                    import ultralytics as _  # noqa: F401 – availability check
                    from irpai.detection.yolo_detector import YOLODetector
                    self._detector = YOLODetector(self.config.detection)
                except ImportError:
                    log.warning("YOLO unavailable, using grid fallback")
                    self._detector = FallbackGridDetector()
        return self._detector

    @property
    def vlm(self) -> BaseVLM | None:
        if self.config.tier != Tier.FULL:
            return None
        if self._vlm is None:
            try:
                from irpai.vlm.florence import FlorenceVLM
                self._vlm = FlorenceVLM(self.config.vlm)
            except ImportError:
                log.warning("Florence-2 dependencies unavailable, skipping VLM")
                return None
        return self._vlm

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

        # -- Layer 1: Preprocessing -----------------------------------------
        log.info("▸ Layer 1: Preprocessing")
        normalized = self._normalizer.normalize(image)

        # -- Layer 2: Detection ---------------------------------------------
        log.info("▸ Layer 2: Detection")
        detections = self._run_detection(normalized)

        # -- Layer 3: OCR ---------------------------------------------------
        log.info("▸ Layer 3: OCR")
        ocr_result = self._run_ocr(normalized)

        # -- Layer 4: Layout Analysis ---------------------------------------
        log.info("▸ Layer 4: Layout Analysis")
        layout = self._layout.analyze(ocr_result, detections)

        # -- Layer 4b: VLM (full tier only) ---------------------------------
        vlm_result: VLMResult | None = None
        if self.vlm is not None:
            log.info("▸ Layer 4b: VLM Captioning")
            vlm_result = self._run_vlm(normalized)

        # -- Layer 5: Context Assembly --------------------------------------
        log.info("▸ Layer 5: Context Assembly")
        total_ms = (time.perf_counter() - t0) * 1000

        metadata = PipelineMetadata(
            irpai_version="0.1.0",
            tier=self.config.tier.value,
            ocr_engine=self.ocr.name,
            detection_model=self.detector.name,
            vlm_model=self.vlm.name if self.vlm else "none",
            total_ms=round(total_ms, 1),
            image_size=normalized.processed_size,
        )

        pack = self._assembler.assemble(
            layout=layout,
            ocr=ocr_result,
            detections=detections,
            vlm=vlm_result,
            metadata=metadata,
        )

        # -- Optional: Research helper --------------------------------------
        if self.config.research.enabled:
            log.info("▸ Research Helper")
            research = self._research.research(pack)
            if research.results:
                pack.summary.key_elements.append(
                    f"Research: {len(research.results)} snippets found"
                )

        log.info(
            "✓ Pipeline complete in %.0f ms — %d regions, ~%d tokens",
            total_ms,
            len(pack.regions),
            pack.token_estimate(),
        )

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
        """Run OCR text extraction."""
        try:
            return self.ocr.recognize(normalized.ocr_ready)
        except Exception:
            log.exception("OCR failed, using empty result")
            w, h = normalized.processed_size
            return OCRResult(image_size=(w, h))

    def _run_vlm(self, normalized: NormalizedImage) -> VLMResult | None:
        """Run VLM captioning (full tier only)."""
        if self.vlm is None:
            return None
        try:
            return self.vlm.run(normalized.color)
        except Exception:
            log.exception("VLM failed")
            return None

    # -- utilities ----------------------------------------------------------

    def warmup(self) -> None:
        """
        Pre-load all models.  Call this once at startup to avoid
        latency on the first :meth:`run` call.
        """
        log.info("Warming up pipeline (tier=%s)", self.config.tier.value)
        _ = self.ocr
        self.detector.warmup()
        if self.vlm is not None:
            self.vlm.warmup()
        log.info("Warmup complete")
