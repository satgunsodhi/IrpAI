"""
IrpAI Pipeline Orchestrator

Main pipeline class that coordinates all stages:
1. Preprocessing → normalized image
2. Vision Branch → UI element detections
3. Text Branch → OCR text extractions
4. Fusion Module → structured context
5. LLM Interface → enhanced prompt & response

Supports:
- Synchronous and asynchronous execution
- Smart caching of processed screenshots
- Parallel vision/text branch processing
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from irpai.config import PipelineConfig
from irpai.fusion import FusionModule
from irpai.llm_interface import LLMInterface, PromptBuilder
from irpai.models import PipelineResult, StructuredContext, UIDetection, TextExtraction
from irpai.ocr import OCRExtractor, PIIRedactor
from irpai.preprocessing import Preprocessor
from irpai.vision import UIDetector

logger = logging.getLogger(__name__)


class IrpAIPipeline:
    """
    Main IrpAI pipeline orchestrator.

    Coordinates all processing stages and manages caching,
    parallel execution, and error handling.

    Usage:
        pipeline = IrpAIPipeline()
        result = pipeline.analyze("screenshot.png", "What error is shown?")
        print(result.llm_response)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._setup_logging()

        # Initialize pipeline stages (lazy-loaded where possible)
        self.preprocessor = Preprocessor(self.config.preprocessing)
        self.ui_detector = UIDetector(self.config.vision, self.config.resolved_device)
        self.ocr_extractor = OCRExtractor(self.config.ocr)
        self.fusion_module = FusionModule(self.config.fusion)
        self.prompt_builder = PromptBuilder(self.config.llm)
        self.llm_interface = LLMInterface(self.config.llm)

        # Cache for processed screenshots
        self._context_cache: dict[str, StructuredContext] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info(
            f"IrpAI Pipeline initialized (device={self.config.resolved_device}, "
            f"async={self.config.enable_async})"
        )

    def _setup_logging(self):
        """Configure logging based on config."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # ── Public API ──────────────────────────────────────────────

    def analyze(
        self,
        image: str | Path | np.ndarray,
        query: Optional[str] = None,
        use_llm: bool = True,
    ) -> PipelineResult:
        """
        Analyze a screenshot and optionally answer a query.

        This is the main entry point for the pipeline.

        Args:
            image: Screenshot path, Path object, or numpy array (BGR)
            query: Optional user question about the screenshot
            use_llm: Whether to generate LLM response (requires API key)

        Returns:
            PipelineResult with structured context and optional LLM response
        """
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        warnings: list[str] = []

        # ── Stage 1: Preprocessing ──
        t0 = time.perf_counter()
        try:
            preprocessed = self.preprocessor.process(image)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise RuntimeError(f"Preprocessing failed: {e}") from e
        stage_timings["preprocessing"] = (time.perf_counter() - t0) * 1000

        # Check image quality
        quality = Preprocessor.compute_quality_metrics(preprocessed)
        if quality["is_low_resolution"]:
            warnings.append(
                f"Low resolution image ({quality['resolution']}). "
                "Results may be less accurate."
            )

        # ── Check cache ──
        img_hash = self._compute_hash(preprocessed)
        if self.config.enable_caching and img_hash in self._context_cache:
            logger.info("Using cached context")
            structured_context = self._context_cache[img_hash]
            stage_timings["cache_hit"] = 0.0
        else:
            # ── Stages 2-4: Detection, OCR, Fusion ──
            structured_context = self._process_image(
                preprocessed, stage_timings, warnings
            )

            # Cache the result
            if self.config.enable_caching:
                self._cache_context(img_hash, structured_context)

        # ── Stage 5: LLM Interface ──
        enhanced_prompt = None
        llm_response = None

        if query:
            t0 = time.perf_counter()
            enhanced_prompt = self.prompt_builder.build_prompt(query, structured_context)
            stage_timings["prompt_building"] = (time.perf_counter() - t0) * 1000

            if use_llm:
                t0 = time.perf_counter()
                try:
                    _, llm_response = self.llm_interface.generate_response(
                        query, structured_context
                    )
                except Exception as e:
                    logger.warning(f"LLM call failed: {e}")
                    warnings.append(f"LLM unavailable: {e}")
                    llm_response = self.llm_interface._generate_fallback_response(
                        query, structured_context
                    )
                stage_timings["llm_inference"] = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        # Compute overall confidence
        confidence = self._compute_confidence(structured_context)

        result = PipelineResult(
            structured_context=structured_context,
            enhanced_prompt=enhanced_prompt,
            llm_response=llm_response,
            processing_time_ms=round(total_ms, 1),
            stage_timings={k: round(v, 1) for k, v in stage_timings.items()},
            confidence=round(confidence, 3),
            warnings=warnings,
        )

        logger.info(
            f"Pipeline complete: {total_ms:.0f}ms total, "
            f"{len(structured_context.ui_detections)} elements, "
            f"confidence={confidence:.2f}"
        )

        return result

    def extract_context(
        self, image: str | Path | np.ndarray
    ) -> StructuredContext:
        """
        Extract structured context without LLM interaction.

        Useful for batch processing or when you want to build
        custom prompts.

        Args:
            image: Screenshot path or numpy array

        Returns:
            StructuredContext with full screenshot analysis
        """
        result = self.analyze(image, query=None, use_llm=False)
        return result.structured_context

    async def analyze_async(
        self,
        image: str | Path | np.ndarray,
        query: Optional[str] = None,
        use_llm: bool = True,
    ) -> PipelineResult:
        """
        Async version of analyze() with parallel vision/text branches.

        Args:
            image: Screenshot path or numpy array
            query: Optional user question
            use_llm: Whether to call LLM

        Returns:
            PipelineResult
        """
        total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        warnings: list[str] = []

        # Preprocessing (sync, fast)
        t0 = time.perf_counter()
        preprocessed = await asyncio.to_thread(self.preprocessor.process, image)
        stage_timings["preprocessing"] = (time.perf_counter() - t0) * 1000

        # Check cache
        img_hash = self._compute_hash(preprocessed)
        if self.config.enable_caching and img_hash in self._context_cache:
            structured_context = self._context_cache[img_hash]
        else:
            # Run vision and OCR in parallel
            t0 = time.perf_counter()
            vision_task = asyncio.to_thread(self.ui_detector.detect, preprocessed)
            ocr_task = asyncio.to_thread(self.ocr_extractor.extract, preprocessed)

            ui_detections, text_extractions = await asyncio.gather(
                vision_task, ocr_task
            )
            stage_timings["parallel_detection"] = (time.perf_counter() - t0) * 1000

            # PII redaction
            if self.config.llm.enable_pii_redaction:
                text_extractions = PIIRedactor.redact(text_extractions)

            # Fusion (sequential, needs both inputs)
            t0 = time.perf_counter()
            structured_context = self.fusion_module.fuse(
                ui_detections, text_extractions, preprocessed.shape
            )
            stage_timings["fusion"] = (time.perf_counter() - t0) * 1000

            if self.config.enable_caching:
                self._cache_context(img_hash, structured_context)

        # LLM (if requested)
        enhanced_prompt = None
        llm_response = None

        if query:
            enhanced_prompt = self.prompt_builder.build_prompt(query, structured_context)
            if use_llm:
                t0 = time.perf_counter()
                try:
                    _, llm_response = await asyncio.to_thread(
                        self.llm_interface.generate_response,
                        query,
                        structured_context,
                    )
                except Exception as e:
                    warnings.append(f"LLM unavailable: {e}")
                    llm_response = self.llm_interface._generate_fallback_response(
                        query, structured_context
                    )
                stage_timings["llm_inference"] = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        return PipelineResult(
            structured_context=structured_context,
            enhanced_prompt=enhanced_prompt,
            llm_response=llm_response,
            processing_time_ms=round(total_ms, 1),
            stage_timings={k: round(v, 1) for k, v in stage_timings.items()},
            confidence=round(self._compute_confidence(structured_context), 3),
            warnings=warnings,
        )

    # ── Internal Methods ────────────────────────────────────────

    def _process_image(
        self,
        preprocessed: np.ndarray,
        stage_timings: dict[str, float],
        warnings: list[str],
    ) -> StructuredContext:
        """Run detection, OCR, and fusion stages."""

        # ── Stage 2: Vision Branch ──
        t0 = time.perf_counter()
        try:
            ui_detections = self.ui_detector.detect(preprocessed)
        except Exception as e:
            logger.warning(f"UI detection failed: {e}. Continuing with OCR only.")
            warnings.append(f"UI detection failed: {e}")
            ui_detections = []
        stage_timings["ui_detection"] = (time.perf_counter() - t0) * 1000

        # ── Stage 3: Text Branch ──
        t0 = time.perf_counter()
        try:
            if self.config.ocr.enable_ensemble:
                text_extractions = self.ocr_extractor.extract_ensemble(
                    preprocessed, ui_detections
                )
            else:
                text_extractions = self.ocr_extractor.extract(
                    preprocessed, ui_detections if ui_detections else None
                )
        except Exception as e:
            logger.warning(f"OCR failed: {e}. Continuing without text.")
            warnings.append(f"OCR failed: {e}")
            text_extractions = []
        stage_timings["ocr"] = (time.perf_counter() - t0) * 1000

        # PII redaction
        if self.config.llm.enable_pii_redaction and text_extractions:
            text_extractions = PIIRedactor.redact(text_extractions)

        # ── Stage 4: Fusion ──
        t0 = time.perf_counter()
        structured_context = self.fusion_module.fuse(
            ui_detections, text_extractions, preprocessed.shape
        )
        stage_timings["fusion"] = (time.perf_counter() - t0) * 1000

        return structured_context

    def _compute_hash(self, img: np.ndarray) -> str:
        """Compute hash of image for caching."""
        return hashlib.md5(img.tobytes()[:10000]).hexdigest()

    def _cache_context(self, img_hash: str, context: StructuredContext):
        """Add context to cache, evicting oldest if full."""
        if len(self._context_cache) >= self.config.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._context_cache))
            del self._context_cache[oldest_key]

        self._context_cache[img_hash] = context

    def _compute_confidence(self, context: StructuredContext) -> float:
        """Compute overall pipeline confidence score."""
        scores = []

        # UI detection confidence
        if context.ui_detections:
            avg_det = sum(d.confidence for d in context.ui_detections) / len(
                context.ui_detections
            )
            scores.append(avg_det)

        # OCR confidence
        if context.text_extractions:
            avg_ocr = sum(e.confidence for e in context.text_extractions) / len(
                context.text_extractions
            )
            scores.append(avg_ocr)

        # Penalty for few detections
        total_elements = len(context.ui_detections)
        if total_elements < 3:
            scores.append(0.3)
        elif total_elements < 10:
            scores.append(0.6)
        else:
            scores.append(0.9)

        return sum(scores) / len(scores) if scores else 0.0

    def visualize(
        self,
        image: str | Path | np.ndarray,
        context: Optional[StructuredContext] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize detections and OCR results on the image.

        Draws bounding boxes and labels on the screenshot.

        Args:
            image: Original screenshot
            context: Pre-computed context (or will process image)
            output_path: Optional path to save visualization

        Returns:
            Annotated image as numpy array
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()

        # Get context if not provided
        if context is None:
            context = self.extract_context(image)

        # Color map for element types
        colors = {
            "button": (0, 255, 0),      # Green
            "text_input": (255, 128, 0), # Orange
            "text_block": (0, 128, 255), # Blue
            "image": (255, 0, 255),      # Magenta
            "menu": (255, 255, 0),       # Cyan
            "modal": (0, 0, 255),        # Red
            "container": (128, 128, 128),# Gray
            "icon": (255, 0, 128),       # Pink
        }
        default_color = (200, 200, 200)

        # Draw UI detections
        for det in context.ui_detections:
            bbox = det.bbox
            color = colors.get(det.element_type.value, default_color)

            # Draw rectangle
            cv2.rectangle(
                img,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                2,
            )

            # Draw label
            label = f"{det.element_type.value}"
            if det.text:
                label += f": {det.text[:20]}"
            label += f" ({det.confidence:.2f})"

            label_y = max(int(bbox.y1) - 5, 15)
            cv2.putText(
                img,
                label,
                (int(bbox.x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, img)
            logger.info(f"Visualization saved to {output_path}")

        return img

    def clear_cache(self):
        """Clear the context cache."""
        self._context_cache.clear()
        logger.info("Context cache cleared")
