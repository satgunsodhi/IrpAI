"""
PaddleOCR backend — primary OCR engine for the *standard* and *full* tiers.

Uses PaddlePaddle's PP-OCRv5 pipeline (detection → recognition) which
runs comfortably on CPU (~10-15 M params total) and optionally on GPU.

Install
-------
    pip install paddlepaddle paddleocr
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from irpai.config import OCRConfig
from irpai.ocr.base import BaseOCR, OCRBlock, OCRLine, OCRResult, OCRWord
from irpai.utils.image import pil_to_cv2
from irpai.utils.logging import get_logger

log = get_logger("ocr.paddle")


class PaddleOCREngine(BaseOCR):
    """
    PaddleOCR wrapper implementing the :class:`BaseOCR` interface.

    Lazily initialises the PaddleOCR model on first call to :meth:`recognize`.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.cfg = config or OCRConfig(engine="paddleocr")
        self._engine: Any = None

    @property
    def name(self) -> str:
        return "PaddleOCR"

    # -- lazy init ----------------------------------------------------------

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed.  Run: pip install paddlepaddle paddleocr"
            ) from exc

        log.info("Initialising PaddleOCR (lang=%s, gpu=%s)", self.cfg.language, self.cfg.use_gpu)

        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=self.cfg.language,
            use_gpu=self.cfg.use_gpu,
            det_db_thresh=self.cfg.det_db_thresh,
            rec_batch_num=self.cfg.rec_batch_num,
            show_log=False,
        )
        return self._engine

    # -- recognition --------------------------------------------------------

    def recognize(self, image: Image.Image | np.ndarray) -> OCRResult:
        engine = self._ensure_engine()

        # PaddleOCR expects a BGR ndarray or file path
        if isinstance(image, Image.Image):
            arr = pil_to_cv2(image)
        else:
            arr = image

        h, w = arr.shape[:2]

        raw: list[Any] = engine.ocr(arr, cls=True)

        if not raw or raw[0] is None:
            log.debug("PaddleOCR returned no results")
            return OCRResult(image_size=(w, h), engine=self.name)

        # raw[0] is a list of [polygon, (text, confidence)]
        blocks: list[OCRBlock] = []
        lines: list[OCRLine] = []

        for item in raw[0]:
            polygon, (text, conf) = item
            # polygon is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

            word = OCRWord(text=text, bbox=bbox, confidence=float(conf))
            line = OCRLine(
                text=text,
                bbox=bbox,
                words=[word],
                confidence=float(conf),
            )
            lines.append(line)

        # Group lines into blocks by vertical proximity
        if self.cfg.paragraph_merge and lines:
            blocks = self._merge_into_blocks(lines)
        else:
            blocks = [
                OCRBlock(lines=[ln], bbox=ln.bbox, confidence=ln.confidence)
                for ln in lines
            ]

        result = OCRResult(blocks=blocks, image_size=(w, h), engine=self.name)
        log.info(
            "PaddleOCR: %d blocks, %d lines, mean_conf=%.2f",
            len(result.blocks),
            len(result.all_lines),
            result.mean_confidence,
        )
        return result

    # -- paragraph merging --------------------------------------------------

    @staticmethod
    def _merge_into_blocks(
        lines: list[OCRLine],
        y_gap: int = 15,
    ) -> list[OCRBlock]:
        """
        Merge OCR lines into paragraph-level blocks based on vertical
        proximity (lines within *y_gap* pixels of each other).
        """
        if not lines:
            return []

        # Sort top-to-bottom, then left-to-right
        lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))

        blocks: list[OCRBlock] = []
        current_lines: list[OCRLine] = [lines[0]]

        for ln in lines[1:]:
            prev_bottom = current_lines[-1].bbox[3]
            curr_top = ln.bbox[1]

            if curr_top - prev_bottom <= y_gap:
                current_lines.append(ln)
            else:
                blocks.append(_lines_to_block(current_lines))
                current_lines = [ln]

        blocks.append(_lines_to_block(current_lines))
        return blocks


def _lines_to_block(lines: list[OCRLine]) -> OCRBlock:
    """Convert a group of lines into a single OCRBlock."""
    x1 = min(ln.bbox[0] for ln in lines)
    y1 = min(ln.bbox[1] for ln in lines)
    x2 = max(ln.bbox[2] for ln in lines)
    y2 = max(ln.bbox[3] for ln in lines)
    avg_conf = sum(ln.confidence for ln in lines) / len(lines)
    return OCRBlock(lines=list(lines), bbox=(x1, y1, x2, y2), confidence=avg_conf)
