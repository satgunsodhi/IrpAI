"""
EasyOCR engine implementation.

A permissively licensed (Apache 2.0) fallback engine that uses PyTorch.
It provides a good balance between accuracy and ease of installation
without requiring invasive licensing like Tesseract's `pytesseract` wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from irpai.ocr.base import BaseOCR, OCRBlock, OCRLine, OCRResult, OCRWord
from irpai.utils.image import pil_to_cv2
from irpai.utils.logging import get_logger

if TYPE_CHECKING:
    from irpai.config import OCRConfig

log = get_logger("ocr.easy_ocr")


class EasyOCREngine(BaseOCR):
    """
    EasyOCR-based text extraction engine.

    Requires `easyocr` and `torch` to be installed.
    """

    def __init__(self, config: OCRConfig) -> None:
        import easyocr  # type: ignore[import-untyped, import-not-found]
        self.config = config

        lang_map = {"en": "en", "ch": "ch_sim", "ja": "ja", "ko": "ko"}
        lang = lang_map.get(self.config.language, self.config.language)

        log.info("Loading EasyOCR models (lang=%s, gpu=False)", lang)
        self._reader = easyocr.Reader([lang], gpu=False)

    @property
    def name(self) -> str:
        return "easyocr"

    def recognize(self, image: Image.Image | np.ndarray) -> OCRResult:
        if isinstance(image, Image.Image):
            arr = pil_to_cv2(image)
        else:
            arr = image

        h, w = arr.shape[:2]
        raw_results = self._reader.readtext(arr)

        if not raw_results:
            return OCRResult(image_size=(w, h), engine=self.name)

        # EasyOCR returns a flat list of lines/words.
        # We will wrap each in an OCRWord, OCRLine, and OCRBlock
        # since EasyOCR doesn't do paragraph blocking natively.
        blocks: list[OCRBlock] = []

        for bbox_points, text, conf in raw_results:
            confidence = float(conf)
            if confidence < self.config.confidence_threshold:
                continue

            # bbox_points is a list of 4 points: [TL, TR, BR, BL]
            # Convert to [x1, y1, x2, y2]
            x_coords = [float(p[0]) for p in bbox_points]
            y_coords = [float(p[1]) for p in bbox_points]

            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))

            # Constrain to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            box = (x1, y1, x2, y2)
            word = OCRWord(text=text, bbox=box, confidence=confidence)
            line = OCRLine(text=text, bbox=box, words=[word], confidence=confidence)
            block = OCRBlock(lines=[line], bbox=box, confidence=confidence)

            blocks.append(block)

        # Basic vertical grouping into paragraphs
        blocks = self._group_blocks(blocks)

        return OCRResult(blocks=blocks, image_size=(w, h), engine=self.name)

    @staticmethod
    def _group_blocks(blocks: list[OCRBlock]) -> list[OCRBlock]:
        """Group vertically adjacent single-line blocks into paragraphs."""
        if not blocks:
            return []

        # Sort top-to-bottom
        sorted_blocks = sorted(blocks, key=lambda b: b.bbox[1])
        merged: list[OCRBlock] = []
        current_block = sorted_blocks[0]

        for next_block in sorted_blocks[1:]:
            curr_box = current_block.bbox
            next_box = next_block.bbox

            # Vertical distance
            y_dist = next_box[1] - curr_box[3]
            line_height = curr_box[3] - curr_box[1]

            # Horizontal overlap
            x_overlap = max(0, min(curr_box[2], next_box[2]) - max(curr_box[0], next_box[0]))
            min_width = min(curr_box[2] - curr_box[0], next_box[2] - next_box[0])

            # If they are close vertically and overlap horizontally, merge them
            if y_dist < line_height * 1.5 and x_overlap > min_width * 0.5:
                current_block.lines.extend(next_block.lines)
                current_block.confidence = (current_block.confidence + next_block.confidence) / 2
                current_block.bbox = (
                    min(curr_box[0], next_box[0]),
                    min(curr_box[1], next_box[1]),
                    max(curr_box[2], next_box[2]),
                    max(curr_box[3], next_box[3]),
                )
            else:
                merged.append(current_block)
                current_block = next_block

        merged.append(current_block)
        return merged
