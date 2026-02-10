"""
Tesseract OCR backend — lightweight fallback for the *lite* tier.

Uses ``pytesseract`` (a thin wrapper around the Tesseract 5.x CLI)
to extract text with positional metadata.  Runs entirely on CPU and
requires the ``tesseract`` binary to be installed on the system.

Install
-------
    brew install tesseract        # macOS
    sudo apt install tesseract-ocr  # Linux
    pip install pytesseract
"""

from __future__ import annotations

from typing import Any
from statistics import median

import numpy as np
from PIL import Image

from irpai.config import OCRConfig
from irpai.ocr.base import BaseOCR, OCRBlock, OCRLine, OCRResult, OCRWord
from irpai.utils.logging import get_logger

log = get_logger("ocr.tesseract")

# Tesseract uses ISO 639-3 codes (eng, fra, deu …)
_LANG_MAP: dict[str, str] = {
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
}


class TesseractOCREngine(BaseOCR):
    """
    Tesseract OCR wrapper implementing the :class:`BaseOCR` interface.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.cfg = config or OCRConfig(engine="tesseract")

    @property
    def name(self) -> str:
        return "Tesseract"

    def recognize(self, image: Image.Image | np.ndarray) -> OCRResult:
        try:
            import pytesseract  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pytesseract is not installed.  Run: pip install pytesseract  "
                "and ensure the tesseract binary is on your PATH."
            ) from exc

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        w, h = image.size
        lang = _LANG_MAP.get(self.cfg.language, self.cfg.language)

        # PSM  6 = assume a single uniform block of text (good per-region)
        # PSM 11 = sparse text — find as much text as possible (better for
        #          full-screen screenshots with varied layout regions)
        # OEM  3 = LSTM + legacy combined (best accuracy)
        psm = 11 if h > w * 1.2 else 6
        tess_config = f"--oem 3 --psm {psm}"

        # DPI hint: if the image lacks DPI metadata Tesseract assumes 72
        # which makes small text unreadable.  Force a sane value.
        if hasattr(image, "info") and "dpi" not in image.info:
            tess_config += " --dpi 150"

        # Get word-level bounding boxes via Tesseract TSV output
        tsv: dict[str, list[Any]] = pytesseract.image_to_data(
            image,
            lang=lang,
            config=tess_config,
            output_type=pytesseract.Output.DICT,
        )

        # Parse TSV into our hierarchy
        lines_by_block: dict[tuple[int, int, int], list[OCRWord]] = {}
        _MIN_CONF = 25  # discard noise below this confidence

        n = len(tsv["text"])
        for i in range(n):
            text = str(tsv["text"][i]).strip()
            conf = float(tsv["conf"][i])
            if not text or conf < _MIN_CONF:
                continue

            block_num = int(tsv["block_num"][i])
            par_num = int(tsv["par_num"][i])
            line_num = int(tsv["line_num"][i])

            x = int(tsv["left"][i])
            y = int(tsv["top"][i])
            bw = int(tsv["width"][i])
            bh = int(tsv["height"][i])
            bbox = (x, y, x + bw, y + bh)

            key = (block_num, par_num, line_num)
            lines_by_block.setdefault(key, []).append(
                OCRWord(text=text, bbox=bbox, confidence=conf / 100.0)
            )

        # Build lines → blocks
        block_groups: dict[int, list[OCRLine]] = {}
        for (block_num, _par, _line), words in sorted(lines_by_block.items()):
            if not words:
                continue
            line_text = " ".join(w.text for w in words)
            x1 = min(w.bbox[0] for w in words)
            y1 = min(w.bbox[1] for w in words)
            x2 = max(w.bbox[2] for w in words)
            y2 = max(w.bbox[3] for w in words)
            avg_conf = sum(w.confidence for w in words) / len(words)

            line = OCRLine(
                text=line_text,
                bbox=(x1, y1, x2, y2),
                words=words,
                confidence=avg_conf,
            )
            block_groups.setdefault(block_num, []).append(line)

        blocks: list[OCRBlock] = []
        for _bnum, blines in sorted(block_groups.items()):
            # Split a block into sub-blocks if there are large vertical gaps
            # between lines (common in mobile UIs).
            if len(blines) > 1:
                blines = sorted(blines, key=lambda ln: ln.bbox[1])
                heights = [ln.height for ln in blines if ln.height > 0]
                gap_threshold = (median(heights) * 1.8) if heights else 20.0
                groups: list[list[OCRLine]] = [[]]
                for ln in blines:
                    if not groups[-1]:
                        groups[-1].append(ln)
                        continue
                    prev = groups[-1][-1]
                    gap = ln.bbox[1] - prev.bbox[3]
                    if gap > gap_threshold:
                        groups.append([ln])
                    else:
                        groups[-1].append(ln)
            else:
                groups = [blines]

            for group in groups:
                if not group:
                    continue
                x1 = min(ln.bbox[0] for ln in group)
                y1 = min(ln.bbox[1] for ln in group)
                x2 = max(ln.bbox[2] for ln in group)
                y2 = max(ln.bbox[3] for ln in group)
                avg_conf = sum(ln.confidence for ln in group) / len(group)
                blocks.append(OCRBlock(lines=group, bbox=(x1, y1, x2, y2), confidence=avg_conf))

        result = OCRResult(blocks=blocks, image_size=(w, h), engine=self.name)
        log.info(
            "Tesseract: %d blocks, %d lines, mean_conf=%.2f",
            len(result.blocks),
            len(result.all_lines),
            result.mean_confidence,
        )
        return result
