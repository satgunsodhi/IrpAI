"""
Abstract base classes and result types for OCR engines.

Every OCR backend (PaddleOCR, Tesseract, …) implements :class:`BaseOCR`
so the pipeline can swap engines transparently.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass
class OCRWord:
    """A single recognised word/token with positional metadata."""

    text: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float = 1.0


@dataclass
class OCRLine:
    """A line of text — one or more contiguous :class:`OCRWord` s."""

    text: str
    bbox: tuple[int, int, int, int]
    words: list[OCRWord] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class OCRBlock:
    """
    A block of text — one or more :class:`OCRLine` s that are
    spatially grouped (e.g. a paragraph, a button label, a heading).
    """

    lines: list[OCRLine] = field(default_factory=list)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 1.0

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def estimated_font_size(self) -> float:
        """Median line height as a proxy for font size."""
        heights = [ln.height for ln in self.lines]
        if not heights:
            return 0.0
        heights.sort()
        mid = len(heights) // 2
        return float(heights[mid])


@dataclass
class OCRResult:
    """
    Complete OCR output for one image.

    Provides blocks → lines → words hierarchy, plus convenience
    accessors for the full text and aggregate confidence.
    """

    blocks: list[OCRBlock] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)  # (w, h)
    engine: str = "unknown"

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    @property
    def all_lines(self) -> list[OCRLine]:
        return [ln for b in self.blocks for ln in b.lines]

    @property
    def mean_confidence(self) -> float:
        confs = [ln.confidence for ln in self.all_lines]
        return sum(confs) / len(confs) if confs else 0.0


class BaseOCR(abc.ABC):
    """
    Abstract OCR engine interface.

    Subclasses must implement :meth:`recognize`.
    """

    @abc.abstractmethod
    def recognize(
        self,
        image: Image.Image | np.ndarray,
    ) -> OCRResult:
        """
        Run OCR on *image* and return structured results.

        Parameters
        ----------
        image:
            A preprocessed image (RGB PIL or BGR ndarray).

        Returns
        -------
        OCRResult
            Hierarchical text extraction results.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable engine name, e.g. ``'PaddleOCR'``."""
