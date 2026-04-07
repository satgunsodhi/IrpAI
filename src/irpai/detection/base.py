"""
Abstract base classes and result types for UI element detectors.

Every detection backend (RT-DETR, rule fallback, …) implements
:class:`BaseDetector` so the pipeline can swap models transparently.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    """A single detected UI element."""

    label: str
    """Semantic class label, e.g. 'button', 'text_field', 'icon'."""

    bbox: tuple[int, int, int, int]
    """Bounding box (x1, y1, x2, y2) in pixel coordinates."""

    confidence: float = 1.0
    """Model confidence ∈ [0, 1]."""

    meta: dict[str, object] = field(default_factory=dict)
    """Optional extra metadata (e.g. icon caption from VLM)."""

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


@dataclass
class DetectionOutput:
    """
    Complete detection output for one image.

    Contains all detected regions plus image-level metadata.
    """

    detections: list[DetectionResult] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)  # (w, h)
    model_name: str = "unknown"
    inference_ms: float = 0.0

    def filter_by_confidence(self, threshold: float) -> list[DetectionResult]:
        return [d for d in self.detections if d.confidence >= threshold]

    def filter_by_label(self, *labels: str) -> list[DetectionResult]:
        label_set = set(labels)
        return [d for d in self.detections if d.label in label_set]

    def sorted_by_position(self) -> list[DetectionResult]:
        """Return detections sorted top-to-bottom, left-to-right."""
        return sorted(self.detections, key=lambda d: (d.bbox[1], d.bbox[0]))


class BaseDetector(abc.ABC):
    """
    Abstract UI element detector interface.

    Subclasses must implement :meth:`detect`.
    """

    @abc.abstractmethod
    def detect(
        self,
        image: Image.Image | np.ndarray,
    ) -> DetectionOutput:
        """
        Detect UI elements in *image*.

        Parameters
        ----------
        image:
            Preprocessed colour image.

        Returns
        -------
        DetectionOutput
            All detected UI regions with labels and bounding boxes.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""

    def warmup(self) -> None:
        """Optional model warmup / preload (override in subclass)."""
