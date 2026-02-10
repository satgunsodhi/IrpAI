"""
Abstract base class and result types for Vision-Language Models.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class VLMResult:
    """Output from a vision-language model inference."""

    caption: str = ""
    """Natural-language description of the image / region."""

    detailed_caption: str = ""
    """More detailed description (if the model supports it)."""

    objects: list[dict[str, object]] = field(default_factory=list)
    """Detected objects with labels and bounding boxes (if OD task was run)."""

    ocr_text: list[dict[str, object]] = field(default_factory=list)
    """OCR results from the VLM (if OCR task was run)."""

    raw_outputs: dict[str, object] = field(default_factory=dict)
    """Raw model outputs keyed by task prompt."""

    model_name: str = "unknown"
    inference_ms: float = 0.0


class BaseVLM(abc.ABC):
    """
    Abstract Vision-Language Model interface.

    Subclasses must implement :meth:`run`.
    """

    @abc.abstractmethod
    def run(
        self,
        image: Image.Image | np.ndarray,
        tasks: tuple[str, ...] | None = None,
    ) -> VLMResult:
        """
        Run one or more vision-language tasks on *image*.

        Parameters
        ----------
        image:
            Preprocessed colour image.
        tasks:
            Task prompts to run (model-specific).  ``None`` = use defaults.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    def warmup(self) -> None:
        """Optional warmup (override in subclass)."""
