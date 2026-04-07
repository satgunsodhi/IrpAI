"""
IrpAI Configuration

Defines the tiered execution model and all configurable parameters
for each pipeline layer.

Tiers
-----
- **lite**     : CPU-only, Tesseract OCR, rule-based layout.  Minimal deps.
- **standard** : CPU, RT-DETR + fallback detection, OCR, layout analysis.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


class Tier(str, enum.Enum):
    """Execution tier — controls which models/backends are loaded."""

    LITE = "lite"
    STANDARD = "standard"


DEFAULT_RTDETR_HF_MODEL_ID = "PekingU/rtdetr_r18vd_coco_o365"
LOCAL_RTDETR_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "rtdetr-ui-mobile-mixed-complete" / "checkpoint-best"
)


def _default_detection_model_id() -> str:
    if LOCAL_RTDETR_CHECKPOINT.exists():
        return str(LOCAL_RTDETR_CHECKPOINT)
    return DEFAULT_RTDETR_HF_MODEL_ID


# ---------------------------------------------------------------------------
# Per-layer configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessingConfig:
    """Image preprocessing settings."""

    target_max_dimension: int = 1920
    """Scale images so the longest side does not exceed this value."""

    denoise_strength: int = 3
    """Non-local means denoising strength (0 = disabled)."""

    adaptive_threshold: bool = False
    """Apply adaptive threshold for very low-contrast screenshots."""

    grayscale_ocr: bool = True
    """Convert to grayscale before OCR pass (improves speed & accuracy)."""

    sharpen: bool = True
    """Apply mild unsharp-mask sharpening after resize."""


@dataclass(frozen=True)
class OCRConfig:
    """OCR engine settings."""

    engine: Literal["easyocr", "paddleocr", "tesseract"] = "easyocr"
    """Primary OCR backend."""

    fallback_engine: Literal["tesseract", "none"] = "none"
    """Fallback if primary fails or is unavailable."""

    language: str = "en"
    """Language code(s) for OCR."""

    confidence_threshold: float = 0.5
    """Minimum confidence threshold for valid text extraction."""

    use_gpu: bool = False
    """Use GPU acceleration for OCR (PaddleOCR only)."""

    det_db_thresh: float = 0.3
    """PaddleOCR text detection threshold."""

    rec_batch_num: int = 6
    """PaddleOCR recognition batch size."""

    paragraph_merge: bool = True
    """Attempt to merge lines into paragraphs by proximity."""

    tesseract_path: str | None = None
    """Path to tesseract executable (if not in PATH)."""


@dataclass(frozen=True)
class DetectionConfig:
    """Detection layer settings."""

    use_ml: bool = True
    """Enable RT-DETR as the primary detector."""

    model_id: str = field(default_factory=_default_detection_model_id)
    """RT-DETR model identifier or local checkpoint path."""

    confidence_threshold: float = 0.35
    """Minimum confidence for ML detections."""

    structural_boost: bool = True
    """Merge structural rule detections with ML detections."""

    nms_iou_threshold: float = 0.35
    """IoU threshold used during final NMS merge."""

    max_detections: int = 120
    """Maximum number of detections returned by the detector."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"
    """Torch device for RT-DETR inference."""


@dataclass(frozen=True)
class LayoutConfig:
    """Layout analysis settings."""

    merge_distance_x: int = 20
    """Max horizontal pixel gap to merge adjacent text blocks."""

    merge_distance_y: int = 12
    """Max vertical pixel gap to merge text lines into a group."""

    min_region_area: int = 400
    """Discard detected regions smaller than this (px²)."""

    heading_font_ratio: float = 1.4
    """Text ≥ this × median font size is classified as a heading."""

    column_detection: bool = True
    """Attempt to detect multi-column layouts."""

    reading_order: Literal["ltr-ttb", "rtl-ttb"] = "ltr-ttb"
    """Assumed reading order — left-to-right top-to-bottom, or RTL."""


@dataclass(frozen=True)
class OutputConfig:
    """Context-pack output format settings."""

    format: Literal["json", "markdown", "yaml"] = "json"
    """Primary output format."""

    include_bboxes: bool = True
    """Include bounding-box coordinates in output."""

    include_confidence: bool = True
    """Include model confidence scores."""

    include_hierarchy: bool = True
    """Include parent→child layout hierarchy."""

    include_summary: bool = True
    """Generate a one-paragraph natural-language summary."""

    max_context_tokens: int = 4096
    """Soft limit — truncate/summarize to fit this token budget."""

    compact: bool = True
    """Minify JSON / strip extra whitespace for token efficiency."""


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class IrpAIConfig:
    """
    Master configuration for the IrpAI pipeline.

    Pass ``tier`` to auto-configure sensible defaults, then override
    individual layer configs as needed.
    """

    tier: Tier = Tier.STANDARD

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "irpai")
    """Directory for downloaded model weights and cached results."""

    verbose: bool = False
    """Enable verbose / debug logging."""

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._apply_tier_defaults()

    def _apply_tier_defaults(self) -> None:
        """Override sub-configs to match the selected tier."""
        if self.tier == Tier.LITE:
            object.__setattr__(self, "detection", DetectionConfig(
                use_ml=False,
                structural_boost=True,
                device="cpu",
            ))
            object.__setattr__(self, "ocr", OCRConfig(
                engine="easyocr",
                fallback_engine="none",
                use_gpu=False,
            ))

        elif self.tier == Tier.STANDARD:
            object.__setattr__(self, "detection", DetectionConfig(
                use_ml=True,
                structural_boost=True,
                device="cpu",
            ))
            object.__setattr__(self, "ocr", OCRConfig(
                engine="easyocr",
                fallback_engine="none",
                use_gpu=False,
            ))
