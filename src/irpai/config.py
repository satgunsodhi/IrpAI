"""
IrpAI Configuration

Defines the tiered execution model and all configurable parameters
for each pipeline layer.

Tiers
-----
- **lite**     : CPU-only, Tesseract OCR, rule-based layout.  Minimal deps.
- **standard** : CPU, PaddleOCR PP-OCRv5, YOLO26n (ONNX), layout analysis.
- **full**     : GPU, PaddleOCR + YOLO + Florence-2 VLM for scene captioning.
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
    FULL = "full"


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
class DetectionConfig:
    """UI element detection settings."""

    model_name: str = "yolo26n"
    """YOLO model variant — n(ano)/s(mall)/m(edium)/l(arge)/x."""

    weights_path: Path | None = None
    """Path to custom fine-tuned weights.  None = download default."""

    confidence_threshold: float = 0.35
    """Minimum confidence to keep a detection."""

    iou_threshold: float = 0.45
    """IoU threshold for NMS."""

    device: str = "cpu"
    """'cpu', 'cuda', 'cuda:0', 'mps', etc."""

    ui_classes: tuple[str, ...] = (
        "button",
        "text_field",
        "icon",
        "image",
        "menu",
        "header",
        "toolbar",
        "sidebar",
        "tab",
        "dropdown",
        "checkbox",
        "radio",
        "slider",
        "card",
        "modal",
        "tooltip",
        "nav_bar",
        "status_bar",
        "content_area",
    )
    """Target UI element classes (used when fine-tuning or mapping)."""


@dataclass(frozen=True)
class OCRConfig:
    """OCR engine settings."""

    engine: Literal["paddleocr", "tesseract"] = "paddleocr"
    """Primary OCR backend."""

    fallback_engine: Literal["tesseract", "none"] = "tesseract"
    """Fallback if primary fails or is unavailable."""

    language: str = "en"
    """Language code(s) for OCR.  PaddleOCR uses 'en', Tesseract uses 'eng'."""

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
class VLMConfig:
    """Vision-Language Model settings (GPU tier only)."""

    model_name: str = "microsoft/Florence-2-base"
    """HuggingFace model identifier."""

    device: str = "cuda"
    """Device for VLM inference."""

    dtype: str = "float16"
    """Weight precision — 'float16', 'bfloat16', 'float32'."""

    max_new_tokens: int = 512
    """Maximum tokens to generate in captioning."""

    tasks: tuple[str, ...] = (
        "<OD>",
        "<OCR_WITH_REGION>",
        "<CAPTION>",
        "<DETAILED_CAPTION>",
    )
    """Florence-2 task prompts to run."""


@dataclass(frozen=True)
class ResearchConfig:
    """Optional research-helper settings."""

    enabled: bool = False
    """Whether to run the research helper."""

    search_provider: Literal["duckduckgo", "serper", "none"] = "duckduckgo"
    """Web search backend."""

    max_results: int = 3
    """Max search results to fetch per query."""

    timeout_seconds: float = 5.0
    """Per-request timeout."""

    api_key: str | None = None
    """API key (Serper only)."""


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
    vlm: VLMConfig = field(default_factory=VLMConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
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
            object.__setattr__(self, "ocr", OCRConfig(
                engine="tesseract",
                fallback_engine="none",
                use_gpu=False,
            ))
            object.__setattr__(self, "detection", DetectionConfig(device="cpu"))
            object.__setattr__(self, "vlm", VLMConfig(device="cpu"))
            object.__setattr__(self, "research", ResearchConfig(enabled=False))

        elif self.tier == Tier.STANDARD:
            object.__setattr__(self, "ocr", OCRConfig(
                engine="paddleocr",
                fallback_engine="tesseract",
                use_gpu=False,
            ))
            object.__setattr__(self, "detection", DetectionConfig(device="cpu"))
            object.__setattr__(self, "research", ResearchConfig(enabled=False))

        elif self.tier == Tier.FULL:
            object.__setattr__(self, "ocr", OCRConfig(
                engine="paddleocr",
                fallback_engine="tesseract",
                use_gpu=True,
            ))
            object.__setattr__(self, "detection", DetectionConfig(device="cuda"))
            object.__setattr__(self, "vlm", VLMConfig(device="cuda"))
