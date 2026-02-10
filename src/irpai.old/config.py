"""
IrpAI Configuration Module

Manages all pipeline configuration including model paths, thresholds,
and processing parameters. Uses Pydantic for validation.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class Platform(str, Enum):
    WINDOWS = "windows"
    MAC = "mac"
    WEB_BROWSER = "web_browser"
    MOBILE = "mobile"
    UNKNOWN = "unknown"


class PreprocessingConfig(BaseModel):
    """Configuration for the preprocessing stage."""

    max_dimension: int = Field(default=1920, description="Max image dimension in pixels")
    bilateral_d: int = Field(default=9, description="Bilateral filter diameter")
    bilateral_sigma_color: float = Field(default=75.0, description="Bilateral filter sigma color")
    bilateral_sigma_space: float = Field(default=75.0, description="Bilateral filter sigma space")
    clahe_clip_limit: float = Field(default=2.0, description="CLAHE clip limit")
    clahe_tile_grid_size: tuple[int, int] = Field(
        default=(8, 8), description="CLAHE tile grid size"
    )
    enable_denoising: bool = Field(default=True, description="Enable bilateral denoising")
    enable_contrast_enhancement: bool = Field(default=True, description="Enable CLAHE")
    enable_orientation_correction: bool = Field(
        default=True, description="Enable orientation detection/correction"
    )


class VisionConfig(BaseModel):
    """Configuration for the vision branch (UI element detection)."""

    model_name: str = Field(
        default="fasterrcnn_resnet50_fpn", description="Detection model architecture"
    )
    confidence_threshold: float = Field(
        default=0.5, description="Minimum detection confidence"
    )
    nms_threshold: float = Field(default=0.3, description="Non-maximum suppression IoU threshold")
    max_detections: int = Field(default=100, description="Maximum detections per image")
    use_quantized: bool = Field(default=False, description="Use INT8 quantized model")
    custom_model_path: Optional[str] = Field(
        default=None, description="Path to custom-trained detection model"
    )
    detection_classes: list[str] = Field(
        default=[
            "button",
            "text_input",
            "text_block",
            "image",
            "menu",
            "modal",
            "container",
            "icon",
            "checkbox",
            "dropdown",
            "tab",
            "link",
        ],
        description="UI element classes to detect",
    )
    anchor_scales: list[int] = Field(
        default=[32, 64, 128, 256, 512], description="Anchor scales for RPN"
    )
    anchor_aspect_ratios: list[float] = Field(
        default=[0.5, 1.0, 2.0], description="Anchor aspect ratios"
    )


class OCRConfig(BaseModel):
    """Configuration for the text branch (OCR engine)."""

    engine: str = Field(default="paddleocr", description="OCR engine: paddleocr, easyocr, tesseract")
    language: str = Field(default="en", description="Primary OCR language")
    confidence_threshold: float = Field(
        default=0.6, description="Minimum OCR confidence threshold"
    )
    use_angle_cls: bool = Field(default=True, description="Enable text orientation classification")
    use_selective_ocr: bool = Field(
        default=True, description="Only OCR text-containing UI regions"
    )
    enable_ensemble: bool = Field(
        default=False, description="Use multi-OCR ensemble (slower, more accurate)"
    )
    max_text_length: int = Field(default=500, description="Maximum text length per region")


class FusionConfig(BaseModel):
    """Configuration for the fusion and structuring module."""

    navbar_threshold: float = Field(
        default=0.10, description="Top % of screen considered navbar"
    )
    footer_threshold: float = Field(
        default=0.90, description="Bottom % of screen considered footer"
    )
    sidebar_threshold: float = Field(
        default=0.20, description="Side % of screen considered sidebar"
    )
    min_containment_overlap: float = Field(
        default=0.5, description="Minimum IoU for text-UI containment"
    )
    enable_platform_detection: bool = Field(
        default=True, description="Detect platform/application type"
    )
    enable_spatial_relationships: bool = Field(
        default=True, description="Compute spatial relationships between elements"
    )
    max_spatial_relationships: int = Field(
        default=50, description="Maximum spatial relationships to compute"
    )


class LLMConfig(BaseModel):
    """Configuration for the LLM interface."""

    approach: str = Field(
        default="direct_injection",
        description="LLM integration approach: direct_injection or embedding_retrieval",
    )
    max_context_tokens: int = Field(
        default=2000, description="Maximum context tokens for LLM prompt"
    )
    max_elements_per_region: int = Field(
        default=10, description="Max UI elements to include per layout region"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence embedding model"
    )
    embedding_top_k: int = Field(
        default=5, description="Top-K elements for embedding retrieval"
    )
    llm_provider: str = Field(
        default="openai", description="LLM provider: openai, local, anthropic"
    )
    llm_model: str = Field(default="gpt-4", description="LLM model name")
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY", description="Environment variable for API key"
    )
    temperature: float = Field(default=0.3, description="LLM temperature")
    enable_pii_redaction: bool = Field(
        default=True, description="Redact PII from context before sending to LLM"
    )


class PipelineConfig(BaseModel):
    """Master configuration for the entire IrpAI pipeline."""

    device: DeviceType = Field(default=DeviceType.AUTO, description="Compute device")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Pipeline-level settings
    enable_caching: bool = Field(default=True, description="Cache processed screenshots")
    cache_max_size: int = Field(default=100, description="Maximum cached screenshots")
    enable_async: bool = Field(default=True, description="Run vision/text branches in parallel")
    verbose: bool = Field(default=False, description="Verbose logging")
    output_dir: Optional[str] = Field(
        default=None, description="Directory for output files"
    )

    @property
    def resolved_device(self) -> str:
        if self.device == DeviceType.AUTO:
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device.value

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        """Load config from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str | Path) -> None:
        """Save config to JSON file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
