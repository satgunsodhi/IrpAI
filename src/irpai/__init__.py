"""
IrpAI — Image Resolution Pipeline AI

A lightweight machine vision pipeline that converts screenshots into
structured context for LLMs, enabling faster and more relevant responses
on everyday hardware.

Architecture (5-layer pipeline):
    1. Preprocessing — normalize, denoise, scale
    2. Detection — RT-DETR primary + rule-based fallback
    3. OCR — text extraction via PaddleOCR / Tesseract
    4. Layout — region grouping + hierarchical structure
    5. Context — structured JSON context pack for LLMs

Tiers:
    - lite     : CPU-only, fallback detection + OCR + layout
    - standard : CPU, RT-DETR + fallback + OCR + layout analysis
"""

__version__ = "0.1.0"
__author__ = "Aakaash"

from irpai.config import IrpAIConfig, Tier
from irpai.models import (
    BoundingBox,
    LayoutRegion,
    LayoutStructure,
    PipelineResult,
    SpatialRelationship,
    StructuredContext,
    TextExtraction,
    UIDetection,
    UIElementType,
)
from irpai.pipeline import IrpAIPipeline

__all__ = [
    "IrpAIPipeline",
    "IrpAIConfig",
    "Tier",
    "__version__",
    # Data models
    "BoundingBox",
    "UIElementType",
    "UIDetection",
    "TextExtraction",
    "SpatialRelationship",
    "LayoutRegion",
    "LayoutStructure",
    "StructuredContext",
    "PipelineResult",
]

