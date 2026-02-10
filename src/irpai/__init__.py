"""
IrpAI — Image Resolution Pipeline AI

A lightweight machine vision pipeline that converts screenshots into
structured context for LLMs, enabling faster and more relevant responses
on everyday hardware.

Architecture (5-layer pipeline):
    1. Preprocessing — normalize, denoise, scale
    2. Detection — UI element detection via YOLO
    3. OCR — text extraction via PaddleOCR / Tesseract
    4. Layout — region grouping + hierarchical structure
    5. Context — structured JSON context pack for LLMs

Tiers:
    - lite   : CPU-only, Tesseract + rule-based layout
    - standard: CPU, PaddleOCR + YOLO (ONNX) + layout analysis
    - full   : GPU, PaddleOCR + YOLO + Florence-2 VLM
"""

__version__ = "0.1.0"
__author__ = "Aakaash"

from irpai.config import IrpAIConfig, Tier
from irpai.pipeline import IrpAIPipeline

__all__ = ["IrpAIPipeline", "IrpAIConfig", "Tier", "__version__"]
