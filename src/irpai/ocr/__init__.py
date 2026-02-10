"""OCR layer — text extraction from screenshot regions."""

from irpai.ocr.base import BaseOCR, OCRResult
from irpai.ocr.engine import create_ocr_engine

__all__ = ["BaseOCR", "OCRResult", "create_ocr_engine"]
