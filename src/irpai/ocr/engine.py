"""
OCR engine factory.

Provides :func:`create_ocr_engine` which returns the appropriate
:class:`BaseOCR` implementation based on configuration.
"""

from __future__ import annotations

from irpai.config import OCRConfig
from irpai.ocr.base import BaseOCR
from irpai.utils.logging import get_logger

log = get_logger("ocr.engine")


def create_ocr_engine(config: OCRConfig | None = None) -> BaseOCR:
    """
    Instantiate the OCR engine for the given *config*.

    Raises ``RuntimeError`` if EasyOCR is not available.
    """
    cfg = config or OCRConfig()
    if cfg.engine != "easyocr":
        log.warning("Unknown OCR engine '%s'; using easyocr", cfg.engine)

    try:
        import easyocr as _  # noqa: F401
        from irpai.ocr.easy_ocr import EasyOCREngine
        return EasyOCREngine(cfg)
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR is required but not installed. "
            "Install with: pip install -e .[ocr-easyocr]"
        ) from exc
