"""
OCR engine factory.

Provides :func:`create_ocr_engine` which returns the appropriate
:class:`BaseOCR` implementation based on configuration, with automatic
fallback if the primary engine is unavailable.
"""

from __future__ import annotations

from irpai.config import OCRConfig
from irpai.ocr.base import BaseOCR
from irpai.utils.logging import get_logger

log = get_logger("ocr.engine")


def create_ocr_engine(config: OCRConfig | None = None) -> BaseOCR:
    """
    Instantiate the best available OCR engine for the given *config*.

    Falls back to the secondary engine if the primary cannot be imported.
    Raises ``RuntimeError`` if no engine is available.
    """
    cfg = config or OCRConfig()

    primary = _try_create(cfg.engine, cfg)
    if primary is not None:
        return primary

    log.warning(
        "Primary OCR engine '%s' unavailable, trying fallback '%s'",
        cfg.engine,
        cfg.fallback_engine,
    )

    if cfg.fallback_engine != "none":
        fallback = _try_create(cfg.fallback_engine, cfg)
        if fallback is not None:
            return fallback

    raise RuntimeError(
        f"No OCR engine available.  Tried: {cfg.engine}, {cfg.fallback_engine}.  "
        "Install paddleocr or pytesseract."
    )


def _try_create(engine_name: str, config: OCRConfig) -> BaseOCR | None:
    """Attempt to import and instantiate the named engine."""
    if engine_name == "easyocr":
        try:
            import easyocr as _  # noqa: F401
            from irpai.ocr.easy_ocr import EasyOCREngine
            return EasyOCREngine(config)
        except ImportError:
            log.debug("easyocr not installed")
            return None

    if engine_name == "paddleocr":
        try:
            import paddleocr as _ # noqa: F401 – availability check
            from irpai.ocr.paddle_ocr import PaddleOCREngine
            return PaddleOCREngine(config)
        except ImportError:
            log.debug("PaddleOCR not installed")
            return None

    if engine_name == "tesseract":
        try:
            import pytesseract as _  # noqa: F401 – availability check
            from irpai.ocr.tesseract_ocr import TesseractOCREngine
            return TesseractOCREngine(config)
        except ImportError:
            log.debug("pytesseract not installed")
            return None

    log.warning("Unknown OCR engine: %s", engine_name)
    return None
