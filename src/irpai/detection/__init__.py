"""Detection layer — UI element detection and region identification."""

from irpai.detection.base import BaseDetector, DetectionResult
from irpai.detection.yolo_detector import FallbackGridDetector

__all__ = ["BaseDetector", "DetectionResult", "FallbackGridDetector"]

