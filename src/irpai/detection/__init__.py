"""Detection layer — UI element detection and region identification."""

from irpai.detection.base import BaseDetector, DetectionResult
from irpai.detection.yolo_detector import YOLODetector

__all__ = ["BaseDetector", "DetectionResult", "YOLODetector"]
