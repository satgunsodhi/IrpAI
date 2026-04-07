"""Detection layer exports."""

from irpai.detection.base import BaseDetector, DetectionOutput, DetectionResult
from irpai.detection.hybrid_detector import HybridDetector
from irpai.detection.rule_fallback import FallbackGridDetector
from irpai.detection.rt_detr_detector import RTDETRDetector

__all__ = [
	"BaseDetector",
	"DetectionOutput",
	"DetectionResult",
	"FallbackGridDetector",
	"HybridDetector",
	"RTDETRDetector",
]

