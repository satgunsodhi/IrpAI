"""Legacy compatibility shim for older imports.

This module intentionally no longer contains YOLO logic.
Use ``irpai.detection.rule_fallback`` and ``irpai.detection.hybrid_detector``
for current detection implementations.
"""

from irpai.detection.rule_fallback import FallbackGridDetector

__all__ = ["FallbackGridDetector"]
