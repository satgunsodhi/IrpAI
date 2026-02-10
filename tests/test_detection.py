"""
Tests for the detection layer.
"""

import numpy as np
from PIL import Image

from irpai.detection.base import DetectionOutput, DetectionResult
from irpai.detection.yolo_detector import FallbackGridDetector


class TestDetectionTypes:
    def test_detection_result(self) -> None:
        det = DetectionResult(label="button", bbox=(10, 20, 100, 50), confidence=0.9)
        assert det.area == 90 * 30
        assert det.center == (55.0, 35.0)

    def test_detection_output_filter_by_confidence(self) -> None:
        dets = [
            DetectionResult(label="button", bbox=(0, 0, 10, 10), confidence=0.9),
            DetectionResult(label="icon", bbox=(20, 20, 30, 30), confidence=0.3),
            DetectionResult(label="text_field", bbox=(40, 40, 50, 50), confidence=0.7),
        ]
        output = DetectionOutput(detections=dets, image_size=(100, 100))

        filtered = output.filter_by_confidence(0.5)
        assert len(filtered) == 2
        assert all(d.confidence >= 0.5 for d in filtered)

    def test_detection_output_filter_by_label(self) -> None:
        dets = [
            DetectionResult(label="button", bbox=(0, 0, 10, 10), confidence=0.9),
            DetectionResult(label="icon", bbox=(20, 20, 30, 30), confidence=0.8),
            DetectionResult(label="button", bbox=(40, 40, 50, 50), confidence=0.7),
        ]
        output = DetectionOutput(detections=dets)

        buttons = output.filter_by_label("button")
        assert len(buttons) == 2

    def test_sorted_by_position(self) -> None:
        dets = [
            DetectionResult(label="a", bbox=(100, 200, 150, 250), confidence=1.0),
            DetectionResult(label="b", bbox=(10, 10, 50, 50), confidence=1.0),
            DetectionResult(label="c", bbox=(50, 100, 100, 150), confidence=1.0),
        ]
        output = DetectionOutput(detections=dets)

        sorted_dets = output.sorted_by_position()
        assert sorted_dets[0].label == "b"  # top-left
        assert sorted_dets[1].label == "c"  # middle
        assert sorted_dets[2].label == "a"  # bottom


class TestFallbackGridDetector:
    def test_solid_image_gives_single_region(self) -> None:
        """A solid-colour image has no edges → one full-image content_area."""
        img = Image.new("RGB", (900, 600), color=(30, 30, 30))
        detector = FallbackGridDetector()
        output = detector.detect(img)

        assert len(output.detections) == 1
        assert output.image_size == (900, 600)
        assert output.detections[0].label == "content_area"
        assert output.detections[0].bbox == (0, 0, 900, 600)

    def test_horizontal_bands_detected(self) -> None:
        """An image with a title bar and status bar should get split."""
        from PIL import ImageDraw
        img = Image.new("RGB", (900, 600), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        # Title bar (larger so band-averaging picks it up)
        draw.rectangle([0, 0, 900, 50], fill=(80, 80, 80))
        # Status bar
        draw.rectangle([0, 560, 900, 600], fill=(0, 122, 204))

        detector = FallbackGridDetector()
        output = detector.detect(img)

        assert output.image_size == (900, 600)
        # Should find at least 2 regions (main + status)
        assert len(output.detections) >= 2
        labels = {d.label for d in output.detections}
        assert "status_bar" in labels or "header" in labels or "content_area" in labels

    def test_sidebar_detected(self) -> None:
        """An image with a very distinct sidebar should be split vertically."""
        from PIL import ImageDraw
        img = Image.new("RGB", (900, 600), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        # Very distinct sidebar (big colour difference)
        draw.rectangle([0, 0, 200, 600], fill=(80, 80, 100))

        detector = FallbackGridDetector()
        output = detector.detect(img)

        assert len(output.detections) >= 2

    def test_numpy_input(self) -> None:
        arr = np.zeros((600, 900, 3), dtype=np.uint8)
        detector = FallbackGridDetector()
        output = detector.detect(arr)

        assert output.image_size == (900, 600)
        assert len(output.detections) > 0
