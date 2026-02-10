"""
Tests for the layout analysis layer.
"""

from irpai.layout.analyzer import LayoutAnalyzer, SemanticRegion
from irpai.layout.hierarchy import HierarchyBuilder, LayoutNode
from irpai.ocr.base import OCRBlock, OCRLine, OCRResult
from irpai.detection.base import DetectionOutput, DetectionResult


def _make_ocr_result() -> OCRResult:
    """Create a realistic OCR result for testing."""
    blocks = [
        OCRBlock(
            lines=[OCRLine(text="My Application", bbox=(10, 5, 300, 30), confidence=0.95)],
            bbox=(10, 5, 300, 30),
            confidence=0.95,
        ),
        OCRBlock(
            lines=[
                OCRLine(text="Welcome to the dashboard", bbox=(220, 50, 700, 75), confidence=0.92),
                OCRLine(text="Here is your overview.", bbox=(220, 80, 700, 100), confidence=0.9),
            ],
            bbox=(220, 50, 700, 100),
            confidence=0.91,
        ),
        OCRBlock(
            lines=[OCRLine(text="Settings", bbox=(10, 60, 180, 80), confidence=0.88)],
            bbox=(10, 60, 180, 80),
            confidence=0.88,
        ),
        OCRBlock(
            lines=[OCRLine(text="Save", bbox=(600, 550, 680, 575), confidence=0.97)],
            bbox=(600, 550, 680, 575),
            confidence=0.97,
        ),
    ]
    return OCRResult(blocks=blocks, image_size=(800, 600), engine="test")


def _make_detections() -> DetectionOutput:
    """Create detection output matching the OCR result."""
    return DetectionOutput(
        detections=[
            DetectionResult(label="header", bbox=(0, 0, 800, 40), confidence=0.85),
            DetectionResult(label="sidebar", bbox=(0, 40, 200, 580), confidence=0.8),
            DetectionResult(label="content_area", bbox=(200, 40, 800, 540), confidence=0.82),
            DetectionResult(label="button", bbox=(590, 545, 690, 580), confidence=0.9),
        ],
        image_size=(800, 600),
        model_name="test",
    )


class TestLayoutAnalyzer:
    def test_ocr_only_analysis(self) -> None:
        ocr = _make_ocr_result()
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(ocr)

        assert len(layout.regions) > 0
        assert len(layout.reading_order) == len(layout.regions)
        assert layout.full_text  # should have some text

    def test_ocr_plus_detection(self) -> None:
        ocr = _make_ocr_result()
        detections = _make_detections()
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(ocr, detections)

        # Should have regions from both detection and unmatched OCR
        assert len(layout.regions) > 0

        # Detection-sourced regions should have text from matching OCR blocks
        roles = {r.role for r in layout.regions}
        assert "heading" in roles or "body" in roles

    def test_reading_order(self) -> None:
        ocr = _make_ocr_result()
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(ocr)

        # Reading order should cover all regions
        assert len(layout.reading_order) == len(layout.regions)
        assert set(layout.reading_order) == set(range(len(layout.regions)))

    def test_empty_input(self) -> None:
        ocr = OCRResult(image_size=(800, 600), engine="test")
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(ocr)

        assert len(layout.regions) == 0
        assert layout.full_text == ""

    def test_heading_detection(self) -> None:
        """Large text should be classified as heading."""
        blocks = [
            OCRBlock(
                lines=[OCRLine(text="BIG TITLE", bbox=(10, 10, 300, 50), confidence=0.95)],
                bbox=(10, 10, 300, 50),
                confidence=0.95,
            ),
            OCRBlock(
                lines=[OCRLine(text="small body text here", bbox=(10, 60, 300, 75), confidence=0.9)],
                bbox=(10, 60, 300, 75),
                confidence=0.9,
            ),
        ]
        ocr = OCRResult(blocks=blocks, image_size=(400, 200), engine="test")
        analyzer = LayoutAnalyzer()
        layout = analyzer.analyze(ocr)

        roles = [r.role for r in layout.regions]
        assert "heading" in roles


class TestHierarchyBuilder:
    def test_nesting(self) -> None:
        """Large container should become parent of small region inside it."""
        regions = [
            SemanticRegion(role="container", bbox=(0, 0, 800, 600)),
            SemanticRegion(role="heading", bbox=(10, 10, 300, 40), text="Title"),
            SemanticRegion(role="body", bbox=(10, 50, 700, 500), text="Content"),
        ]
        builder = HierarchyBuilder()
        roots = builder.build(regions)

        assert len(roots) == 1  # container is the root
        assert roots[0].role == "container"
        assert len(roots[0].children) == 2

    def test_flat_layout(self) -> None:
        """Non-overlapping regions should all be roots."""
        regions = [
            SemanticRegion(role="heading", bbox=(0, 0, 400, 40), text="Left"),
            SemanticRegion(role="heading", bbox=(400, 0, 800, 40), text="Right"),
        ]
        builder = HierarchyBuilder()
        roots = builder.build(regions)

        assert len(roots) == 2

    def test_to_dict(self) -> None:
        regions = [
            SemanticRegion(role="card", bbox=(0, 0, 300, 200)),
            SemanticRegion(role="heading", bbox=(10, 10, 280, 40), text="Card Title"),
        ]
        builder = HierarchyBuilder()
        roots = builder.build(regions)

        d = roots[0].to_dict()
        assert d["role"] == "card"
        assert "children" in d

    def test_empty(self) -> None:
        builder = HierarchyBuilder()
        roots = builder.build([])
        assert roots == []
