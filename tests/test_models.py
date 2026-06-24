"""
Tests for the irpai.models module.

Validates the Pydantic data models and their converter methods that
bridge to the internal dataclass types.
"""

import pytest

from irpai.models import (
    BoundingBox,
    LayoutRegion,
    LayoutStructure,
    PipelineResult,
    SpatialRelationship,
    StructuredContext,
    TextExtraction,
    UIDetection,
    UIElementType,
)


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_creation(self) -> None:
        bb = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        assert bb.x1 == 10
        assert bb.y2 == 70

    def test_width_and_height(self) -> None:
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        assert bb.width == 100
        assert bb.height == 50

    def test_area(self) -> None:
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        assert bb.area == 5000

    def test_area_degenerate(self) -> None:
        bb = BoundingBox(x1=50, y1=50, x2=50, y2=50)
        assert bb.area == 0

    def test_center(self) -> None:
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bb.center == (50.0, 50.0)

    def test_contains_point_inside(self) -> None:
        bb = BoundingBox(x1=10, y1=10, x2=90, y2=90)
        assert bb.contains_point(50, 50) is True

    def test_contains_point_outside(self) -> None:
        bb = BoundingBox(x1=10, y1=10, x2=90, y2=90)
        assert bb.contains_point(5, 5) is False

    def test_contains_point_edge(self) -> None:
        bb = BoundingBox(x1=10, y1=10, x2=90, y2=90)
        assert bb.contains_point(10, 10) is True

    def test_iou_identical(self) -> None:
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bb.iou(bb) == 1.0

    def test_iou_non_overlapping(self) -> None:
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        assert a.iou(b) == 0.0

    def test_iou_partial_overlap(self) -> None:
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        iou = a.iou(b)
        assert 0.0 < iou < 1.0
        # Intersection = 50*50 = 2500, union = 10000+10000-2500 = 17500
        assert abs(iou - 2500 / 17500) < 1e-6

    def test_to_list(self) -> None:
        bb = BoundingBox(x1=1, y1=2, x2=3, y2=4)
        assert bb.to_list() == [1, 2, 3, 4]

    def test_from_list(self) -> None:
        bb = BoundingBox.from_list([10.5, 20.5, 30.5, 40.5])
        assert bb.x1 == 10.5
        assert bb.y2 == 40.5

    def test_to_tuple(self) -> None:
        bb = BoundingBox(x1=10.7, y1=20.3, x2=30.9, y2=40.1)
        t = bb.to_tuple()
        assert t == (10, 20, 30, 40)
        assert all(isinstance(v, int) for v in t)

    def test_from_tuple(self) -> None:
        bb = BoundingBox.from_tuple((5, 10, 15, 20))
        assert bb.x1 == 5.0
        assert bb.y2 == 20.0

    def test_roundtrip_tuple(self) -> None:
        original = (10, 20, 300, 400)
        bb = BoundingBox.from_tuple(original)
        assert bb.to_tuple() == original


# ---------------------------------------------------------------------------
# UIElementType
# ---------------------------------------------------------------------------


class TestUIElementType:
    def test_button_value(self) -> None:
        assert UIElementType.BUTTON.value == "button"

    def test_from_string(self) -> None:
        assert UIElementType("checkbox") == UIElementType.CHECKBOX

    def test_unknown_fallback(self) -> None:
        with pytest.raises(ValueError):
            UIElementType("nonexistent_type")


# ---------------------------------------------------------------------------
# UIDetection
# ---------------------------------------------------------------------------


class TestUIDetection:
    def test_creation(self) -> None:
        det = UIDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
            element_type=UIElementType.BUTTON,
            confidence=0.95,
            text="Submit",
        )
        assert det.element_type == UIElementType.BUTTON
        assert det.text == "Submit"
        assert det.confidence == 0.95

    def test_defaults(self) -> None:
        det = UIDetection(bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10))
        assert det.element_type == UIElementType.UNKNOWN
        assert det.confidence == 0.0
        assert det.text is None
        assert det.orphaned is False

    def test_from_detection_result(self) -> None:
        from irpai.detection.base import DetectionResult

        dr = DetectionResult(
            label="button",
            bbox=(10, 20, 110, 70),
            confidence=0.88,
        )
        det = UIDetection.from_detection_result(dr)
        assert det.element_type == UIElementType.BUTTON
        assert det.confidence == 0.88
        assert det.bbox.x1 == 10
        assert det.bbox.y2 == 70
        assert det.text_region is True

    def test_from_detection_result_unknown_label(self) -> None:
        from irpai.detection.base import DetectionResult

        dr = DetectionResult(label="widget_xyz", bbox=(0, 0, 50, 50))
        det = UIDetection.from_detection_result(dr)
        assert det.element_type == UIElementType.UNKNOWN


# ---------------------------------------------------------------------------
# TextExtraction
# ---------------------------------------------------------------------------


class TestTextExtraction:
    def test_creation(self) -> None:
        te = TextExtraction(
            text="Hello World",
            bbox=BoundingBox(x1=0, y1=0, x2=200, y2=30),
            confidence=0.92,
        )
        assert te.text == "Hello World"
        assert te.confidence == 0.92

    def test_from_ocr_line(self) -> None:
        from irpai.ocr.base import OCRLine

        line = OCRLine(text="Test text", bbox=(10, 20, 200, 40), confidence=0.85)
        te = TextExtraction.from_ocr_line(line)
        assert te.text == "Test text"
        assert te.confidence == 0.85
        assert te.bbox.x1 == 10

    def test_from_ocr_block(self) -> None:
        from irpai.ocr.base import OCRBlock, OCRLine

        block = OCRBlock(
            lines=[
                OCRLine(text="Line 1", bbox=(10, 10, 200, 30)),
                OCRLine(text="Line 2", bbox=(10, 35, 200, 55)),
            ],
            bbox=(10, 10, 200, 55),
            confidence=0.90,
        )
        te = TextExtraction.from_ocr_block(block)
        assert "Line 1" in te.text
        assert "Line 2" in te.text
        assert te.confidence == 0.90


# ---------------------------------------------------------------------------
# SpatialRelationship
# ---------------------------------------------------------------------------


class TestSpatialRelationship:
    def test_creation(self) -> None:
        rel = SpatialRelationship(
            subject="button:Submit",
            relation="above",
            obj="text_block:Form",
            distance=25.0,
        )
        assert rel.subject == "button:Submit"
        assert rel.relation == "above"
        assert rel.distance == 25.0

    def test_no_distance(self) -> None:
        rel = SpatialRelationship(subject="a", relation="overlaps", obj="b")
        assert rel.distance is None


# ---------------------------------------------------------------------------
# LayoutStructure
# ---------------------------------------------------------------------------


class TestLayoutStructure:
    def test_empty(self) -> None:
        layout = LayoutStructure()
        assert layout.navbar == []
        assert layout.main_content == []

    def test_with_elements(self) -> None:
        det = UIDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=800, y2=40),
            element_type=UIElementType.MENU,
        )
        layout = LayoutStructure(navbar=[det])
        assert len(layout.navbar) == 1
        assert layout.navbar[0].element_type == UIElementType.MENU


# ---------------------------------------------------------------------------
# StructuredContext
# ---------------------------------------------------------------------------


class TestStructuredContext:
    def test_default(self) -> None:
        ctx = StructuredContext()
        assert ctx.platform == "unknown"
        assert ctx.app_title == "Unknown Application"
        assert ctx.ui_detections == []

    def test_to_summary(self) -> None:
        ctx = StructuredContext(
            platform="windows",
            app_title="VS Code",
            ui_detections=[
                UIDetection(
                    bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                    element_type=UIElementType.BUTTON,
                    text="Save",
                ),
            ],
            text_extractions=[
                TextExtraction(
                    text="Hello",
                    bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                ),
            ],
        )
        summary = ctx.to_summary()
        assert "windows" in summary
        assert "VS Code" in summary
        assert "1" in summary  # 1 UI element

    def test_from_context_pack(self) -> None:
        from irpai.context.schema import ContextPack, PipelineMetadata, Region, ScreenSummary

        pack = ContextPack(
            summary=ScreenSummary(
                description="Test app",
                app_type="ide",
                key_elements=["editor", "terminal"],
            ),
            regions=[
                Region(role="button", text="Click me", bbox=(10, 20, 100, 50), confidence=0.9),
                Region(role="body", text="Main content here", bbox=(10, 60, 780, 500)),
            ],
            meta=PipelineMetadata(
                irpai_version="0.1.0",
                tier="standard",
                total_ms=123.4,
            ),
        )

        ctx = StructuredContext.from_context_pack(pack)
        assert ctx.platform == "ide"
        assert len(ctx.ui_detections) == 2
        assert len(ctx.text_extractions) == 2  # both regions have text
        assert "Click me" in ctx.all_text
        assert ctx.metadata["tier"] == "standard"


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_creation(self) -> None:
        ctx = StructuredContext(platform="web_browser", app_title="Google Chrome")
        result = PipelineResult(
            structured_context=ctx,
            processing_time_ms=456.7,
            confidence=0.85,
            warnings=["Low resolution"],
        )
        assert result.processing_time_ms == 456.7
        assert result.confidence == 0.85
        assert len(result.warnings) == 1

    def test_from_context_pack(self) -> None:
        from irpai.context.schema import ContextPack, Region, ScreenSummary

        pack = ContextPack(
            summary=ScreenSummary(description="Test"),
            regions=[Region(role="body", text="Hello")],
        )
        result = PipelineResult.from_context_pack(
            pack,
            processing_time_ms=100.0,
            confidence=0.9,
        )
        assert result.processing_time_ms == 100.0
        assert result.structured_context.ui_detections[0].text == "Hello"
