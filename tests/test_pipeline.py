"""
IrpAI Test Suite - Core Pipeline Tests

Tests for preprocessing, models, fusion, and pipeline integration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest  # type: ignore[import-not-found]

from irpai.config import PipelineConfig, PreprocessingConfig, FusionConfig
from irpai.models import (
    BoundingBox,
    LayoutStructure,
    SpatialRelationship,
    StructuredContext,
    TextExtraction,
    UIDetection,
    UIElementType,
)
from irpai.preprocessing import OrientationDetector, Preprocessor
from irpai.fusion import (
    FusionModule,
    LayoutDetector,
    PlatformDetector,
    SpatialTextAligner,
)
from irpai.ocr import PIIRedactor
from irpai.llm_interface import PromptBuilder


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a synthetic test screenshot (1920x1080)."""
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 240  # Light gray background

    # Navbar (top bar)
    cv2.rectangle(img, (0, 0), (1920, 60), (50, 50, 50), -1)
    cv2.putText(img, "MyApp - Dashboard", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Sidebar (left)
    cv2.rectangle(img, (0, 60), (250, 1080), (220, 220, 220), -1)
    cv2.putText(img, "Home", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Settings", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Main content area
    cv2.putText(img, "Welcome to Dashboard", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "This is sample content", (400, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)

    # Buttons
    cv2.rectangle(img, (400, 400), (550, 450), (0, 120, 215), -1)
    cv2.putText(img, "Submit", (420, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.rectangle(img, (580, 400), (730, 450), (200, 200, 200), -1)
    cv2.putText(img, "Cancel", (600, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Footer
    cv2.rectangle(img, (0, 1030), (1920, 1080), (50, 50, 50), -1)
    cv2.putText(img, "Copyright 2026", (800, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


@pytest.fixture
def sample_image_path(sample_image) -> str:
    """Save sample image to temp file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, sample_image)
        return f.name


@pytest.fixture
def sample_ui_detections() -> list[UIDetection]:
    """Create sample UI detections."""
    return [
        UIDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=1920, y2=60),
            element_type=UIElementType.MENU,
            confidence=0.9,
            text_region=True,
        ),
        UIDetection(
            bbox=BoundingBox(x1=0, y1=60, x2=250, y2=1080),
            element_type=UIElementType.CONTAINER,
            confidence=0.85,
            text_region=True,
        ),
        UIDetection(
            bbox=BoundingBox(x1=400, y1=400, x2=550, y2=450),
            element_type=UIElementType.BUTTON,
            confidence=0.94,
            text_region=True,
        ),
        UIDetection(
            bbox=BoundingBox(x1=580, y1=400, x2=730, y2=450),
            element_type=UIElementType.BUTTON,
            confidence=0.91,
            text_region=True,
        ),
        UIDetection(
            bbox=BoundingBox(x1=400, y1=160, x2=900, y2=300),
            element_type=UIElementType.TEXT_BLOCK,
            confidence=0.88,
            text_region=True,
        ),
    ]


@pytest.fixture
def sample_text_extractions() -> list[TextExtraction]:
    """Create sample text extractions."""
    return [
        TextExtraction(
            text="MyApp - Dashboard",
            bbox=BoundingBox(x1=20, y1=10, x2=350, y2=50),
            confidence=0.96,
        ),
        TextExtraction(
            text="Home",
            bbox=BoundingBox(x1=20, y1=100, x2=100, y2=130),
            confidence=0.95,
        ),
        TextExtraction(
            text="Settings",
            bbox=BoundingBox(x1=20, y1=150, x2=130, y2=180),
            confidence=0.94,
        ),
        TextExtraction(
            text="Welcome to Dashboard",
            bbox=BoundingBox(x1=400, y1=170, x2=800, y2=220),
            confidence=0.97,
        ),
        TextExtraction(
            text="Submit",
            bbox=BoundingBox(x1=420, y1=415, x2=540, y2=445),
            confidence=0.98,
        ),
        TextExtraction(
            text="Cancel",
            bbox=BoundingBox(x1=600, y1=415, x2=720, y2=445),
            confidence=0.93,
        ),
        TextExtraction(
            text="Copyright 2026",
            bbox=BoundingBox(x1=800, y1=1040, x2=1000, y2=1070),
            confidence=0.90,
        ),
    ]


# ── BoundingBox Tests ───────────────────────────────────────


class TestBoundingBox:
    def test_properties(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        assert bbox.width == 90
        assert bbox.height == 60
        assert bbox.area == 5400
        assert bbox.center == (55.0, 50.0)

    def test_contains_point(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        assert bbox.contains_point(50, 50) is True
        assert bbox.contains_point(5, 50) is False
        assert bbox.contains_point(50, 90) is False

    def test_iou_identical(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        assert abs(bbox.iou(bbox) - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        bbox_iou = a.iou(b)
        assert bbox_iou == 0.0

    def test_iou_partial_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        iou = a.iou(b)
        assert 0 < iou < 1

    def test_to_from_list(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        coords = bbox.to_list()
        restored = BoundingBox.from_list(coords)
        assert bbox == restored


# ── Preprocessing Tests ─────────────────────────────────────


class TestPreprocessor:
    def test_process_numpy_input(self, sample_image):
        preprocessor = Preprocessor()
        result = preprocessor.process(sample_image)
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # BGR
        assert result.dtype == np.uint8

    def test_process_file_input(self, sample_image_path):
        preprocessor = Preprocessor()
        result = preprocessor.process(sample_image_path)
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3

    def test_resolution_normalization(self):
        # Create oversized image
        big_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        config = PreprocessingConfig(max_dimension=1920)
        preprocessor = Preprocessor(config)
        result = preprocessor.process(big_img)
        assert max(result.shape[:2]) <= 1920

    def test_no_upscaling(self):
        # Small images should NOT be upscaled
        small_img = np.zeros((480, 640, 3), dtype=np.uint8)
        preprocessor = Preprocessor()
        result = preprocessor.process(small_img)
        assert result.shape[:2] == (480, 640)

    def test_quality_metrics(self, sample_image):
        metrics = Preprocessor.compute_quality_metrics(sample_image)
        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "sharpness" in metrics
        assert "resolution" in metrics
        assert "is_low_resolution" in metrics
        assert metrics["is_low_resolution"] is False  # 1920x1080

    def test_file_not_found(self):
        preprocessor = Preprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.process("/nonexistent/path.png")

    def test_disabled_stages(self, sample_image):
        config = PreprocessingConfig(
            enable_denoising=False,
            enable_contrast_enhancement=False,
            enable_orientation_correction=False,
        )
        preprocessor = Preprocessor(config)
        result = preprocessor.process(sample_image)
        assert isinstance(result, np.ndarray)


class TestOrientationDetector:
    def test_detect_normal(self, sample_image):
        angle = OrientationDetector.detect_orientation(sample_image)
        assert angle in [0, 90, 180, 270]

    def test_rotate_0(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = OrientationDetector.rotate_image(img, 0)
        assert result.shape == img.shape

    def test_rotate_90(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = OrientationDetector.rotate_image(img, 90)
        assert result.shape == (200, 100, 3)

    def test_rotate_180(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = OrientationDetector.rotate_image(img, 180)
        assert result.shape == (100, 200, 3)


# ── Fusion Tests ────────────────────────────────────────────


class TestSpatialTextAligner:
    def test_text_assigned_to_button(self, sample_ui_detections, sample_text_extractions):
        aligned = SpatialTextAligner.align(sample_ui_detections, sample_text_extractions)

        # Find the submit button
        submit_buttons = [
            d for d in aligned
            if d.element_type == UIElementType.BUTTON and d.text and "Submit" in d.text
        ]
        assert len(submit_buttons) >= 1

    def test_orphaned_text_creates_element(self):
        ui_dets = []  # No UI detections
        text_exts = [
            TextExtraction(
                text="Orphaned text",
                bbox=BoundingBox(x1=500, y1=500, x2=700, y2=530),
                confidence=0.9,
            )
        ]

        aligned = SpatialTextAligner.align(ui_dets, text_exts)
        assert len(aligned) == 1
        assert aligned[0].orphaned is True
        assert aligned[0].text == "Orphaned text"


class TestLayoutDetector:
    def test_navbar_detection(self, sample_ui_detections):
        detector = LayoutDetector()
        layout = detector.detect(sample_ui_detections, (1080, 1920, 3))

        # Top menu should be in navbar
        assert len(layout.navbar) > 0

    def test_main_content_detection(self, sample_ui_detections):
        detector = LayoutDetector()
        layout = detector.detect(sample_ui_detections, (1080, 1920, 3))

        assert len(layout.main_content) > 0

    def test_empty_detections(self):
        detector = LayoutDetector()
        layout = detector.detect([], (1080, 1920, 3))

        assert len(layout.navbar) == 0
        assert len(layout.sidebar) == 0
        assert len(layout.main_content) == 0
        assert len(layout.footer) == 0


class TestPlatformDetector:
    def test_detect_web(self, sample_ui_detections, sample_text_extractions):
        # Add browser-like text
        browser_text = TextExtraction(
            text="https://www.example.com",
            bbox=BoundingBox(x1=200, y1=10, x2=500, y2=40),
            confidence=0.95,
        )
        exts = sample_text_extractions + [browser_text]

        platform, app = PlatformDetector.detect(
            sample_ui_detections, exts, (1080, 1920, 3)
        )
        assert platform in ["web_browser", "desktop", "unknown"]

    def test_detect_mobile(self):
        # Narrow image = mobile
        platform, _ = PlatformDetector.detect([], [], (800, 400, 3))
        assert platform == "mobile"


class TestFusionModule:
    def test_full_fusion(self, sample_ui_detections, sample_text_extractions):
        fusion = FusionModule()
        context = fusion.fuse(
            sample_ui_detections,
            sample_text_extractions,
            (1080, 1920, 3),
        )

        assert isinstance(context, StructuredContext)
        assert context.platform != ""
        assert len(context.ui_detections) > 0
        assert len(context.all_text) > 0
        assert "total_elements" in context.metadata

    def test_empty_inputs(self):
        fusion = FusionModule()
        context = fusion.fuse([], [], (1080, 1920, 3))

        assert isinstance(context, StructuredContext)
        assert context.platform == "unknown"


# ── OCR Utility Tests ───────────────────────────────────────


class TestPIIRedactor:
    def test_redact_email(self):
        exts = [
            TextExtraction(
                text="Contact: user@example.com",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=20),
                confidence=0.9,
            )
        ]
        redacted = PIIRedactor.redact(exts)
        assert "user@example.com" not in redacted[0].text
        assert "[REDACTED_EMAIL]" in redacted[0].text

    def test_redact_phone(self):
        exts = [
            TextExtraction(
                text="Call: 555-123-4567",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=20),
                confidence=0.9,
            )
        ]
        redacted = PIIRedactor.redact(exts)
        assert "555-123-4567" not in redacted[0].text

    def test_detect_pii(self):
        found = PIIRedactor.detect_pii("Email: a@b.com, Phone: 555-123-4567")
        types = [f["type"] for f in found]
        assert "email" in types
        assert "phone" in types

    def test_no_pii(self):
        exts = [
            TextExtraction(
                text="Hello world",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=20),
                confidence=0.9,
            )
        ]
        redacted = PIIRedactor.redact(exts)
        assert redacted[0].text == "Hello world"


# ── LLM Interface Tests ────────────────────────────────────


class TestPromptBuilder:
    def test_build_direct_prompt(
        self, sample_ui_detections, sample_text_extractions
    ):
        fusion = FusionModule()
        context = fusion.fuse(
            sample_ui_detections, sample_text_extractions, (1080, 1920, 3)
        )

        builder = PromptBuilder()
        prompt = builder.build_prompt("Where is the submit button?", context)

        assert "submit" in prompt.lower() or "Submit" in prompt
        assert "USER QUESTION" in prompt
        assert "Where is the submit button?" in prompt

    def test_prompt_includes_platform(
        self, sample_ui_detections, sample_text_extractions
    ):
        fusion = FusionModule()
        context = fusion.fuse(
            sample_ui_detections, sample_text_extractions, (1080, 1920, 3)
        )

        builder = PromptBuilder()
        prompt = builder.build_prompt("test", context)

        assert "APPLICATION" in prompt


# ── Config Tests ────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.preprocessing.max_dimension == 1920
        assert config.ocr.confidence_threshold == 0.6
        assert config.vision.confidence_threshold == 0.5

    def test_save_load_config(self):
        config = PipelineConfig()
        config.verbose = True
        config.ocr.language = "ch"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            config.to_file(f.name)
            loaded = PipelineConfig.from_file(f.name)

        assert loaded.verbose is True
        assert loaded.ocr.language == "ch"

    def test_device_resolution(self):
        config = PipelineConfig()
        device = config.resolved_device
        assert device in ["cpu", "cuda"]


# ── Structured Context Tests ────────────────────────────────


class TestStructuredContext:
    def test_to_summary(self, sample_ui_detections, sample_text_extractions):
        fusion = FusionModule()
        context = fusion.fuse(
            sample_ui_detections, sample_text_extractions, (1080, 1920, 3)
        )

        summary = context.to_summary()
        assert "Platform" in summary
        assert "Application" in summary
        assert "Total UI Elements" in summary

    def test_serialization(self, sample_ui_detections, sample_text_extractions):
        fusion = FusionModule()
        context = fusion.fuse(
            sample_ui_detections, sample_text_extractions, (1080, 1920, 3)
        )

        # Should be JSON-serializable
        json_str = context.model_dump_json()
        assert isinstance(json_str, str)

        # Should be recoverable
        data = json.loads(json_str)
        assert "platform" in data
        assert "layout" in data
        assert "all_text" in data
