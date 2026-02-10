"""
IrpAI Stage 4: Fusion & Structuring Module

Combines vision and text branches into hierarchical, semantically
meaningful structured context:
- Spatial-text alignment
- Layout region detection
- Platform & application detection
- Spatial relationship inference
- Structured JSON output generation
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from irpai.config import FusionConfig
from irpai.models import (
    BoundingBox,
    LayoutStructure,
    SpatialRelationship,
    StructuredContext,
    TextExtraction,
    UIDetection,
    UIElementType,
)

logger = logging.getLogger(__name__)


class SpatialTextAligner:
    """Aligns extracted text to detected UI elements based on spatial containment."""

    @staticmethod
    def align(
        ui_detections: list[UIDetection],
        text_extractions: list[TextExtraction],
    ) -> list[UIDetection]:
        """
        Map text strings to their containing UI elements.

        Rules:
        1. If text center is inside UI bbox → assign to that UI element
        2. If text overlaps multiple UI elements → assign to smallest container
        3. If text is orphaned → create standalone text_block element

        Args:
            ui_detections: Detected UI elements
            text_extractions: Extracted text regions

        Returns:
            Updated UI detections with aligned text
        """
        # Work on copies
        detections = [d.model_copy(deep=True) for d in ui_detections]
        assigned_texts = set()

        for idx, text_item in enumerate(text_extractions):
            text_cx, text_cy = text_item.bbox.center

            # Find all UI elements containing the text center
            containers = []
            for det in detections:
                if det.bbox.contains_point(text_cx, text_cy):
                    containers.append(det)

            if containers:
                # Assign to smallest container (most specific)
                smallest = min(containers, key=lambda e: e.bbox.area)

                # Append text if element already has text
                if smallest.text:
                    smallest.text += " " + text_item.text
                else:
                    smallest.text = text_item.text
                    smallest.text_confidence = text_item.confidence

                assigned_texts.add(idx)
            else:
                # Check for partial overlap (IoU > 0.3)
                best_overlap = 0.0
                best_det = None
                for det in detections:
                    iou = text_item.bbox.iou(det.bbox)
                    if iou > 0.3 and iou > best_overlap:
                        best_overlap = iou
                        best_det = det

                if best_det:
                    if best_det.text:
                        best_det.text += " " + text_item.text
                    else:
                        best_det.text = text_item.text
                        best_det.text_confidence = text_item.confidence
                    assigned_texts.add(idx)

        # Create orphaned text elements for unassigned text
        for idx, text_item in enumerate(text_extractions):
            if idx not in assigned_texts:
                detections.append(
                    UIDetection(
                        bbox=text_item.bbox,
                        element_type=UIElementType.TEXT_BLOCK,
                        confidence=text_item.confidence,
                        text=text_item.text,
                        text_confidence=text_item.confidence,
                        text_region=True,
                        orphaned=True,
                    )
                )

        return detections


class LayoutDetector:
    """Detects layout regions (navbar, sidebar, footer, main content, modal)."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def detect(
        self, ui_detections: list[UIDetection], img_shape: tuple[int, ...]
    ) -> LayoutStructure:
        """
        Classify UI elements into layout regions using spatial heuristics.

        Common UI Layout Patterns:
        - Navbar: Top 10% of screen, horizontal elements
        - Sidebar: Left or right 15-20%, vertical elements
        - Footer: Bottom 10%, horizontal elements
        - Main Content: Central region, largest area
        - Modal: Centered, overlays other content

        Args:
            ui_detections: List of detected UI elements
            img_shape: Image dimensions (H, W, C)

        Returns:
            LayoutStructure with classified elements
        """
        h, w = img_shape[:2]
        layout = LayoutStructure()

        navbar_y = h * self.config.navbar_threshold
        footer_y = h * self.config.footer_threshold
        sidebar_x_left = w * self.config.sidebar_threshold
        sidebar_x_right = w * (1.0 - self.config.sidebar_threshold)

        for elem in ui_detections:
            cx, cy = elem.bbox.center

            # Modal detection: centered, medium-to-large
            if elem.element_type == UIElementType.MODAL:
                layout.modal.append(elem)
                continue

            # Check if element looks like a modal (centered + significant size)
            rel_w = elem.bbox.width / w
            rel_h = elem.bbox.height / h
            if (
                0.2 < rel_w < 0.8
                and 0.15 < rel_h < 0.8
                and w * 0.15 < cx < w * 0.85
                and h * 0.15 < cy < h * 0.85
                and rel_w * rel_h > 0.1
            ):
                # Could be a modal, but only if it overlaps other regions
                pass  # Let it fall through to normal classification

            # Navbar: top region
            if elem.bbox.y2 < navbar_y:
                layout.navbar.append(elem)

            # Footer: bottom region
            elif elem.bbox.y1 > footer_y:
                layout.footer.append(elem)

            # Sidebar: left or right edge
            elif elem.bbox.x2 < sidebar_x_left or elem.bbox.x1 > sidebar_x_right:
                layout.sidebar.append(elem)

            # Main content: everything else
            else:
                layout.main_content.append(elem)

        # Sort elements within each region by reading order
        for region_name in ["navbar", "sidebar", "main_content", "footer", "modal"]:
            elements = getattr(layout, region_name)
            elements.sort(key=lambda e: (e.bbox.y1, e.bbox.x1))

        logger.debug(
            f"Layout detection: navbar={len(layout.navbar)}, sidebar={len(layout.sidebar)}, "
            f"main={len(layout.main_content)}, footer={len(layout.footer)}, "
            f"modal={len(layout.modal)}"
        )

        return layout


class PlatformDetector:
    """Detects the platform and application from screenshot content."""

    # Known application signatures in title text
    APP_SIGNATURES = {
        "visual studio code": "VS Code",
        "vscode": "VS Code",
        "chrome": "Google Chrome",
        "firefox": "Mozilla Firefox",
        "safari": "Safari",
        "edge": "Microsoft Edge",
        "github": "GitHub",
        "gmail": "Gmail",
        "outlook": "Microsoft Outlook",
        "excel": "Microsoft Excel",
        "word": "Microsoft Word",
        "slack": "Slack",
        "discord": "Discord",
        "terminal": "Terminal",
        "cmd": "Command Prompt",
        "powershell": "PowerShell",
        "file explorer": "File Explorer",
        "finder": "Finder",
        "xcode": "Xcode",
        "intellij": "IntelliJ IDEA",
        "pycharm": "PyCharm",
    }

    @classmethod
    def detect(
        cls,
        ui_detections: list[UIDetection],
        text_extractions: list[TextExtraction],
        img_shape: tuple[int, ...],
    ) -> tuple[str, str]:
        """
        Identify platform (Windows/Mac/Web/Mobile) and application.

        Uses window decorations, UI patterns, and OCR text in title region.

        Args:
            ui_detections: Detected UI elements
            text_extractions: Extracted text
            img_shape: Image dimensions (H, W, C)

        Returns:
            Tuple of (platform, app_title)
        """
        h, w = img_shape[:2]
        platform = "unknown"
        app_title = "Unknown Application"

        # Collect text from title region (top 60 pixels)
        title_texts = []
        for ext in text_extractions:
            if ext.bbox.y2 < 60:
                title_texts.append(ext.text)

        all_title_text = " ".join(title_texts).lower()
        all_text = " ".join(ext.text for ext in text_extractions).lower()

        # Platform detection
        platform = cls._detect_platform(ui_detections, all_title_text, h, w)

        # Application detection
        app_title = cls._detect_application(title_texts, all_text)

        logger.debug(f"Platform: {platform}, App: {app_title}")
        return platform, app_title

    @classmethod
    def _detect_platform(
        cls,
        ui_detections: list[UIDetection],
        title_text: str,
        h: int,
        w: int,
    ) -> str:
        """Detect platform from UI cues."""
        # Mobile detection (aspect ratio)
        if w <= 480 or (h > w * 1.5):
            return "mobile"

        # Windows cues
        windows_cues = ["×", "minimize", "maximize", "restore"]
        if any(cue in title_text for cue in windows_cues):
            return "windows"

        # Mac cues
        mac_cues = ["●", "◉", "apple"]
        if any(cue in title_text for cue in mac_cues):
            return "mac"

        # Web browser cues
        browser_cues = ["http", "https", "www.", ".com", ".org", "chrome", "firefox", "safari"]
        if any(cue in title_text for cue in browser_cues):
            return "web_browser"

        # Count menu elements as web indicator
        menu_count = sum(1 for d in ui_detections if d.element_type == UIElementType.MENU)
        if menu_count > 3:
            return "web_browser"

        # Default: desktop (could be Windows or Mac)
        if w >= 1200:
            return "desktop"

        return "unknown"

    @classmethod
    def _detect_application(cls, title_texts: list[str], all_text: str) -> str:
        """Detect application from text content."""
        # Check title text first
        combined_title = " ".join(title_texts).strip()

        for signature, app_name in cls.APP_SIGNATURES.items():
            if signature in combined_title.lower():
                # Use the actual title text if available
                if combined_title:
                    return combined_title.split(" - ")[0].strip()[:100] or app_name
                return app_name

        # Check all text for app signatures
        for signature, app_name in cls.APP_SIGNATURES.items():
            if signature in all_text:
                return app_name

        # Return title text if available
        if combined_title:
            return combined_title.split(" - ")[0].strip()[:100]

        return "Unknown Application"


class SpatialRelationshipDetector:
    """Infers spatial relationships between UI elements."""

    @staticmethod
    def compute(
        ui_detections: list[UIDetection],
        max_relationships: int = 50,
    ) -> list[SpatialRelationship]:
        """
        Compute pairwise spatial relationships between significant UI elements.

        Relations: above, below, left_of, right_of, inside, overlaps

        Args:
            ui_detections: UI elements to analyze
            max_relationships: Maximum relationships to return

        Returns:
            List of SpatialRelationship objects
        """
        # Only consider elements with text (more meaningful)
        labeled_elements = [d for d in ui_detections if d.text]

        if len(labeled_elements) < 2:
            return []

        relationships = []

        for i, elem_a in enumerate(labeled_elements):
            for j, elem_b in enumerate(labeled_elements):
                if i >= j:
                    continue

                if len(relationships) >= max_relationships:
                    break

                rel = SpatialRelationshipDetector._compute_relation(elem_a, elem_b)
                if rel:
                    relationships.append(rel)

            if len(relationships) >= max_relationships:
                break

        return relationships

    @staticmethod
    def _compute_relation(
        a: UIDetection, b: UIDetection
    ) -> Optional[SpatialRelationship]:
        """Compute the spatial relationship between two elements."""
        a_cx, a_cy = a.bbox.center
        b_cx, b_cy = b.bbox.center

        # Label for elements
        a_label = f"{a.element_type.value}:{(a.text or 'unknown')[:30]}"
        b_label = f"{b.element_type.value}:{(b.text or 'unknown')[:30]}"

        # Check containment
        iou = a.bbox.iou(b.bbox)
        if iou > 0.8:
            if a.bbox.area > b.bbox.area:
                return SpatialRelationship(
                    subject=b_label, relation="inside", obj=a_label
                )
            else:
                return SpatialRelationship(
                    subject=a_label, relation="inside", obj=b_label
                )

        # Check overlap
        if iou > 0.1:
            return SpatialRelationship(
                subject=a_label, relation="overlaps", obj=b_label
            )

        # Directional relationships
        dx = b_cx - a_cx
        dy = b_cy - a_cy

        distance = (dx**2 + dy**2) ** 0.5

        # Only report close relationships
        img_diag = (
            (max(a.bbox.x2, b.bbox.x2)) ** 2 + (max(a.bbox.y2, b.bbox.y2)) ** 2
        ) ** 0.5
        if distance > img_diag * 0.5:
            return None

        # Determine primary direction
        if abs(dy) > abs(dx):
            if dy > 0:
                relation = "above"  # a is above b
            else:
                relation = "below"  # a is below b
        else:
            if dx > 0:
                relation = "left_of"  # a is left of b
            else:
                relation = "right_of"  # a is right of b

        return SpatialRelationship(
            subject=a_label,
            relation=relation,
            obj=b_label,
            distance=round(distance, 1),
        )


class FusionModule:
    """
    Main fusion module combining vision and text branches
    into structured context.

    Pipeline:
    1. Spatial-text alignment
    2. Layout region detection
    3. Platform/application detection
    4. Spatial relationship inference
    5. Structured output generation
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.aligner = SpatialTextAligner()
        self.layout_detector = LayoutDetector(self.config)
        self.spatial_detector = SpatialRelationshipDetector()

    def fuse(
        self,
        ui_detections: list[UIDetection],
        text_extractions: list[TextExtraction],
        img_shape: tuple[int, ...],
    ) -> StructuredContext:
        """
        Combine vision and text outputs into structured context.

        Args:
            ui_detections: UI elements from vision branch
            text_extractions: Text regions from OCR branch
            img_shape: Image dimensions (H, W, C)

        Returns:
            StructuredContext with hierarchical layout
        """
        start_time = time.perf_counter()

        # Step 1: Spatial-text alignment
        aligned_detections = self.aligner.align(ui_detections, text_extractions)

        # Step 2: Layout region detection
        layout = self.layout_detector.detect(aligned_detections, img_shape)

        # Step 3: Platform & application detection
        if self.config.enable_platform_detection:
            platform, app_title = PlatformDetector.detect(
                aligned_detections, text_extractions, img_shape
            )
        else:
            platform = "unknown"
            app_title = "Unknown Application"

        # Step 4: Spatial relationships
        spatial_rels = []
        if self.config.enable_spatial_relationships:
            spatial_rels = self.spatial_detector.compute(
                aligned_detections, self.config.max_spatial_relationships
            )

        # Step 5: Build complete text
        all_text = " ".join(
            ext.text for ext in text_extractions if ext.text.strip()
        )

        # Step 6: Context-aware post-processing
        layout = self._post_process(layout)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        context = StructuredContext(
            platform=platform,
            app_title=app_title,
            layout=layout,
            spatial_relationships=spatial_rels,
            all_text=all_text,
            text_extractions=text_extractions,
            ui_detections=aligned_detections,
            metadata={
                "total_elements": len(aligned_detections),
                "total_text_regions": len(text_extractions),
                "image_dimensions": list(img_shape[:2]),
                "fusion_time_ms": round(elapsed_ms, 1),
                "layout_summary": {
                    "navbar": len(layout.navbar),
                    "sidebar": len(layout.sidebar),
                    "main_content": len(layout.main_content),
                    "footer": len(layout.footer),
                    "modal": len(layout.modal),
                },
            },
        )

        logger.info(
            f"Fusion complete: {len(aligned_detections)} elements, "
            f"{len(spatial_rels)} relationships in {elapsed_ms:.1f}ms"
        )

        return context

    def _post_process(self, layout: LayoutStructure) -> LayoutStructure:
        """
        Apply domain-knowledge corrections to fix common mistakes.

        Rules:
        1. Buttons rarely contain long text (>50 chars)
        2. Navbar elements should be horizontally aligned
        3. Text inputs near labels should be associated
        """
        # Rule 1: Long text in buttons → likely text_block
        for region_name in ["navbar", "sidebar", "main_content", "footer", "modal"]:
            elements = getattr(layout, region_name)
            for elem in elements:
                if (
                    elem.element_type == UIElementType.BUTTON
                    and elem.text
                    and len(elem.text) > 50
                ):
                    elem.element_type = UIElementType.TEXT_BLOCK

        # Rule 2: Sort navbar left-to-right
        layout.navbar.sort(key=lambda e: e.bbox.x1)

        # Rule 3: Associate labels with text inputs
        for region_name in ["main_content", "modal"]:
            elements = getattr(layout, region_name)
            text_inputs = [e for e in elements if e.element_type == UIElementType.TEXT_INPUT]

            for inp in text_inputs:
                # Find text above or to the left
                for elem in elements:
                    if elem.element_type != UIElementType.TEXT_BLOCK or not elem.text:
                        continue

                    elem_cx, elem_cy = elem.bbox.center
                    inp_cx, inp_cy = inp.bbox.center

                    # Text just above the input
                    if (
                        abs(elem_cx - inp_cx) < inp.bbox.width * 0.5
                        and 0 < inp_cy - elem_cy < inp.bbox.height * 3
                    ):
                        inp.label = elem.text
                        break

                    # Text just left of the input
                    if (
                        abs(elem_cy - inp_cy) < inp.bbox.height * 0.5
                        and 0 < inp.bbox.x1 - elem.bbox.x2 < inp.bbox.width * 0.3
                    ):
                        inp.label = elem.text
                        break

        return layout
