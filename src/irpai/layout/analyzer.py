"""
Layout Analyzer

Combines detection regions and OCR blocks to assign semantic roles
(heading, button label, body text, menu item, …) and group related
elements into coherent screen sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from irpai.config import LayoutConfig
from irpai.detection.base import DetectionOutput, DetectionResult
from irpai.ocr.base import OCRBlock, OCRResult
from irpai.utils.image import (
    BBox,
    box_area,
    box_center,
    boxes_overlap,
    compute_iou,
    merge_boxes,
)
from irpai.utils.logging import get_logger

log = get_logger("layout")


# ---------------------------------------------------------------------------
# Semantic region — the unified unit after merging detection + OCR
# ---------------------------------------------------------------------------


@dataclass
class SemanticRegion:
    """
    A region on screen with a semantic role and associated text.

    This is the primary output of the layout layer, combining spatial
    information from detection with textual content from OCR.
    """

    role: str
    """Semantic role: 'heading', 'body', 'button', 'menu_item', 'nav', 'image', …"""

    bbox: BBox
    """Bounding box (x1, y1, x2, y2)."""

    text: str = ""
    """Extracted / associated text content."""

    confidence: float = 1.0
    """Aggregate confidence from the contributing models."""

    source: str = "merged"
    """How this region was produced: 'detection', 'ocr', 'merged'."""

    children: list[SemanticRegion] = field(default_factory=list)
    """Nested sub-regions (for hierarchical layouts)."""

    meta: dict[str, object] = field(default_factory=dict)
    """Extra metadata (detection label, font size, …)."""

    @property
    def center(self) -> tuple[float, float]:
        return box_center(self.bbox)

    @property
    def area(self) -> int:
        return box_area(self.bbox)


@dataclass
class LayoutResult:
    """Complete layout analysis output."""

    regions: list[SemanticRegion] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    reading_order: list[int] = field(default_factory=list)
    """Indices into *regions* in reading order."""

    @property
    def text_regions(self) -> list[SemanticRegion]:
        return [r for r in self.regions if r.text.strip()]

    @property
    def full_text(self) -> str:
        ordered = (
            [self.regions[i] for i in self.reading_order] if self.reading_order else self.regions
        )
        return "\n".join(r.text for r in ordered if r.text.strip())


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class LayoutAnalyzer:
    """
    Merges detection output and OCR output into semantic regions.

    Steps:
        1. Map each OCR block to its overlapping detection region (if any).
        2. Classify unmatched OCR blocks by heuristic (font size → heading).
        3. Merge nearby text blocks that share a detection region.
        4. Assign reading order (LTR top-to-bottom, or RTL).
    """

    def __init__(self, config: LayoutConfig | None = None) -> None:
        self.cfg = config or LayoutConfig()

    def analyze(
        self,
        ocr: OCRResult,
        detections: DetectionOutput | None = None,
    ) -> LayoutResult:
        """
        Run layout analysis.

        Parameters
        ----------
        ocr:
            Structured OCR output (blocks / lines / words).
        detections:
            Optional UI element detections.  If ``None``, layout is
            inferred purely from OCR spatial heuristics.
        """
        regions: list[SemanticRegion] = []

        # Median font size across all OCR blocks (for heading detection)
        font_sizes = [b.estimated_font_size for b in ocr.blocks if b.estimated_font_size > 0]
        median_font = median(font_sizes) if font_sizes else 14.0

        # -- Step 1: create regions from detection boxes --------------------
        det_regions: list[SemanticRegion] = []
        if detections:
            for det in detections.detections:
                role = self._detection_label_to_role(det.label)
                det_regions.append(
                    SemanticRegion(
                        role=role,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        source="detection",
                        meta={"detection_label": det.label},
                    )
                )

        # If detection produced a single catch-all region, prefer OCR-driven
        # segmentation so we can split mobile UIs into meaningful parts.
        if det_regions and len(det_regions) == 1:
            only = det_regions[0]
            if only.meta.get("detection_label") == "content_area":
                img_w, img_h = ocr.image_size
                if img_w > 0 and img_h > 0:
                    cover = box_area(only.bbox) / float(img_w * img_h)
                    if cover >= 0.9 and len(ocr.blocks) >= 2:
                        det_regions = []

        # -- Step 2: associate OCR blocks with detection regions ------------
        used_ocr: set[int] = set()

        for dr in det_regions:
            matched_text_parts: list[str] = []
            matched_confs: list[float] = []

            for idx, block in enumerate(ocr.blocks):
                if idx in used_ocr:
                    continue
                if self._block_inside_region(block.bbox, dr.bbox):
                    matched_text_parts.append(block.text)
                    matched_confs.append(block.confidence)
                    used_ocr.add(idx)

            if matched_text_parts:
                dr.text = "\n".join(matched_text_parts)
                dr.confidence = (dr.confidence + sum(matched_confs) / len(matched_confs)) / 2

            regions.append(dr)

        # -- Step 3: create regions for unmatched OCR blocks ----------------
        for idx, block in enumerate(ocr.blocks):
            if idx in used_ocr:
                continue
            if not block.text.strip():
                continue
            if box_area(block.bbox) < self.cfg.min_region_area:
                continue

            role = self._classify_text_block(block, median_font)
            regions.append(
                SemanticRegion(
                    role=role,
                    bbox=block.bbox,
                    text=block.text,
                    confidence=block.confidence,
                    source="ocr",
                    meta={"font_size": block.estimated_font_size},
                )
            )

        # -- Step 4: merge nearby regions of the same role ------------------
        regions = self._merge_nearby(regions)

        # -- Step 5: assign reading order -----------------------------------
        reading_order = self._compute_reading_order(regions)

        result = LayoutResult(
            regions=regions,
            image_size=ocr.image_size,
            reading_order=reading_order,
        )

        log.info(
            "Layout: %d regions (%d with text)",
            len(regions),
            len(result.text_regions),
        )
        return result

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _detection_label_to_role(label: str) -> str:
        """Map detection model class names to semantic roles."""
        role_map: dict[str, str] = {
            "button": "button",
            "text_field": "input",
            "icon": "icon",
            "image": "image",
            "menu": "menu",
            "header": "heading",
            "toolbar": "toolbar",
            "sidebar": "sidebar",
            "tab": "tab",
            "dropdown": "dropdown",
            "checkbox": "checkbox",
            "radio": "radio",
            "slider": "slider",
            "card": "card",
            "modal": "modal",
            "tooltip": "tooltip",
            "nav_bar": "nav",
            "status_bar": "status_bar",
            "content_area": "body",
            # Fallback for COCO-mapped labels
            "avatar": "image",
            "device_frame": "container",
            "screen_area": "container",
            "widget": "widget",
            "input_area": "input",
        }
        return role_map.get(label, "unknown")

    def _classify_text_block(self, block: OCRBlock, median_font: float) -> str:
        """Classify a standalone OCR block by heuristic analysis."""
        fs = block.estimated_font_size
        text = block.text.strip()
        bbox = block.bbox

        # Heading: large font OR short single-line all-caps
        if fs >= median_font * self.cfg.heading_font_ratio:
            return "heading"
        if len(block.lines) == 1 and len(text) < 60 and text.isupper():
            return "heading"

        # Menu / nav-like: short labels stacked vertically (sidebar items)
        if len(block.lines) >= 2:
            avg_line_len = sum(len(ln.text.strip()) for ln in block.lines) / len(block.lines)
            if avg_line_len < 20:
                return "menu_item"

        # Code-like: contains indentation, braces, keywords
        code_indicators = ("def ", "class ", "import ", "    ", "(){", "=>", "//", "#")
        if any(ind in text for ind in code_indicators):
            return "code"

        # Button-like: very short text, small area
        if len(text) < 25 and len(block.lines) == 1:
            return "label"

        # Default: body text
        return "body"

    @staticmethod
    def _block_inside_region(block_bbox: BBox, region_bbox: BBox) -> bool:
        """Check if a text block's centre falls inside a detection region."""
        cx, cy = box_center(block_bbox)
        x1, y1, x2, y2 = region_bbox
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _merge_nearby(self, regions: list[SemanticRegion]) -> list[SemanticRegion]:
        """Merge adjacent regions that share the same role and are close."""
        if len(regions) <= 1:
            return regions

        merged: list[SemanticRegion] = []
        used: set[int] = set()

        for i, a in enumerate(regions):
            if i in used:
                continue
            group = [a]
            for j, b in enumerate(regions):
                if j <= i or j in used:
                    continue
                if a.role == b.role and self._should_merge(a.bbox, b.bbox):
                    group.append(b)
                    used.add(j)

            if len(group) == 1:
                merged.append(a)
            else:
                merged.append(self._fuse_regions(group))
            used.add(i)

        return merged

    def _should_merge(self, a: BBox, b: BBox) -> bool:
        """Decide whether two boxes should be merged based on proximity."""
        dx = self.cfg.merge_distance_x
        dy = self.cfg.merge_distance_y

        # Expand box A by the merge distance and check overlap
        expanded = (a[0] - dx, a[1] - dy, a[2] + dx, a[3] + dy)
        return boxes_overlap(expanded, b, threshold=0.0)

    @staticmethod
    def _fuse_regions(group: list[SemanticRegion]) -> SemanticRegion:
        """Fuse multiple regions into one."""
        combined_bbox = merge_boxes([r.bbox for r in group])
        combined_text = "\n".join(r.text for r in group if r.text)
        avg_conf = sum(r.confidence for r in group) / len(group)
        return SemanticRegion(
            role=group[0].role,
            bbox=combined_bbox,
            text=combined_text,
            confidence=avg_conf,
            source="merged",
            children=group,
        )

    def _compute_reading_order(self, regions: list[SemanticRegion]) -> list[int]:
        """
        Sort region indices into reading order.

        Uses a column-aware sort: first groups regions into approximate
        columns, then sorts within each column top-to-bottom.
        """
        if not regions:
            return []

        if not self.cfg.column_detection or len(regions) <= 2:
            # Simple top-to-bottom, left-to-right
            indexed = sorted(
                range(len(regions)),
                key=lambda i: (regions[i].bbox[1], regions[i].bbox[0]),
            )
            return indexed

        # Detect columns by clustering x-centres
        centres = [(i, box_center(r.bbox)[0]) for i, r in enumerate(regions)]
        centres.sort(key=lambda t: t[1])

        columns: list[list[int]] = []
        current_col: list[int] = [centres[0][0]]
        current_x = centres[0][1]

        image_w = max(r.bbox[2] for r in regions) if regions else 1
        col_threshold = image_w * 0.15  # 15% of image width

        for idx, cx in centres[1:]:
            if abs(cx - current_x) < col_threshold:
                current_col.append(idx)
            else:
                columns.append(current_col)
                current_col = [idx]
                current_x = cx
        columns.append(current_col)

        # Sort within each column by y-position
        order: list[int] = []
        for col in columns:
            col.sort(key=lambda i: regions[i].bbox[1])
            order.extend(col)

        return order
