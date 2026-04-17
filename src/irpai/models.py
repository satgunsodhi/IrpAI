"""
IrpAI Data Models

Dataclass models for structured data flowing through the pipeline.

These provide rich, validated domain types with utility methods
(IoU computation, spatial containment, serialisation helpers, etc.)
while remaining interoperable with the lightweight dataclass types
used internally by each pipeline sub-package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UIElementType(str, Enum):
    """Semantic type of a detected UI element."""

    BUTTON = "button"
    TEXT_INPUT = "text_input"
    TEXT_BLOCK = "text_block"
    IMAGE = "image"
    MENU = "menu"
    MODAL = "modal"
    CONTAINER = "container"
    ICON = "icon"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    TAB = "tab"
    LINK = "link"
    UNKNOWN = "unknown"


class LayoutRegion(str, Enum):
    """Named layout region on screen."""

    NAVBAR = "navbar"
    SIDEBAR = "sidebar"
    FOOTER = "footer"
    MAIN_CONTENT = "main_content"
    MODAL = "modal"


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box ``[x1, y1, x2, y2]``."""

    x1: float
    y1: float
    x2: float
    y2: float

    # -- computed properties ------------------------------------------------

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    # -- spatial helpers ----------------------------------------------------

    def contains_point(self, x: float, y: float) -> bool:
        """Return ``True`` if the point ``(x, y)`` lies inside the box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection-over-Union with another box."""
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    # -- serialisation helpers ----------------------------------------------

    def to_list(self) -> list[float]:
        """Return ``[x1, y1, x2, y2]`` as a plain list."""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: list[float]) -> BoundingBox:
        """Create from a ``[x1, y1, x2, y2]`` list."""
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convert to the ``BBox`` tuple alias used by the new sub-packages."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @classmethod
    def from_tuple(cls, bbox: tuple[int, int, int, int] | tuple[float, float, float, float]) -> BoundingBox:
        """Create from a ``(x1, y1, x2, y2)`` tuple."""
        return cls(x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]))

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


# ---------------------------------------------------------------------------
# Detection models
# ---------------------------------------------------------------------------

@dataclass
class UIDetection:
    """A detected UI element with optional OCR text."""

    bbox: BoundingBox
    element_type: UIElementType = UIElementType.UNKNOWN
    confidence: float = 0.0
    text_region: bool = False
    text: Optional[str] = None
    text_confidence: Optional[float] = None
    icon_type: Optional[str] = None
    label: Optional[str] = None
    orphaned: bool = False

    # -- converter from new dataclass type ----------------------------------

    @classmethod
    def from_detection_result(
        cls,
        det: object,  # irpai.detection.base.DetectionResult (avoid circular import)
    ) -> UIDetection:
        """
        Create a ``UIDetection`` from a :class:`~irpai.detection.base.DetectionResult`.

        Maps the string ``label`` to the closest :class:`UIElementType` enum value.
        """
        from irpai.detection.base import DetectionResult

        if not isinstance(det, DetectionResult):
            raise TypeError(f"Expected DetectionResult, got {type(det).__name__}")

        # Map label string → UIElementType
        try:
            elem_type = UIElementType(det.label)
        except ValueError:
            elem_type = UIElementType.UNKNOWN

        return cls(
            bbox=BoundingBox.from_tuple(det.bbox),
            element_type=elem_type,
            confidence=det.confidence,
            text_region=elem_type in (
                UIElementType.BUTTON,
                UIElementType.TEXT_INPUT,
                UIElementType.TEXT_BLOCK,
                UIElementType.MENU,
                UIElementType.TAB,
                UIElementType.LINK,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bbox": self.bbox.to_dict(),
            "element_type": self.element_type.value,
            "confidence": self.confidence,
        }
        if self.text is not None:
            d["text"] = self.text
        if self.label is not None:
            d["label"] = self.label
        return d


# ---------------------------------------------------------------------------
# OCR / text models
# ---------------------------------------------------------------------------

@dataclass
class TextExtraction:
    """An extracted text region from OCR."""

    text: str
    bbox: BoundingBox
    confidence: float = 0.0
    ui_region: Optional[str] = None
    language: Optional[str] = None

    # -- converter from new dataclass type ----------------------------------

    @classmethod
    def from_ocr_line(
        cls,
        line: object,  # irpai.ocr.base.OCRLine
    ) -> TextExtraction:
        """
        Create a ``TextExtraction`` from a :class:`~irpai.ocr.base.OCRLine`.
        """
        from irpai.ocr.base import OCRLine

        if not isinstance(line, OCRLine):
            raise TypeError(f"Expected OCRLine, got {type(line).__name__}")

        return cls(
            text=line.text,
            bbox=BoundingBox.from_tuple(line.bbox),
            confidence=line.confidence,
        )

    @classmethod
    def from_ocr_block(
        cls,
        block: object,  # irpai.ocr.base.OCRBlock
    ) -> TextExtraction:
        """
        Create a ``TextExtraction`` from a :class:`~irpai.ocr.base.OCRBlock`.
        """
        from irpai.ocr.base import OCRBlock

        if not isinstance(block, OCRBlock):
            raise TypeError(f"Expected OCRBlock, got {type(block).__name__}")

        return cls(
            text=block.text,
            bbox=BoundingBox.from_tuple(block.bbox),
            confidence=block.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }
        if self.ui_region is not None:
            d["ui_region"] = self.ui_region
        return d


# ---------------------------------------------------------------------------
# Spatial relationships
# ---------------------------------------------------------------------------

@dataclass
class SpatialRelationship:
    """Spatial relationship between two UI elements."""

    subject: str
    relation: str  # above, below, left_of, right_of, inside, overlaps
    obj: str  # named 'obj' because 'object' is reserved
    distance: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
        }
        if self.distance is not None:
            d["distance"] = self.distance
        return d


# ---------------------------------------------------------------------------
# Layout structure
# ---------------------------------------------------------------------------

@dataclass
class LayoutStructure:
    """Hierarchical layout of detected UI elements by screen region."""

    navbar: list[UIDetection] = field(default_factory=list)
    sidebar: list[UIDetection] = field(default_factory=list)
    footer: list[UIDetection] = field(default_factory=list)
    main_content: list[UIDetection] = field(default_factory=list)
    modal: list[UIDetection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: [d.to_dict() for d in getattr(self, name)]
            for name in ("navbar", "sidebar", "footer", "main_content", "modal")
        }


# ---------------------------------------------------------------------------
# Structured context (complete pipeline output)
# ---------------------------------------------------------------------------

@dataclass
class StructuredContext:
    """Complete structured context output from IrpAI pipeline."""

    platform: str = "unknown"
    app_title: str = "Unknown Application"
    layout: LayoutStructure = field(default_factory=LayoutStructure)
    spatial_relationships: list[SpatialRelationship] = field(default_factory=list)
    all_text: str = ""
    text_extractions: list[TextExtraction] = field(default_factory=list)
    ui_detections: list[UIDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate human-readable summary of the context."""
        lines = [
            f"Platform: {self.platform}",
            f"Application: {self.app_title}",
            f"Total UI Elements: {len(self.ui_detections)}",
            f"Total Text Regions: {len(self.text_extractions)}",
        ]

        for region_name in ["navbar", "sidebar", "main_content", "footer", "modal"]:
            elements = getattr(self.layout, region_name)
            if elements:
                elem_descs = []
                for e in elements[:5]:
                    desc = e.element_type.value
                    if e.text:
                        desc += f' ("{e.text[:40]}")'
                    elem_descs.append(desc)
                lines.append(f"  {region_name.upper()}: {', '.join(elem_descs)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "app_title": self.app_title,
            "layout": self.layout.to_dict(),
            "spatial_relationships": [r.to_dict() for r in self.spatial_relationships],
            "all_text": self.all_text,
            "text_extractions": [t.to_dict() for t in self.text_extractions],
            "ui_detections": [d.to_dict() for d in self.ui_detections],
            "metadata": self.metadata,
        }

    def to_json(self, *, compact: bool = True) -> str:
        d = self.to_dict()
        if compact:
            return json.dumps(d, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(d, indent=2, ensure_ascii=False)

    # -- converter from new ContextPack ------------------------------------

    @classmethod
    def from_context_pack(
        cls,
        pack: object,  # irpai.context.schema.ContextPack
    ) -> StructuredContext:
        """
        Create a ``StructuredContext`` from a :class:`~irpai.context.schema.ContextPack`.

        This bridges the new pipeline output format to the legacy model.
        """
        from irpai.context.schema import ContextPack

        if not isinstance(pack, ContextPack):
            raise TypeError(f"Expected ContextPack, got {type(pack).__name__}")

        # Convert regions → UIDetection list
        detections: list[UIDetection] = []
        for region in pack.regions:
            # Map role → UIElementType
            try:
                elem_type = UIElementType(region.role)
            except ValueError:
                elem_type = UIElementType.UNKNOWN

            detections.append(UIDetection(
                bbox=BoundingBox.from_tuple(region.bbox) if region.bbox else BoundingBox(x1=0, y1=0, x2=0, y2=0),
                element_type=elem_type,
                confidence=region.confidence,
                text=region.text or None,
            ))

        # Extract text regions
        text_extractions: list[TextExtraction] = []
        for region in pack.regions:
            if region.text and region.text.strip():
                text_extractions.append(TextExtraction(
                    text=region.text,
                    bbox=BoundingBox.from_tuple(region.bbox) if region.bbox else BoundingBox(x1=0, y1=0, x2=0, y2=0),
                    confidence=region.confidence,
                ))

        # Collect full text
        all_text = " ".join(
            r.text for r in pack.regions if r.text and r.text.strip()
        )

        return cls(
            platform=pack.summary.app_type or "unknown",
            app_title=pack.summary.description[:100] if pack.summary.description else "Unknown Application",
            all_text=all_text,
            text_extractions=text_extractions,
            ui_detections=detections,
            metadata={
                "irpai_version": pack.meta.irpai_version,
                "tier": pack.meta.tier,
                "total_ms": pack.meta.total_ms,
            },
        )


# ---------------------------------------------------------------------------
# Pipeline result (top-level output)
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Final result from the IrpAI pipeline."""

    structured_context: StructuredContext
    enhanced_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    processing_time_ms: float = 0.0
    stage_timings: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_context_pack(
        cls,
        pack: object,  # irpai.context.schema.ContextPack
        *,
        processing_time_ms: float = 0.0,
        confidence: float = 0.0,
    ) -> PipelineResult:
        """
        Create a ``PipelineResult`` from a :class:`~irpai.context.schema.ContextPack`.
        """
        context = StructuredContext.from_context_pack(pack)
        return cls(
            structured_context=context,
            processing_time_ms=processing_time_ms,
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_context": self.structured_context.to_dict(),
            "processing_time_ms": self.processing_time_ms,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }

    def to_json(self, *, compact: bool = True) -> str:
        d = self.to_dict()
        if compact:
            return json.dumps(d, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(d, indent=2, ensure_ascii=False)
