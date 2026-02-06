"""
IrpAI Data Models

Pydantic models for structured data flowing through the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class UIElementType(str, Enum):
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
    NAVBAR = "navbar"
    SIDEBAR = "sidebar"
    FOOTER = "footer"
    MAIN_CONTENT = "main_content"
    MODAL = "modal"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: BoundingBox) -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: list[float]) -> BoundingBox:
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


class UIDetection(BaseModel):
    """A detected UI element."""

    bbox: BoundingBox
    element_type: UIElementType = UIElementType.UNKNOWN
    confidence: float = 0.0
    text_region: bool = False
    text: Optional[str] = None
    text_confidence: Optional[float] = None
    icon_type: Optional[str] = None
    label: Optional[str] = None
    orphaned: bool = False


class TextExtraction(BaseModel):
    """An extracted text region from OCR."""

    text: str
    bbox: BoundingBox
    confidence: float = 0.0
    ui_region: Optional[str] = None
    language: Optional[str] = None


class SpatialRelationship(BaseModel):
    """Spatial relationship between two UI elements."""

    subject: str
    relation: str  # above, below, left_of, right_of, inside, overlaps
    obj: str  # "object" is reserved in Python
    distance: Optional[float] = None


class LayoutStructure(BaseModel):
    """Hierarchical layout of detected UI elements."""

    navbar: list[UIDetection] = Field(default_factory=list)
    sidebar: list[UIDetection] = Field(default_factory=list)
    footer: list[UIDetection] = Field(default_factory=list)
    main_content: list[UIDetection] = Field(default_factory=list)
    modal: list[UIDetection] = Field(default_factory=list)


class StructuredContext(BaseModel):
    """Complete structured context output from IrpAI pipeline."""

    platform: str = "unknown"
    app_title: str = "Unknown Application"
    layout: LayoutStructure = Field(default_factory=LayoutStructure)
    spatial_relationships: list[SpatialRelationship] = Field(default_factory=list)
    all_text: str = ""
    text_extractions: list[TextExtraction] = Field(default_factory=list)
    ui_detections: list[UIDetection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

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


class PipelineResult(BaseModel):
    """Final result from the IrpAI pipeline."""

    structured_context: StructuredContext
    enhanced_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    processing_time_ms: float = 0.0
    stage_timings: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
