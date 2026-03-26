"""
Context Assembler

Takes the outputs of all pipeline layers (preprocessing metadata,
detection regions, OCR text, layout analysis) and assembles them
into a :class:`ContextPack`.
"""

from __future__ import annotations

import re
from collections import Counter

from irpai.config import OutputConfig
from irpai.context.schema import (
    ContextPack,
    PipelineMetadata,
    Region,
    ScreenSummary,
)
from irpai.detection.base import DetectionOutput
from irpai.layout.analyzer import LayoutResult, SemanticRegion
from irpai.layout.hierarchy import HierarchyBuilder, LayoutNode
from irpai.ocr.base import OCRResult
from irpai.utils.logging import get_logger

log = get_logger("context")


# Known app-type detection patterns (checked against full OCR text)
_APP_PATTERNS: list[tuple[str, str]] = [
    (r"Visual Studio Code|\.py|\.ts|\.js|def |class |import ", "ide"),
    (r"https?://|\.com|\.org|Search|Tab|Bookmark", "browser"),
    (r"\$\s|bash|zsh|fish|>>>|terminal", "terminal"),
    (r"Slack|Discord|Teams|Messages|Chat|Send", "chat"),
    (r"Mail|Inbox|Subject|From:|To:", "email"),
    (r"Finder|Explorer|Desktop|Documents|Downloads", "file_manager"),
    (r"Settings|Preferences|Configuration|Options", "settings"),
    (r"Figma|Sketch|Adobe|Canvas|Layer", "design"),
    (r"Sheet|Cell|Row|Column|SUM|VLOOKUP", "spreadsheet"),
    (r"Slide|Presentation|Speaker Notes", "presentation"),
]


class ContextAssembler:
    """
    Assembles a :class:`ContextPack` from pipeline layer outputs.

    This is the final stage of the IrpAI pipeline — it merges all
    extracted information into a single structured object ready for
    LLM prompt injection.
    """

    def __init__(self, config: OutputConfig | None = None) -> None:
        self.cfg = config or OutputConfig()
        self._hierarchy_builder = HierarchyBuilder()

    def assemble(
        self,
        layout: LayoutResult,
        ocr: OCRResult | None = None,
        detections: DetectionOutput | None = None,
        metadata: PipelineMetadata | None = None,
    ) -> ContextPack:
        """
        Build the context pack.

        Parameters
        ----------
        layout:
            Layout analysis result (required — this is the primary source).
        ocr:
            Raw OCR result (for full-text access and confidence data).
        detections:
            Raw detection output (for element counts and model info).
        metadata:
            Pipeline run metadata (versions, timing, etc.).
        """
        # -- Regions --------------------------------------------------------
        regions = self._layout_to_regions(layout)

        # -- Hierarchy ------------------------------------------------------
        hierarchy: dict = {}
        if self.cfg.include_hierarchy:
            tree = self._hierarchy_builder.build(layout.regions)
            hierarchy = {"roots": [node.to_dict() for node in tree]}

        # -- Summary --------------------------------------------------------
        summary = self._build_summary(layout, ocr)

        # -- Assemble -------------------------------------------------------
        pack = ContextPack(
            summary=summary,
            regions=regions,
            hierarchy=hierarchy,
            meta=metadata or PipelineMetadata(),
        )

        # -- Token budget enforcement --------------------------------------
        self._enforce_token_budget(pack)

        log.info(
            "Context pack: %d regions, ~%d tokens",
            len(pack.regions),
            pack.token_estimate(),
        )
        return pack

    # -- helpers ------------------------------------------------------------

    def _layout_to_regions(self, layout: LayoutResult) -> list[Region]:
        """Convert SemanticRegions → schema Regions in reading order."""
        ordered_indices = layout.reading_order or list(range(len(layout.regions)))
        regions: list[Region] = []

        for idx in ordered_indices:
            if idx >= len(layout.regions):
                continue
            sr = layout.regions[idx]
            regions.append(self._semantic_to_region(sr))

        return regions

    def _semantic_to_region(self, sr: SemanticRegion) -> Region:
        """Recursively convert a SemanticRegion to a schema Region."""
        children = [self._semantic_to_region(c) for c in sr.children]
        return Region(
            role=sr.role,
            text=sr.text.strip(),
            bbox=sr.bbox if self.cfg.include_bboxes else None,
            confidence=sr.confidence if self.cfg.include_confidence else 1.0,
            children=children,
            properties=dict(sr.meta) if sr.meta else {},
        )

    def _build_summary(
        self,
        layout: LayoutResult,
        ocr: OCRResult | None,
    ) -> ScreenSummary:
        """Generate the natural-language screen summary."""
        full_text = layout.full_text or (ocr.full_text if ocr else "")
        app_type = self._detect_app_type(full_text)

        # Key elements: most common roles
        role_counts = Counter(r.role for r in layout.regions if r.role != "unknown")
        key_elements = [
            f"{count}× {role}" for role, count in role_counts.most_common(8)
        ]

        # Description
        description = self._generate_description(layout, app_type)

        return ScreenSummary(
            description=description,
            app_type=app_type,
            key_elements=key_elements,
        )

    @staticmethod
    def _detect_app_type(text: str) -> str:
        """Heuristic app-type detection from OCR text."""
        for pattern, app_type in _APP_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return app_type
        return "unknown"

    @staticmethod
    def _generate_description(
        layout: LayoutResult,
        app_type: str,
    ) -> str:
        """Build a one-paragraph description of the screen."""
        parts: list[str] = []

        if app_type != "unknown":
            parts.append(f"Application type: {app_type}.")

        n_text = len(layout.text_regions)
        n_total = len(layout.regions)
        parts.append(f"Contains {n_total} regions ({n_text} with text).")

        # Headings
        headings = [r.text.strip() for r in layout.regions if r.role == "heading" and r.text]
        if headings:
            parts.append("Headings: " + "; ".join(headings[:5]) + ".")

        return " ".join(parts)

    def _enforce_token_budget(self, pack: ContextPack) -> None:
        """
        Trim region text if the pack exceeds the token budget.

        Strategy: truncate the longest text regions first, keeping
        headings and labels intact.
        """
        budget = self.cfg.max_context_tokens
        current = pack.token_estimate()

        if current <= budget:
            return

        log.warning(
            "Context pack (%d tokens) exceeds budget (%d), truncating",
            current,
            budget,
        )

        # Sort regions by text length (longest first) — truncate those
        sortable = [(i, r) for i, r in enumerate(pack.regions) if r.role in ("body", "unknown")]
        sortable.sort(key=lambda t: len(t[1].text), reverse=True)

        for idx, region in sortable:
            if pack.token_estimate() <= budget:
                break
            # Truncate to half
            text = region.text
            if len(text) > 100:
                pack.regions[idx] = Region(
                    role=region.role,
                    text=text[: len(text) // 2] + " [truncated]",
                    bbox=region.bbox,
                    confidence=region.confidence,
                    children=region.children,
                    properties=region.properties,
                )

