"""
Hierarchy Builder

Converts flat :class:`SemanticRegion` lists into a parent→child tree
by spatial containment analysis.  This produces the ``hierarchy`` field
in the final context pack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irpai.layout.analyzer import SemanticRegion
from irpai.utils.image import BBox, box_area, box_center
from irpai.utils.logging import get_logger

log = get_logger("layout.hierarchy")


@dataclass
class LayoutNode:
    """
    A node in the layout hierarchy tree.

    Each node wraps a :class:`SemanticRegion` and may have child nodes
    for regions that are spatially contained within it.
    """

    region: SemanticRegion
    children: list[LayoutNode] = field(default_factory=list)
    depth: int = 0

    @property
    def role(self) -> str:
        return self.region.role

    @property
    def text(self) -> str:
        return self.region.text

    @property
    def bbox(self) -> BBox:
        return self.region.bbox

    def flatten(self) -> list[LayoutNode]:
        """Depth-first flattening of the tree."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.flatten())
        return nodes

    def to_dict(self) -> dict:
        """Serialise to a nested dictionary (for JSON output)."""
        d: dict = {
            "role": self.role,
            "bbox": list(self.bbox),
        }
        if self.text:
            d["text"] = self.text
        if self.region.confidence < 1.0:
            d["confidence"] = round(self.region.confidence, 3)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


class HierarchyBuilder:
    """
    Builds a layout tree from a flat list of :class:`SemanticRegion` s.

    Algorithm:
        1. Sort regions by area (largest first).
        2. For each region, check if it is *contained* within any
           already-placed region.
        3. If so, attach it as a child of the smallest enclosing parent.
        4. Otherwise, add it as a root node.

    Containment is defined as: the child's centre point falls inside the
    parent's bounding box AND the parent's area is ≥ 1.5× the child's.
    """

    def __init__(self, containment_ratio: float = 1.5) -> None:
        self.containment_ratio = containment_ratio

    def build(self, regions: list[SemanticRegion]) -> list[LayoutNode]:
        """
        Build the hierarchy tree.

        Parameters
        ----------
        regions:
            Flat list of semantic regions from layout analysis.

        Returns
        -------
        list[LayoutNode]
            Root-level nodes of the layout tree.
        """
        if not regions:
            return []

        # Sort by area descending (largest regions first → they become parents)
        sorted_regions = sorted(regions, key=lambda r: box_area(r.bbox), reverse=True)

        roots: list[LayoutNode] = []
        all_nodes: list[LayoutNode] = []

        for region in sorted_regions:
            node = LayoutNode(region=region)

            # Find the smallest existing node that contains this one
            parent = self._find_parent(node, all_nodes)
            if parent is not None:
                node.depth = parent.depth + 1
                parent.children.append(node)
            else:
                roots.append(node)

            all_nodes.append(node)

        log.info(
            "Hierarchy: %d roots, %d total nodes, max depth %d",
            len(roots),
            len(all_nodes),
            max((n.depth for n in all_nodes), default=0),
        )

        return roots

    def _find_parent(
        self,
        child: LayoutNode,
        candidates: list[LayoutNode],
    ) -> LayoutNode | None:
        """
        Find the smallest candidate that spatially contains *child*.
        """
        child_center = box_center(child.bbox)
        child_area = box_area(child.bbox)

        best: LayoutNode | None = None
        best_area = float("inf")

        for candidate in candidates:
            cand_area = box_area(candidate.bbox)

            # Parent must be meaningfully larger
            if cand_area < child_area * self.containment_ratio:
                continue

            # Child's centre must be inside parent's box
            x1, y1, x2, y2 = candidate.bbox
            cx, cy = child_center
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue

            # Prefer the smallest valid parent
            if cand_area < best_area:
                best = candidate
                best_area = cand_area

        return best
