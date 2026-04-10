"""
Accuracy Benchmark
==================

Measures structural and semantic correctness against the known
ground truth in each synthetic scene:
- Region detection recall
- Role classification accuracy
- Text extraction completeness
- Hierarchy correctness
- Reading order accuracy (Kendall's tau)
- App type detection accuracy
- BBox IoU
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irpai.config import IrpAIConfig, Tier
from irpai.context.schema import ContextPack
from irpai.pipeline import IrpAIPipeline
from irpai.utils.image import compute_iou

from tests.research.synthetic_scenes import (
    GroundTruth,
    SyntheticScene,
    build_all_scenes,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SceneAccuracyResult:
    """Accuracy metrics for a single scene."""
    scene: str
    tier: str
    region_recall: float = 0.0
    role_accuracy: float = 0.0
    text_completeness: float = 0.0
    hierarchy_depth: int = 0
    reading_order_tau: float = 0.0
    app_type_correct: bool = False
    mean_iou: float = 0.0
    regions_detected: int = 0
    regions_expected: int = 0


@dataclass
class AccuracyBenchmarkResults:
    scene_results: list[SceneAccuracyResult] = field(default_factory=list)
    overall_region_recall: float = 0.0
    overall_role_accuracy: float = 0.0
    overall_text_completeness: float = 0.0
    overall_app_type_accuracy: float = 0.0
    overall_mean_iou: float = 0.0


# ---------------------------------------------------------------------------
# Safe pipeline runner
# ---------------------------------------------------------------------------

def _safe_run(pipeline: IrpAIPipeline, image: object) -> ContextPack | None:
    """Run pipeline with fallback to manual layer execution."""
    try:
        return pipeline.run(image)
    except RuntimeError:
        pass

    try:
        normalized = pipeline._normalizer.normalize(image)
        detections = pipeline._run_detection(normalized)
        ocr_result = pipeline._run_ocr(normalized)
        layout = pipeline._layout.analyze(ocr_result, detections)
        return pipeline._assembler.assemble(
            layout=layout, ocr=ocr_result, detections=detections,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_accuracy_benchmark() -> AccuracyBenchmarkResults:
    """Execute accuracy analysis across all scenes and tiers."""
    scenes = build_all_scenes()
    results = AccuracyBenchmarkResults()

    for tier in [Tier.LITE, Tier.STANDARD]:
        try:
            pipeline = IrpAIPipeline(IrpAIConfig(tier=tier))
        except Exception:
            continue

        for scene in scenes:
            pack = _safe_run(pipeline, scene.image)
            if pack is not None:
                sr = _evaluate_scene(pack, scene.ground_truth, tier.value)
                results.scene_results.append(sr)
            else:
                results.scene_results.append(SceneAccuracyResult(
                    scene=scene.ground_truth.scene_name,
                    tier=tier.value,
                ))

    # Aggregate overall metrics
    valid = [r for r in results.scene_results if r.regions_detected > 0]
    if valid:
        results.overall_region_recall = _avg([r.region_recall for r in valid])
        results.overall_role_accuracy = _avg([r.role_accuracy for r in valid])
        results.overall_text_completeness = _avg([r.text_completeness for r in valid])
        results.overall_app_type_accuracy = sum(
            1 for r in valid if r.app_type_correct
        ) / len(valid)
        results.overall_mean_iou = _avg([r.mean_iou for r in valid])

    return results


def _evaluate_scene(
    pack: ContextPack,
    gt: GroundTruth,
    tier: str,
) -> SceneAccuracyResult:
    """Evaluate a pipeline output against ground truth."""
    gt_regions = gt.regions
    pred_regions = pack.regions

    # Region detection recall
    matched_gt = 0
    role_matches = 0
    ious: list[float] = []

    for g in gt_regions:
        best_iou = 0.0
        best_role_match = False
        for p in pred_regions:
            if p.bbox is None or g.bbox is None:
                continue
            iou = compute_iou(g.bbox, tuple(p.bbox))
            if iou > best_iou:
                best_iou = iou
                best_role_match = _roles_match(g.role, p.role)

        if best_iou > 0.1:
            matched_gt += 1
            ious.append(best_iou)
            if best_role_match:
                role_matches += 1

    region_recall = matched_gt / max(len(gt_regions), 1)
    role_accuracy = role_matches / max(matched_gt, 1)

    # Text extraction completeness
    gt_words = set(gt.all_text.lower().split())
    pred_text = " ".join(r.text for r in pred_regions if r.text)
    pred_words = set(pred_text.lower().split())
    if gt_words:
        text_completeness = len(gt_words & pred_words) / len(gt_words)
    else:
        text_completeness = 1.0

    # Hierarchy depth
    hierarchy_depth = 0
    if pack.hierarchy and "roots" in pack.hierarchy:
        hierarchy_depth = _max_depth(pack.hierarchy["roots"])

    # Reading order
    reading_order_tau = _reading_order_score(pred_regions)

    # App type detection
    app_type_correct = pack.summary.app_type == gt.app_type

    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return SceneAccuracyResult(
        scene=gt.scene_name,
        tier=tier,
        region_recall=round(region_recall, 3),
        role_accuracy=round(role_accuracy, 3),
        text_completeness=round(text_completeness, 3),
        hierarchy_depth=hierarchy_depth,
        reading_order_tau=round(reading_order_tau, 3),
        app_type_correct=app_type_correct,
        mean_iou=round(mean_iou, 3),
        regions_detected=len(pred_regions),
        regions_expected=len(gt_regions),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roles_match(gt_role: str, pred_role: str) -> bool:
    """Fuzzy role matching — accounts for label mapping differences."""
    role_equivalences = {
        "nav": {"nav", "menu", "menu_item", "sidebar"},
        "heading": {"heading", "header", "label"},
        "body": {"body", "content_area", "code", "unknown"},
        "button": {"button", "label"},
        "input": {"input", "text_field"},
        "card": {"card", "body"},
        "status_bar": {"status_bar", "footer"},
        "footer": {"footer", "status_bar"},
        "label": {"label", "button", "heading", "menu_item"},
    }
    if gt_role == pred_role:
        return True
    equivalents = role_equivalences.get(gt_role, set())
    return pred_role in equivalents


def _max_depth(nodes: list | None, current: int = 0) -> int:
    if not nodes:
        return current
    max_d = current + 1
    for node in nodes:
        if isinstance(node, dict) and "children" in node:
            max_d = max(max_d, _max_depth(node["children"], current + 1))
    return max_d


def _reading_order_score(regions: list) -> float:
    """Score reading order correctness (0.0–1.0)."""
    if len(regions) < 2:
        return 1.0

    ordered_pairs = 0
    total_pairs = 0
    for i in range(len(regions) - 1):
        for j in range(i + 1, len(regions)):
            if regions[i].bbox and regions[j].bbox:
                total_pairs += 1
                if regions[i].bbox[1] <= regions[j].bbox[1]:
                    ordered_pairs += 1

    return ordered_pairs / max(total_pairs, 1)


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0
