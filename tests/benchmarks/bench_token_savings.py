"""
Token Savings Benchmark
=======================

Quantifies IrpAI's context compression efficiency:
- Raw-to-context compression ratio
- Format comparison (JSON vs Markdown vs system_prompt)
- Token budget compliance
- Semantic density (roles per token)
- Hierarchy overhead
- BBox omission savings
"""

from __future__ import annotations

from dataclasses import dataclass, field

from irpai.config import IrpAIConfig, OutputConfig, Tier
from irpai.context.formatter import format_for_llm
from irpai.context.schema import ContextPack
from irpai.pipeline import IrpAIPipeline

from tests.research.synthetic_scenes import SyntheticScene, build_all_scenes


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FormatComparison:
    """Token counts across output formats."""

    scene: str
    json_tokens: int = 0
    markdown_tokens: int = 0
    system_prompt_tokens: int = 0
    json_chars: int = 0
    markdown_chars: int = 0
    system_prompt_chars: int = 0


@dataclass
class SceneTokenResult:
    """Token analysis for a single scene."""

    scene: str
    raw_text_chars: int = 0
    context_json_chars: int = 0
    compression_ratio: float = 0.0
    context_tokens: int = 0
    within_budget: bool = False
    semantic_density: float = 0.0
    hierarchy_tokens: int = 0
    no_hierarchy_tokens: int = 0
    hierarchy_overhead_pct: float = 0.0
    with_bbox_tokens: int = 0
    no_bbox_tokens: int = 0
    bbox_savings_pct: float = 0.0


@dataclass
class TokenSavingsBenchmarkResults:
    scene_results: list[SceneTokenResult] = field(default_factory=list)
    format_comparisons: list[FormatComparison] = field(default_factory=list)
    overall_compression_ratio: float = 0.0
    budget_compliance_pct: float = 0.0
    avg_semantic_density: float = 0.0


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
            layout=layout,
            ocr=ocr_result,
            detections=detections,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET = 4096


def run_token_savings_benchmark() -> TokenSavingsBenchmarkResults:
    """Execute the token savings benchmark."""
    scenes = build_all_scenes()
    results = TokenSavingsBenchmarkResults()

    config = IrpAIConfig(tier=Tier.LITE)
    try:
        pipeline = IrpAIPipeline(config)
    except Exception:
        return results

    for scene in scenes:
        sr, fc = _analyze_scene(pipeline, scene)
        if sr:
            results.scene_results.append(sr)
        if fc:
            results.format_comparisons.append(fc)

    # Aggregates
    if results.scene_results:
        results.overall_compression_ratio = round(
            sum(r.compression_ratio for r in results.scene_results) / len(results.scene_results),
            1,
        )
        within = sum(1 for r in results.scene_results if r.within_budget)
        results.budget_compliance_pct = round(within / len(results.scene_results) * 100, 1)
        results.avg_semantic_density = round(
            sum(r.semantic_density for r in results.scene_results) / len(results.scene_results),
            2,
        )

    return results


def _analyze_scene(
    pipeline: IrpAIPipeline,
    scene: SyntheticScene,
) -> tuple[SceneTokenResult | None, FormatComparison | None]:
    """Analyze token metrics for a single scene."""
    gt = scene.ground_truth
    name = gt.scene_name

    pack = _safe_run(pipeline, scene.image)
    if pack is None:
        return None, None

    # Raw text from ground truth
    raw_chars = len(gt.all_text)

    # Context output sizes
    json_out = pack.to_json(compact=True)
    json_tokens = len(json_out) // 4

    # Format comparison
    md_out = format_for_llm(pack, style="markdown")
    sp_out = format_for_llm(pack, style="system_prompt")

    fc = FormatComparison(
        scene=name,
        json_tokens=json_tokens,
        markdown_tokens=len(md_out) // 4,
        system_prompt_tokens=len(sp_out) // 4,
        json_chars=len(json_out),
        markdown_chars=len(md_out),
        system_prompt_chars=len(sp_out),
    )

    # Compression ratio
    compression_ratio = raw_chars / max(len(json_out), 1)

    # Semantic density
    unique_roles = len(set(r.role for r in pack.regions))
    semantic_density = (unique_roles / max(json_tokens, 1)) * 1000

    # Hierarchy overhead
    json_with_hierarchy = pack.to_json(compact=True, include_hierarchy=True)
    json_no_hierarchy = pack.to_json(compact=True, include_hierarchy=False)
    h_tokens = len(json_with_hierarchy) // 4
    nh_tokens = len(json_no_hierarchy) // 4
    h_overhead = ((h_tokens - nh_tokens) / max(nh_tokens, 1)) * 100

    # BBox savings
    json_with_bbox = pack.to_json(compact=True, include_bboxes=True)
    json_no_bbox = pack.to_json(compact=True, include_bboxes=False)
    bb_tokens = len(json_with_bbox) // 4
    nbb_tokens = len(json_no_bbox) // 4
    bb_savings = ((bb_tokens - nbb_tokens) / max(bb_tokens, 1)) * 100

    sr = SceneTokenResult(
        scene=name,
        raw_text_chars=raw_chars,
        context_json_chars=len(json_out),
        compression_ratio=round(compression_ratio, 2),
        context_tokens=json_tokens,
        within_budget=json_tokens <= DEFAULT_TOKEN_BUDGET,
        semantic_density=round(semantic_density, 2),
        hierarchy_tokens=h_tokens,
        no_hierarchy_tokens=nh_tokens,
        hierarchy_overhead_pct=round(h_overhead, 1),
        with_bbox_tokens=bb_tokens,
        no_bbox_tokens=nbb_tokens,
        bbox_savings_pct=round(bb_savings, 1),
    )

    return sr, fc
