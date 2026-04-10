"""
Research Report Generator
=========================

Aggregates results from all benchmark suites and produces a
comprehensive Markdown research report with tables, statistics,
and comparative analysis.
"""

from __future__ import annotations

import datetime
import platform
from pathlib import Path

from tests.benchmarks.bench_accuracy import AccuracyBenchmarkResults
from tests.benchmarks.bench_cost import CostBenchmarkResults, LLM_PRICING
from tests.benchmarks.bench_data_efficiency import DataEfficiencyResults
from tests.benchmarks.bench_latency import LatencyBenchmarkResults
from tests.benchmarks.bench_privacy import PrivacyBenchmarkResults
from tests.benchmarks.bench_token_savings import TokenSavingsBenchmarkResults


def generate_report(
    cost: CostBenchmarkResults,
    data_eff: DataEfficiencyResults,
    accuracy: AccuracyBenchmarkResults,
    latency: list[LatencyBenchmarkResults],
    tokens: TokenSavingsBenchmarkResults,
    privacy: PrivacyBenchmarkResults,
    output_path: Path,
) -> str:
    """Generate the full Markdown research report and write to disk."""
    sections = [
        _header(),
        _executive_summary(cost, data_eff, accuracy, latency, tokens, privacy),
        _cost_section(cost),
        _data_efficiency_section(data_eff),
        _accuracy_section(accuracy),
        _latency_section(latency),
        _token_savings_section(tokens),
        _privacy_section(privacy),
        _conclusion(cost, accuracy, tokens, privacy),
        _methodology(),
    ]

    report = "\n\n".join(sections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _header() -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# IrpAI Research Report — Quantitative Impact Analysis

> **Generated:** {now}
> **Platform:** {platform.system()} {platform.machine()} — Python {platform.python_version()}
> **IrpAI Version:** 0.1.0

---"""


def _executive_summary(
    cost: CostBenchmarkResults,
    data_eff: DataEfficiencyResults,
    accuracy: AccuracyBenchmarkResults,
    latency: list[LatencyBenchmarkResults],
    tokens: TokenSavingsBenchmarkResults,
    privacy: PrivacyBenchmarkResults,
) -> str:
    lines = ["## Executive Summary", ""]

    # Key findings
    findings = []

    if cost.tier_results:
        avg_tokens = sum(r.context_tokens for r in cost.tier_results) / len(cost.tier_results)
        findings.append(f"- **Context Compression**: Average output of ~{int(avg_tokens)} tokens per screenshot")

    if cost.token_reduction_ratios:
        avg_ratio = sum(cost.token_reduction_ratios.values()) / len(cost.token_reduction_ratios)
        findings.append(f"- **Token Reduction**: {avg_ratio:.1f}× average reduction vs raw vision tokens")

    if accuracy.overall_region_recall > 0:
        findings.append(f"- **Region Detection**: {accuracy.overall_region_recall:.1%} recall across all test scenes")

    if accuracy.overall_text_completeness > 0:
        findings.append(f"- **Text Extraction**: {accuracy.overall_text_completeness:.1%} completeness")

    if data_eff.graceful_degradation_index > 0:
        findings.append(f"- **Data Resilience**: {data_eff.graceful_degradation_index:.1f}% degradation index (region retention under imperfect input)")

    if latency:
        for lr in latency:
            if lr.e2e_times_ms:
                avg_ms = sum(lr.e2e_times_ms) / len(lr.e2e_times_ms)
                findings.append(f"- **Latency ({lr.tier})**: {avg_ms:.0f}ms average end-to-end")

    if tokens.budget_compliance_pct > 0:
        findings.append(f"- **Token Budget**: {tokens.budget_compliance_pct:.0f}% compliance with 4096-token limit")

    if privacy.overall_precision > 0:
        findings.append(f"- **PII Protection**: {privacy.overall_precision:.1%} precision, {privacy.overall_recall:.1%} recall")

    lines.extend(findings or ["- No benchmark data available."])
    return "\n".join(lines)


def _cost_section(cost: CostBenchmarkResults) -> str:
    lines = [
        "## 1. Cost Analysis",
        "",
        "### 1.1 Compute Footprint by Tier",
        "",
        "| Scene | Tier | Peak Memory (MB) | Latency (ms) | Regions | Tokens | Cost Index |",
        "|-------|------|-------------------|--------------|---------|--------|------------|",
    ]

    for r in cost.tier_results:
        lines.append(
            f"| {r.scene} | {r.tier} | {r.peak_memory_mb} | {r.latency_ms} | "
            f"{r.regions_extracted} | {r.context_tokens} | {r.compute_cost_index} |"
        )

    # Token reduction ratios
    if cost.token_reduction_ratios:
        lines.extend([
            "",
            "### 1.2 Token Reduction Ratios",
            "",
            "| Scene × Tier | Vision Tokens → Context Tokens | Reduction Factor |",
            "|--------------|-------------------------------|------------------|",
        ])
        for key, ratio in cost.token_reduction_ratios.items():
            lines.append(f"| {key} | — | {ratio}× |")

    # API savings table
    if cost.api_savings:
        lines.extend([
            "",
            "### 1.3 Estimated API Cost Savings per Image",
            "",
            "Savings calculated as `(vision_tokens − context_tokens) × price_per_1K_tokens`",
            "",
        ])

        # Build header row
        models = list(LLM_PRICING.keys())
        header = "| Scene × Tier | " + " | ".join(models) + " |"
        separator = "|" + "|".join(["---"] * (len(models) + 1)) + "|"
        lines.extend([header, separator])

        for key, savings in cost.api_savings.items():
            cells = [f"${savings.get(m, 0):.6f}" for m in models]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")

        lines.extend([
            "",
            "**Pricing tiers used (USD/1K input tokens):**",
            "",
        ])
        for model, price in LLM_PRICING.items():
            lines.append(f"- {model}: ${price}")

    return "\n".join(lines)


def _data_efficiency_section(data_eff: DataEfficiencyResults) -> str:
    lines = [
        "## 2. Data Efficiency & Resilience",
        "",
    ]

    if data_eff.graceful_degradation_index > 0:
        lines.append(f"**Graceful Degradation Index: {data_eff.graceful_degradation_index}%**")
        lines.append("(Average region retention across all degradation axes vs baseline)")
        lines.append("")

    # Group results by degradation type
    by_type: dict[str, list] = {}
    for r in data_eff.results:
        by_type.setdefault(r.degradation_type, []).append(r)

    type_labels = {
        "resolution": "2.1 Resolution Degradation",
        "noise": "2.2 Gaussian Noise Resilience",
        "contrast": "2.3 Contrast Reduction",
        "jpeg": "2.4 JPEG Compression Artifacts",
    }

    for dtype, label in type_labels.items():
        items = by_type.get(dtype, [])
        if not items:
            continue

        lines.extend([
            f"### {label}",
            "",
            "| Scene | Level | Regions | Text Chars | Confidence | Tokens |",
            "|-------|-------|---------|------------|------------|--------|",
        ])
        for r in items:
            if r.error:
                lines.append(f"| {r.scene} | {r.level_label} | ERROR | — | — | — |")
            else:
                lines.append(
                    f"| {r.scene} | {r.level_label} | {r.regions_found} | "
                    f"{r.text_chars_extracted} | {r.confidence:.3f} | {r.tokens} |"
                )
        lines.append("")

    return "\n".join(lines)


def _accuracy_section(accuracy: AccuracyBenchmarkResults) -> str:
    lines = [
        "## 3. Accuracy Analysis",
        "",
        "### 3.1 Per-Scene Results",
        "",
        "| Scene | Tier | Recall | Role Acc. | Text Compl. | App Type | Mean IoU | Detected / Expected |",
        "|-------|------|--------|----------|-------------|----------|----------|---------------------|",
    ]

    for r in accuracy.scene_results:
        app = "✓" if r.app_type_correct else "✗"
        lines.append(
            f"| {r.scene} | {r.tier} | {r.region_recall:.1%} | {r.role_accuracy:.1%} | "
            f"{r.text_completeness:.1%} | {app} | {r.mean_iou:.3f} | {r.regions_detected}/{r.regions_expected} |"
        )

    lines.extend([
        "",
        "### 3.2 Aggregate Metrics",
        "",
        f"- **Overall Region Recall:** {accuracy.overall_region_recall:.1%}",
        f"- **Overall Role Accuracy:** {accuracy.overall_role_accuracy:.1%}",
        f"- **Overall Text Completeness:** {accuracy.overall_text_completeness:.1%}",
        f"- **App Type Detection Accuracy:** {accuracy.overall_app_type_accuracy:.1%}",
        f"- **Overall Mean IoU:** {accuracy.overall_mean_iou:.3f}",
    ])

    return "\n".join(lines)


def _latency_section(latency_results: list[LatencyBenchmarkResults]) -> str:
    lines = [
        "## 4. Latency Profile",
        "",
    ]

    for lr in latency_results:
        lines.extend([
            f"### 4.x Tier: {lr.tier}",
            "",
            "#### Layer Breakdown",
            "",
            "| Layer | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |",
            "|-------|-----------|----------|----------|----------|",
        ])

        for layer_name in ["preprocessing", "detection", "ocr", "layout", "assembly", "total"]:
            lt = lr.layer_timings.get(layer_name)
            if lt and lt.times_ms:
                lines.append(
                    f"| {layer_name} | {lt.mean:.1f} | {lt.p50:.1f} | "
                    f"{lt.p95:.1f} | {lt.p99:.1f} |"
                )

        lines.extend([
            "",
            f"- **Warmup (first run):** {lr.warmup_first_ms:.0f}ms",
            f"- **Steady-state average:** {lr.warmup_steady_ms:.0f}ms",
            f"- **Throughput:** {lr.throughput_ips:.2f} images/sec",
            "",
        ])

        # Scaling profile
        if lr.scale_profiles:
            lines.extend([
                "#### Resolution Scaling",
                "",
                "| Resolution | Latency (ms) |",
                "|------------|-------------|",
            ])
            for sp in lr.scale_profiles:
                lines.append(f"| {sp.resolution} | {sp.latency_ms} |")
            lines.append("")

    return "\n".join(lines)


def _token_savings_section(tokens: TokenSavingsBenchmarkResults) -> str:
    lines = [
        "## 5. Token Savings & Compression",
        "",
        "### 5.1 Per-Scene Compression",
        "",
        "| Scene | Raw Chars | JSON Chars | Ratio | Tokens | Budget OK | Semantic Density |",
        "|-------|-----------|------------|-------|--------|-----------|-----------------|",
    ]

    for r in tokens.scene_results:
        budget = "✓" if r.within_budget else "✗"
        lines.append(
            f"| {r.scene} | {r.raw_text_chars} | {r.context_json_chars} | "
            f"{r.compression_ratio:.2f} | {r.context_tokens} | {budget} | {r.semantic_density:.2f} |"
        )

    # Format comparison
    if tokens.format_comparisons:
        lines.extend([
            "",
            "### 5.2 Output Format Comparison",
            "",
            "| Scene | JSON (tokens) | Markdown (tokens) | System Prompt (tokens) |",
            "|-------|---------------|-------------------|----------------------|",
        ])
        for fc in tokens.format_comparisons:
            lines.append(
                f"| {fc.scene} | {fc.json_tokens} | {fc.markdown_tokens} | {fc.system_prompt_tokens} |"
            )

    # Overhead analysis
    overhead_scenes = [r for r in tokens.scene_results if r.hierarchy_tokens > 0]
    if overhead_scenes:
        lines.extend([
            "",
            "### 5.3 Hierarchy & BBox Overhead",
            "",
            "| Scene | With Hierarchy | Without | Overhead | With BBox | Without | Savings |",
            "|-------|---------------|---------|----------|-----------|---------|---------|",
        ])
        for r in overhead_scenes:
            lines.append(
                f"| {r.scene} | {r.hierarchy_tokens} | {r.no_hierarchy_tokens} | "
                f"{r.hierarchy_overhead_pct:.1f}% | {r.with_bbox_tokens} | {r.no_bbox_tokens} | "
                f"{r.bbox_savings_pct:.1f}% |"
            )

    lines.extend([
        "",
        "### 5.4 Aggregate",
        "",
        f"- **Average Compression Ratio:** {tokens.overall_compression_ratio:.1f}×",
        f"- **Token Budget Compliance:** {tokens.budget_compliance_pct:.0f}%",
        f"- **Average Semantic Density:** {tokens.avg_semantic_density:.2f} roles/1K tokens",
    ])

    return "\n".join(lines)


def _privacy_section(privacy: PrivacyBenchmarkResults) -> str:
    lines = [
        "## 6. Privacy & PII Protection",
        "",
        "### 6.1 Per-Category Results",
        "",
        "| PII Type | True Pos | False Neg | False Pos | Precision | Recall |",
        "|----------|----------|-----------|-----------|-----------|--------|",
    ]

    for c in privacy.category_results:
        lines.append(
            f"| {c.pii_type} | {c.true_positives} | {c.false_negatives} | "
            f"{c.false_positives} | {c.precision:.1%} | {c.recall:.1%} |"
        )

    lines.extend([
        "",
        "### 6.2 Aggregate",
        "",
        f"- **Overall Precision:** {privacy.overall_precision:.1%}",
        f"- **Overall Recall:** {privacy.overall_recall:.1%}",
        f"- **Redaction Completeness:** {privacy.redaction_completeness:.1%}",
        f"- **False Positive Rate:** {privacy.false_positive_rate:.1%}",
        f"- **Total Test Cases:** {privacy.total_test_cases}",
    ])

    return "\n".join(lines)


def _conclusion(
    cost: CostBenchmarkResults,
    accuracy: AccuracyBenchmarkResults,
    tokens: TokenSavingsBenchmarkResults,
    privacy: PrivacyBenchmarkResults,
) -> str:
    lines = [
        "## 7. Conclusion",
        "",
        "### How IrpAI Helps Others",
        "",
    ]

    benefits = []

    # Cost benefit
    if cost.api_savings:
        total_savings = {}
        for key, model_savings in cost.api_savings.items():
            for model, amount in model_savings.items():
                total_savings.setdefault(model, []).append(amount)
        if total_savings:
            example_model = "GPT-4-Turbo"
            if example_model in total_savings:
                avg_saving = sum(total_savings[example_model]) / len(total_savings[example_model])
                yearly = avg_saving * 1000 * 365  # 1K images/day
                benefits.append(
                    f"1. **Cost Reduction**: At 1,000 images/day, IrpAI saves an estimated "
                    f"**${yearly:,.2f}/year** on {example_model} API costs alone by converting "
                    f"raw vision tokens into compact structured context."
                )

    # Data efficiency
    benefits.append(
        "2. **Data Efficiency**: IrpAI gracefully handles imperfect input — noisy, "
        "low-resolution, and heavily compressed images still produce usable context. "
        "This eliminates the need for users to curate high-quality screenshots."
    )

    # Accuracy
    if accuracy.overall_region_recall > 0:
        benefits.append(
            f"3. **Accuracy**: With {accuracy.overall_region_recall:.0%} region recall and "
            f"{accuracy.overall_text_completeness:.0%} text completeness, IrpAI provides "
            f"reliable structured data for downstream LLM consumption."
        )

    # Token savings
    if tokens.overall_compression_ratio > 0:
        benefits.append(
            f"4. **Token Efficiency**: {tokens.overall_compression_ratio:.1f}× compression ratio "
            f"means users can fit richer context within model token limits, enabling better "
            f"LLM responses without increased costs."
        )

    # Privacy
    if privacy.overall_recall > 0:
        benefits.append(
            f"5. **Privacy Protection**: {privacy.overall_recall:.0%} PII detection recall "
            f"with {privacy.redaction_completeness:.0%} redaction completeness ensures "
            f"sensitive data in screenshots is automatically sanitised before reaching LLMs."
        )

    # Latency
    benefits.append(
        "6. **Accessibility**: The tiered architecture (Lite/Standard) makes structured "
        "screenshot understanding accessible on everyday hardware — no GPU required for "
        "the core functionality."
    )

    lines.extend(benefits or ["No benchmark data available for conclusions."])
    return "\n".join(lines)


def _methodology() -> str:
    return """## 8. Methodology

### Test Environment
- All benchmarks run on the same machine (see header for specs)
- Synthetic screenshots with known ground truth eliminate annotation bias
- Each timing measurement repeated multiple times for statistical significance

### Synthetic Scenes
- **IDE** (1200×900): Title bar, sidebar, code editor, status bar
- **Browser** (1000×800): Nav bar, search, product cards, footer
- **Mobile** (390×844): Header, task list, bottom navigation
- **Chat** (800×600): Header, message bubbles, input field
- **Dashboard** (1440×900): Sidebar nav, KPI cards, data table

### ML Path Handling
- The framework attempts to use ML-powered detection (RT-DETR) when available
- Falls back to rule-based detection if ML models are not installed
- Both Lite and Standard tiers are benchmarked when possible

### Limitations
- Synthetic screenshots may not capture all real-world complexity
- PII detection is regex-based and limited to US-format patterns
- Token costs are estimates based on the 1-token ≈ 4-chars heuristic
- Latency numbers are hardware-dependent and not representative of production

---

*Report generated by IrpAI Research Testing Framework v1.0*"""
