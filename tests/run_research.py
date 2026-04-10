#!/usr/bin/env python3
"""
IrpAI Research Framework — Single-Command Entry Point
=====================================================

Run all benchmark suites and generate a comprehensive Markdown
research report.

Usage::

    python tests/run_research.py

Output::

    tests/research_output/research_output.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from tests.benchmarks.bench_accuracy import run_accuracy_benchmark
    from tests.benchmarks.bench_cost import run_cost_benchmark
    from tests.benchmarks.bench_data_efficiency import run_data_efficiency_benchmark
    from tests.benchmarks.bench_latency import run_latency_benchmark
    from tests.benchmarks.bench_privacy import run_privacy_benchmark
    from tests.benchmarks.bench_token_savings import run_token_savings_benchmark
    from tests.research.report_generator import generate_report

    output_dir = PROJECT_ROOT / "tests" / "research_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "research_output.md"

    t_total = time.perf_counter()
    print("=" * 60)
    print("  IrpAI Research Testing Framework")
    print("=" * 60)
    print()

    # 1. Cost Analysis
    print("▸ [1/6] Running cost analysis benchmark...")
    t0 = time.perf_counter()
    cost_results = run_cost_benchmark()
    print(f"  ✓ Cost analysis complete ({(time.perf_counter() - t0):.1f}s) "
          f"— {len(cost_results.tier_results)} tier×scene results")
    print()

    # 2. Data Efficiency
    print("▸ [2/6] Running data efficiency benchmark...")
    t0 = time.perf_counter()
    data_eff_results = run_data_efficiency_benchmark()
    print(f"  ✓ Data efficiency complete ({(time.perf_counter() - t0):.1f}s) "
          f"— {len(data_eff_results.results)} degradation tests")
    print()

    # 3. Accuracy
    print("▸ [3/6] Running accuracy benchmark...")
    t0 = time.perf_counter()
    accuracy_results = run_accuracy_benchmark()
    print(f"  ✓ Accuracy complete ({(time.perf_counter() - t0):.1f}s) "
          f"— {len(accuracy_results.scene_results)} scene results")
    print()

    # 4. Latency
    print("▸ [4/6] Running latency profiling benchmark...")
    t0 = time.perf_counter()
    latency_results = run_latency_benchmark()
    print(f"  ✓ Latency profiling complete ({(time.perf_counter() - t0):.1f}s) "
          f"— {len(latency_results)} tier profiles")
    print()

    # 5. Token Savings
    print("▸ [5/6] Running token savings benchmark...")
    t0 = time.perf_counter()
    token_results = run_token_savings_benchmark()
    print(f"  ✓ Token savings complete ({(time.perf_counter() - t0):.1f}s) "
          f"— {len(token_results.scene_results)} scene analyses")
    print()

    # 6. Privacy
    print("▸ [6/6] Running privacy benchmark...")
    t0 = time.perf_counter()
    privacy_results = run_privacy_benchmark()
    print(f"  ✓ Privacy benchmark complete ({(time.perf_counter() - t0):.1f}s) "
          f"— {privacy_results.total_test_cases} test cases")
    print()

    # Generate report
    print("▸ Generating research report...")
    report = generate_report(
        cost=cost_results,
        data_eff=data_eff_results,
        accuracy=accuracy_results,
        latency=latency_results,
        tokens=token_results,
        privacy=privacy_results,
        output_path=output_path,
    )

    total_time = time.perf_counter() - t_total
    print()
    print("=" * 60)
    print(f"  ✓ Research report generated in {total_time:.1f}s")
    print(f"  📄 {output_path}")
    print("=" * 60)
    print()

    # Print executive summary
    summary_end = report.find("## 1.")
    if summary_end > 0:
        print(report[:summary_end])


if __name__ == "__main__":
    main()
