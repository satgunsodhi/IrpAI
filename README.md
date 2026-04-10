# IrpAI — Image Resolution Pipeline AI

> A lightweight machine vision pipeline that converts screenshots into structured context for LLMs, enabling faster and more relevant responses on everyday hardware.

## Why IrpAI?

Even strong LLMs often miss UI structure, key text, and spatial relationships when given a raw screenshot. IrpAI bridges that gap by automatically extracting structured context — text, layout, element types, reading order — and packaging it so the LLM can respond faster and more accurately, without extra user explanation.

### Measured Impact

Empirical results from the built-in research framework (run on Darwin arm64, Python 3.12.6):

| Metric | Result |
|--------|--------|
| Region detection recall | **63.6%** across 5 scene types |
| Text extraction completeness | **81.9%** |
| Data resilience index | **97.4%** region retention under noisy/low-res/compressed input |
| Token budget compliance | **100%** of outputs stay within 4096 tokens |
| PII detection & redaction | **100% precision, 100% recall** across 5 PII categories |
| Latency — Lite tier | **636ms** avg end-to-end (1.57 images/sec) |
| Latency — Standard tier | **857ms** avg end-to-end (1.17 images/sec) |
| API cost savings (GPT-4-Turbo) | ~**$492/year** at 1,000 images/day for structured scenes |

> Full benchmark data: [`tests/research_output/research_output.md`](tests/research_output/research_output.md)

---

## Quick Start

```bash
# Minimal install (CPU-only fallback detection + Tesseract OCR)
pip install -e .

# Standard install (RT-DETR primary detector)
pip install -e ".[detection-ml]"

# Training extras (fine-tuning scripts + datasets)
pip install -e ".[training]"

# Full install (all optional extras)
pip install -e ".[all]"
```

### Contributor Setup

```bash
# 1) Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

# 2) Install package + dev tools
pip install -e ".[training]"
```

---

## Usage

### Python API

```python
from irpai import IrpAIPipeline, IrpAIConfig, Tier

pipeline = IrpAIPipeline(IrpAIConfig(tier=Tier.STANDARD))
pack = pipeline.run("screenshot.png")

# Compact JSON for programmatic LLM injection
print(pack.to_json())

# Markdown for chat interfaces
print(pack.to_markdown())

# Key metrics
print(f"Regions:  {len(pack.regions)}")
print(f"Tokens:   ~{pack.token_estimate()}")
print(f"App type: {pack.summary.app_type}")
print(f"Time:     {pack.meta.total_ms:.0f} ms")
```

### Format for LLM prompts

```python
from irpai.context.formatter import format_for_llm

# Three output styles
context = format_for_llm(pack, style="system_prompt")  # injects as system message
context = format_for_llm(pack, style="markdown")        # human-readable
context = format_for_llm(pack, style="json")            # structured dict
```

### CLI

```bash
irpai screenshot.png --tier standard --format json --pretty
irpai screenshot.png --tier lite --format markdown -o context.md
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Input Screenshot                      │
├──────────────────────────────────────────────────────────┤
│  Layer 1 │ PREPROCESSING                                 │
│          │ Scale → Denoise → Sharpen → OCR-ready variant │
├──────────────────────────────────────────────────────────┤
│  Layer 2 │ DETECTION                                     │
│          │ HybridDetector                                 │
│          │ RT-DETR primary + rule-based fallback          │
├──────────────────────────────────────────────────────────┤
│  Layer 3 │ OCR                                           │
│          │ EasyOCR / PaddleOCR (primary)                 │
│          │ Tesseract 5 (fallback)                        │
│          │ → blocks, lines, words + positions            │
├──────────────────────────────────────────────────────────┤
│  Layer 4 │ LAYOUT ANALYSIS                               │
│          │ Merge detection + OCR → semantic regions      │
│          │ Classify roles (heading/body/button/nav/…)    │
│          │ Build hierarchy tree + reading order          │
├──────────────────────────────────────────────────────────┤
│  Layer 4b│ VLM CAPTIONING (full tier only)               │
│          │ Florence-2-base scene captioning + OD + OCR   │
├──────────────────────────────────────────────────────────┤
│  Layer 5 │ CONTEXT ASSEMBLY                              │
│          │ → Structured JSON / Markdown / System prompt  │
│          │ Token-budget enforcement, reading-order sort  │
├──────────────────────────────────────────────────────────┤
│  Optional│ RESEARCH HELPER                               │
│          │ Web search from error messages / URLs / cues  │
└──────────────────────────────────────────────────────────┘
```

## Tiers

| Tier | Dependencies | Hardware | Avg Latency | Accuracy |
|------|-------------|----------|------------|----------|
| **lite** | Rule fallback + Tesseract OCR | CPU, 4 GB RAM | ~636ms | ★★☆☆ |
| **standard** | RT-DETR + EasyOCR/PaddleOCR | CPU, 8 GB RAM | ~857ms | ★★★☆ |
| **full** | + Florence-2 VLM | GPU 6 GB VRAM | ~1–2s | ★★★★ |

Latency figures measured on Apple M-series (arm64). Lite warmup: 763ms → 636ms steady-state. Standard warmup: 2.4s → 857ms steady-state (RT-DETR model load).

### Resolution Scaling (Lite tier)

| Resolution | Latency |
|------------|---------|
| 640×480 | 473ms |
| 1024×768 | 620ms |
| 1280×720 | 722ms |
| 1920×1080 | 1,275ms |
| 2560×1440 | 1,335ms |
| 3840×2160 | 1,378ms |

---

## Output Example

```json
{
  "summary": {
    "description": "A VS Code editor showing a Python file with ...",
    "app_type": "ide",
    "key_elements": ["3× body", "2× heading", "1× button", "1× nav"]
  },
  "regions": [
    {"role": "heading", "text": "main.py", "bbox": [10, 5, 200, 30], "confidence": 0.91},
    {"role": "nav", "text": "Explorer\nSearch\nExtensions", "bbox": [0, 40, 200, 600], "confidence": 0.85},
    {"role": "body", "text": "def main():\n    ...", "bbox": [210, 40, 800, 500], "confidence": 0.88},
    {"role": "button", "text": "Run", "bbox": [700, 5, 780, 30], "confidence": 0.79}
  ],
  "hierarchy": {
    "roots": [
      {"role": "heading", "bbox": [10, 5, 200, 30], "text": "main.py", "children": []},
      {"role": "nav", "bbox": [0, 40, 200, 600], "text": "...", "children": []}
    ]
  },
  "meta": {
    "tier": "standard",
    "ocr_engine": "tesseract",
    "detection_model": "HybridDetector(ml=RTDETRDetector)",
    "total_ms": 287.3,
    "image_size": [1200, 900]
  }
}
```

### Output format comparison (real benchmark data)

| Format | IDE scene | Browser scene | Dashboard scene |
|--------|-----------|--------------|----------------|
| JSON (tokens) | 413 | 960 | 1,032 |
| Markdown (tokens) | 117 | 222 | 167 |
| System Prompt (tokens) | 716 | 1,973 | 2,176 |

### Token overhead options

```python
# Include or omit bounding boxes — saves 7–13% of tokens
pack.to_json(include_bboxes=False)

# Include or omit hierarchy tree — 35–56% overhead
pack.to_json(include_hierarchy=False)
```

---

## Project Structure

```
irpai/
├── __init__.py              # Package entry point
├── config.py                # Tiered configuration system
├── pipeline.py              # Main orchestrator (5-layer pipeline)
├── cli.py                   # Command-line interface
├── models.py                # Internal data models (Region, LayoutResult, …)
├── preprocessing/
│   └── normalizer.py        # Scale, denoise, sharpen, OCR-ready variant
├── detection/
│   ├── base.py              # Abstract detector interface + DetectionResult
│   ├── hybrid_detector.py   # RT-DETR + rule-based NMS orchestrator
│   ├── rt_detr_detector.py  # Primary ML detector (HuggingFace RT-DETR)
│   └── rule_fallback.py     # Rule-based backup (grid + color split + contour)
├── ocr/
│   ├── base.py              # Abstract OCR interface + OCRResult
│   ├── engine.py            # Factory with priority-ordered auto-fallback
│   ├── easy_ocr.py          # EasyOCR backend
│   ├── paddle_ocr.py        # PaddleOCR backend
│   └── tesseract_ocr.py     # Tesseract 5 backend
├── layout/
│   ├── analyzer.py          # Region grouping + role classification
│   └── hierarchy.py         # Spatial containment tree builder
├── vlm/
│   ├── base.py              # Abstract VLM interface
│   └── florence.py          # Florence-2 integration (full tier)
├── context/
│   ├── schema.py            # ContextPack / Region / Summary output types
│   ├── assembler.py         # Merges all layers → structured output pack
│   └── formatter.py         # JSON / Markdown / system-prompt serialisation
├── research/
│   └── helper.py            # Web search from screenshot cues
└── utils/
    ├── image.py             # Shared image operations (IoU, load, convert)
    ├── pii.py               # PII detection and redaction utilities
    └── logging.py           # Logger setup

tests/
├── conftest.py              # Shared fixtures (synthetic scenes, pipelines)
├── test_preprocessing.py    # Unit tests — normalizer
├── test_detection.py        # Unit tests — hybrid detector + NMS
├── test_ocr.py              # Unit tests — OCR backends
├── test_layout.py           # Unit tests — layout analyser + hierarchy
├── test_context.py          # Unit tests — assembler + formatter
├── test_models.py           # Unit tests — data model validation
├── test_pipeline.py         # Integration tests — end-to-end pipeline
├── benchmarks/
│   ├── bench_cost.py        # API cost savings across 7 LLM pricing tiers
│   ├── bench_data_efficiency.py  # Resilience under image degradation
│   ├── bench_accuracy.py    # Recall, role accuracy, IoU vs ground truth
│   ├── bench_latency.py     # Per-layer + end-to-end timing profiles
│   ├── bench_token_savings.py    # Compression ratio, format comparison
│   └── bench_privacy.py     # PII detection precision/recall
├── research/
│   ├── synthetic_scenes.py  # 5 synthetic screenshot types with ground truth
│   └── report_generator.py  # Aggregates results → Markdown research report
└── run_research.py          # Single-command research framework entry point
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Research Benchmarks

Run the full research framework to reproduce the published metrics:

```bash
source .venv/bin/activate
python tests/run_research.py
```

This runs all 6 benchmark suites (cost, data efficiency, accuracy, latency, token savings, privacy) against 5 synthetic scene types and writes a full Markdown report to `tests/research_output/research_output.md`.

Individual suites can also be imported and called programmatically:

```python
from tests.benchmarks.bench_accuracy import run_accuracy_benchmark
from tests.benchmarks.bench_latency import run_latency_benchmark

accuracy = run_accuracy_benchmark()
print(f"Region recall: {accuracy.overall_region_recall:.1%}")

latency = run_latency_benchmark()
for tier in latency:
    print(f"{tier.tier}: {tier.warmup_steady_ms:.0f}ms avg, {tier.throughput_ips:.2f} img/s")
```

### Benchmark coverage

| Suite | What it measures |
|-------|-----------------|
| `bench_cost` | Memory footprint, token reduction vs raw vision tokens, API cost savings across GPT-4o / GPT-4-Turbo / Claude-3.5 Sonnet / Claude-3 Opus / Gemini-1.5 Pro/Flash / Llama-3-70B |
| `bench_data_efficiency` | Region retention across resolution (25–100%), noise (σ 0–50), contrast (0.2–1.0), JPEG quality (10–95) |
| `bench_accuracy` | Region recall, role classification accuracy, text completeness, mean IoU, app type detection |
| `bench_latency` | Layer-by-layer P50/P95/P99, warmup cost, resolution scaling profile |
| `bench_token_savings` | JSON/Markdown/system-prompt token counts, hierarchy overhead, bbox omission savings |
| `bench_privacy` | Per-category PII precision/recall (email, phone, SSN, credit card, IP), redaction completeness |

---

## Training

RT-DETR fine-tuning material is in [docs/RTDETR_TRAINING.md](docs/RTDETR_TRAINING.md).

Single command (cross-platform):

```bash
python examples/training/run_mobile_training.py
```

Manual mobile-first workflow (advanced):

```bash
# 1) Build a mixed COCO dataset with guaranteed mobile coverage
python examples/training/prepare_mobile_mixed_coco.py \
  --output-root data/screenspot_mobile_mixed_coco \
  --total-samples 240 \
  --val-ratio 0.2 \
  --mobile-min-ratio 0.5

# 2) Train RT-DETR on the prepared dataset
python examples/training/finetune_rtdetr_ui.py \
  --train-images data/screenspot_mobile_mixed_coco/train/images \
  --train-annotations data/screenspot_mobile_mixed_coco/train/annotations.json \
  --val-images data/screenspot_mobile_mixed_coco/val/images \
  --val-annotations data/screenspot_mobile_mixed_coco/val/annotations.json \
  --output-dir checkpoints/rtdetr-ui-mobile-mixed-complete \
  --model-id PekingU/rtdetr_r18vd_coco_o365 \
  --epochs 10 \
  --batch-size 4
```

Best trained model path:

```text
checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best
```

Repository hygiene:

- Generated datasets in `data/` are git-ignored.
- Trained checkpoints in `checkpoints/` are git-ignored.
- You do not need to push either to GitHub.

---

## Model Release (GitHub)

Use this flow to publish the best checkpoint as a GitHub Release asset.

```bash
# 1. Export checkpoint zip
python examples/training/run_mobile_training.py --skip-prepare --export-best-zip

# 2. Authenticate GitHub CLI (one-time)
gh auth login

# 3. Create release
gh release create v0.1.0-models \
  --title "v0.1.0 models" \
  --notes "RT-DETR mobile mixed checkpoint-best"

# 4. Upload artifact
gh release upload v0.1.0-models \
  artifacts/rtdetr-ui-mobile-mixed-complete-checkpoint-best.zip --clobber

# 5. Verify
gh release view v0.1.0-models
```

---

## License

MIT
