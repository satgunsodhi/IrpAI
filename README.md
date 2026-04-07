# IrpAI — Image Resolution Pipeline AI

> A lightweight machine vision pipeline that converts screenshots into structured context for LLMs, enabling faster and more relevant responses on everyday hardware.

## Why IrpAI?

Even strong LLMs often miss UI structure, key text, and spatial relationships when given a raw screenshot. IrpAI bridges that gap by automatically extracting structured context — text, layout, element types, reading order — and packaging it so the LLM can respond faster and more accurately, without extra user explanation.

## Quick Start

```bash
# Minimal install (CPU-only fallback detection + EasyOCR)
pip install -e .

# Standard install (RT-DETR primary + fallback detector)
pip install -e ".[detection-ml]"

# Training extras (fine-tuning scripts + datasets)
pip install -e ".[training]"

# Full install (all optional extras)
pip install -e ".[all]"
```

## Contributor Setup

```bash
# 1) Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

# 2) Install package + training tools
pip install -e ".[training]"
```

### Basic usage

```python
from irpai import IrpAIPipeline, IrpAIConfig, Tier

pipeline = IrpAIPipeline(IrpAIConfig(tier=Tier.STANDARD))
pack = pipeline.run("screenshot.png")

# JSON context for LLM injection
print(pack.to_json())

# Markdown format for chat
print(pack.to_markdown())
```

### CLI

```bash
irpai screenshot.png --tier standard --format json --pretty
irpai screenshot.png --tier lite --format markdown -o context.md
```

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
│          │ EasyOCR (primary) or Tesseract 5 fallback      │
│          │ (fallback) → blocks, lines, words + positions │
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

| Tier | Dependencies | Hardware | Speed | Accuracy |
|------|-------------|----------|-------|----------|
| **lite** | Rule fallback detector + EasyOCR | CPU, 4 GB RAM | ~100 ms | ★★☆☆ |
| **standard** | RT-DETR + EasyOCR | CPU, 8 GB RAM | ~300 ms | ★★★☆ |
| **full** | + Florence-2 VLM | GPU 6 GB VRAM | ~800 ms | ★★★★ |

## Output Example

```json
{
  "summary": {
    "description": "A VS Code editor showing a Python file with ...",
    "app_type": "ide",
    "key_elements": ["3× body", "2× heading", "1× button", "1× nav"]
  },
  "regions": [
    {"role": "heading", "text": "main.py", "bbox": [10, 5, 200, 30]},
    {"role": "nav", "text": "Explorer\nSearch\nExtensions", "bbox": [0, 40, 200, 600]},
    {"role": "body", "text": "def main():\n    ...", "bbox": [210, 40, 800, 500]},
    {"role": "button", "text": "Run", "bbox": [700, 5, 780, 30]}
  ],
  "hierarchy": {
    "roots": [
      {"role": "heading", "bbox": [10, 5, 200, 30], "text": "main.py"},
      {"role": "nav", "bbox": [0, 40, 200, 600], "text": "..."}
    ]
  },
  "meta": {
    "tier": "standard",
    "ocr_engine": "PaddleOCR",
    "total_ms": 287.3
  }
}
```

## Project Structure

```
irpai/
├── __init__.py              # Package entry point
├── config.py                # Tiered configuration system
├── pipeline.py              # Main orchestrator
├── cli.py                   # Command-line interface
├── preprocessing/
│   └── normalizer.py        # Image scaling, denoising, sharpening
├── detection/
│   ├── base.py              # Abstract detector interface
│   ├── hybrid_detector.py   # RT-DETR + fallback orchestrator
│   ├── rt_detr_detector.py  # Primary ML detector
│   └── rule_fallback.py     # Rule-based backup detector
├── ocr/
│   ├── base.py              # Abstract OCR interface
│   ├── paddle_ocr.py        # PaddleOCR backend
│   ├── tesseract_ocr.py     # Tesseract backend
│   └── engine.py            # Factory with auto-fallback
├── layout/
│   ├── analyzer.py          # Region grouping + role classification
│   └── hierarchy.py         # Spatial containment tree builder
├── vlm/
│   ├── base.py              # Abstract VLM interface
│   └── florence.py           # Florence-2 integration
├── context/
│   ├── schema.py            # ContextPack / Region / Summary types
│   ├── assembler.py         # Merges layers → structured output
│   └── formatter.py         # JSON / Markdown / system-prompt output
├── research/
│   └── helper.py            # Web search from screenshot cues
└── utils/
    ├── image.py             # Shared image operations
    └── logging.py           # Logger setup
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

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

- Generated datasets in `data/` are ignored.
- Trained checkpoints in `checkpoints/` are ignored.
- You do not need to push either to GitHub.

## Model Release (GitHub)

Use this flow to publish the best checkpoint as a GitHub Release asset (recommended).

1. Build/export best checkpoint zip:

```bash
python examples/training/run_mobile_training.py --skip-prepare --export-best-zip
```

2. Authenticate GitHub CLI (one time):

```bash
gh auth login
```

3. Create a release tag and release:

```bash
gh release create v0.1.0-models \
  --title "v0.1.0 models" \
  --notes "RT-DETR mobile mixed checkpoint-best"
```

4. Upload artifact zip to the release:

```bash
gh release upload v0.1.0-models artifacts/rtdetr-ui-mobile-mixed-complete-checkpoint-best.zip --clobber
```

5. Verify uploaded assets:

```bash
gh release view v0.1.0-models
```

## License

MIT
