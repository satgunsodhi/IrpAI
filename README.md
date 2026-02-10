# IrpAI — Image Resolution Pipeline AI

> A lightweight machine vision pipeline that converts screenshots into structured context for LLMs, enabling faster and more relevant responses on everyday hardware.

## Why IrpAI?

Even strong LLMs often miss UI structure, key text, and spatial relationships when given a raw screenshot. IrpAI bridges that gap by automatically extracting structured context — text, layout, element types, reading order — and packaging it so the LLM can respond faster and more accurately, without extra user explanation.

## Quick Start

```bash
# Minimal install (CPU only, Tesseract OCR)
pip install -e ".[ocr-tesseract]"

# Standard install (PaddleOCR + YOLO)
pip install -e ".[ocr-paddle,detection]"

# Full install (includes Florence-2 VLM)
pip install -e ".[all]"
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
│          │ YOLO26n UI element detection (buttons, menus, │
│          │ headers, icons, text fields, nav bars …)      │
├──────────────────────────────────────────────────────────┤
│  Layer 3 │ OCR                                           │
│          │ PaddleOCR PP-OCRv5 (primary) or Tesseract 5   │
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
| **lite** | Tesseract | CPU, 4 GB RAM | ~100 ms | ★★☆☆ |
| **standard** | PaddleOCR + YOLO | CPU, 8 GB RAM | ~300 ms | ★★★☆ |
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
│   └── yolo_detector.py     # YOLO + grid fallback
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

## License

MIT
