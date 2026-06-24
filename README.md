# IrpAI - Image Resolution Pipeline AI

IrpAI is a lightweight machine vision pipeline that converts screenshots into structured context for LLMs.

It extracts:
- semantic UI regions
- OCR text
- hierarchy and reading order
- metadata (tier, detector, timing, image size)

The output is optimized for prompt injection and downstream automation.

## Authors

- **Satgun Singh Sodhi**
- **Aakaash S S**
- **Agrim Dev**

## Why IrpAI

Raw screenshots are noisy and expensive for LLMs to parse directly. IrpAI adds a deterministic pre-processing and structure extraction layer so models receive cleaner context instead of raw pixels.

## Runtime Overview

The active pipeline is 5 layers:

1. Preprocessing (normalize, denoise, sharpen)
2. Detection (RT-DETR primary + rule fallback)
3. OCR (EasyOCR)
4. Layout analysis (roles, hierarchy, reading order)
5. Context assembly (JSON, Markdown, system prompt)

## Tiers

Current runtime supports two tiers:

| Tier | Detection Path | Typical Use |
|---|---|---|
| lite | Rule fallback only | Minimal dependency environments |
| standard | RT-DETR primary + fallback | Higher quality detection path |

Notes:
- Both tiers use the same preprocessing, layout, and context assembly logic.
- OCR runs through EasyOCR when OCR dependencies are installed.
- If OCR dependencies are missing, the pipeline still returns structured output with empty OCR content.

## Installation

Base package:

```bash
pip install -e .
```

Optional extras from pyproject.toml:

```bash
pip install -e ".[ocr-easyocr]"
pip install -e ".[detection-ml]"
pip install -e ".[training]"
pip install -e ".[all]"
pip install -e ".[dev]"
```

What they include:
- ocr-easyocr: EasyOCR backend and torch runtime
- detection-ml: RT-DETR dependencies (transformers + torch)
- training: fine-tuning and dataset tooling
- all: ocr-easyocr + detection-ml
- dev: pytest, ruff, mypy

## Quick Start

### Python API

```python
from irpai import IrpAIPipeline, IrpAIConfig, Tier

pipeline = IrpAIPipeline(IrpAIConfig(tier=Tier.STANDARD))
pack = pipeline.run("screenshot.png")

print(pack.to_json())
print(pack.to_markdown())
print(pack.token_estimate())
```

### LLM Formatting Helpers

```python
from irpai.context.formatter import format_for_llm

json_context = format_for_llm(pack, style="json")
markdown_context = format_for_llm(pack, style="markdown")
system_prompt_context = format_for_llm(pack, style="system_prompt")
```

### CLI

```bash
irpai screenshot.png --tier standard --format json --pretty
irpai screenshot.png --tier lite --format markdown -o context.md
irpai screenshot.png --tier standard --format system_prompt
```

CLI supports:
- tiers: lite, standard
- output formats: json, markdown, system_prompt

## Output Model

Primary output type is ContextPack in src/irpai/context/schema.py.

Top-level fields:
- summary
- regions
- hierarchy
- meta

Serialization methods:
- to_json(compact=True|False)
- to_markdown()
- to_dict(...)

## Example (Compact JSON)

```json
{
	"summary": {
		"description": "...",
		"app_type": "ide",
		"key_elements": ["..."]
	},
	"regions": [
		{
			"role": "button",
			"text": "Run",
			"bbox": [700, 5, 780, 30],
			"confidence": 0.79
		}
	],
	"meta": {
		"tier": "standard",
		"ocr_engine": "easyocr",
		"detection_model": "HybridDetector(...)",
		"total_ms": 287.3,
		"image_size": [1200, 900]
	}
}
```

## Configuration

Main configuration lives in src/irpai/config.py.

Key config objects:
- PreprocessingConfig
- DetectionConfig
- OCRConfig
- LayoutConfig
- OutputConfig
- IrpAIConfig

Default detector model behavior:
- Uses local checkpoint at checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best if present
- Otherwise falls back to PekingU/rtdetr_r18vd_coco_o365

## Training RT-DETR

Training guides and scripts:
- docs/RTDETR_TRAINING.md
- examples/training/run_mobile_training.py
- examples/training/prepare_mobile_mixed_coco.py
- examples/training/finetune_rtdetr_ui.py

Quick command:

```bash
python examples/training/run_mobile_training.py
```

## Testing and Benchmarks

Install test dependencies:

```bash
pip install -e ".[all,dev]"
```

Run test suite:

```bash
pytest
```

Run research benchmark suite:

```bash
python tests/run_research.py
```

Benchmark report output:
- tests/research_output/research_output.md

## Project Structure (High Level)

```text
src/irpai/
	config.py
	pipeline.py
	context/
	detection/
	layout/
	ocr/
	preprocessing/
	utils/
	vlm/

docs/
	ARCHITECTURE.md
	RTDETR_TRAINING.md

examples/training/
tests/
```

## Build and Release

See:
- build_all.py (build script)
- RELEASE.md

## Roadmap

Planned work includes:
- fuller VLM enrichment integration
- stronger UI role mapping for generic detector labels
- deeper benchmark coverage for UI-specific datasets
