# IrpAI - Technical Architecture

> Version: 0.1.x  
> Date: April 2026  
> Status: Active implementation + training path

---

## Table of Contents

1. System Overview
2. Runtime Architecture
3. Pipeline Layers
4. Detection Design (RT-DETR + Fallback)
5. Tiered Execution Model
6. Output Schema
7. Data Models
8. Training and Fine-Tuning Path
9. Feasibility and Licensing
10. Roadmap

---

## 1. System Overview

### 1.1 Problem

General-purpose LLMs often misread screenshots because they miss spatial layout, fail on small text, and do not reliably distinguish interactive components from plain content.

### 1.2 Solution

IrpAI inserts a preprocessing and structure-extraction layer before the LLM:

```
Screenshot → IrpAI Pipeline → Context Pack → LLM
```

The pipeline converts raw pixels into structured, token-efficient context containing:
- extracted text
- semantic regions with roles
- hierarchy and reading order
- runtime metadata

### 1.3 Core Principles

| Principle | Implementation |
|---|---|
| Graceful degradation | ML detector has a built-in rule fallback path |
| Practical accuracy | RT-DETR primary path for semantic object reasoning |
| Lightweight baseline | Lite tier runs without heavy ML detection dependencies |
| Token efficiency | Structured output and budget-aware context assembly |
| Modular design | Swappable interfaces for OCR, detection, formatting |
| Standard dataclasses | All models use Python `dataclasses` — no Pydantic dependency |

---

## 2. Runtime Architecture

### 2.1 High-Level Flow

```
                    +----------------+
                    |   Screenshot   |
                    +-------+--------+
                            |
                    +-------v--------+
                    | Layer 1        |
                    | Preprocessing  |
                    | Normalizer     |
                    +-------+--------+
                            |
            +---------------+---------------+
            |                               |
    +-------v--------+              +-------v--------+
    | Layer 2        |              | Layer 3        |
    | Detection      |              | OCR            |
    | HybridDetector |              | Engine Factory |
    +-------+--------+              +-------+--------+
            |                               |
            +---------------+---------------+
                            |
                    +-------v--------+
                    | Layer 4        |
                    | Layout Analyzer|
                    +-------+--------+
                            |
                    +-------v--------+
                    | Layer 5        |
                    | Context        |
                    | Assembler      |
                    +-------+--------+
                            |
                    +-------v--------+
                    | ContextPack    |
                    +----------------+
```

Layers 2 and 3 (Detection and OCR) execute sequentially but are independent of each other — their results are merged downstream in Layer 4.

### 2.2 Detection Sub-Flow

```
HybridDetector
   |
   +--> RT-DETR (primary, ML) -------------------+
   |                                              |
   +--> FallbackGridDetector (rules, backup) ----+--> Merge + NMS --> final detections
```

Behavior:
- If RT-DETR loads and inference succeeds, ML detections form the primary output.
- Structural rule detections are merged when `structural_boost=True` (default).
- If RT-DETR fails or dependencies are unavailable, the fallback detector becomes the full detection path without pipeline failure.

---

## 3. Pipeline Layers

### 3.1 Layer 1: Preprocessing (`irpai.preprocessing.normalizer`)

Purpose:
- normalize image size and quality
- produce RGB and OCR-ready variants

Key operations:
- optional resize to `target_max_dimension` (default 1920 px on longest side)
- denoising (non-local means)
- sharpening (unsharp-mask)
- grayscale OCR image preparation

### 3.2 Layer 2: Detection (`irpai.detection`)

Active modules:

| File | Role |
|------|------|
| `hybrid_detector.py` | Orchestration, fusion, and NMS |
| `rt_detr_detector.py` | ML primary detector (HuggingFace RT-DETR) |
| `rule_fallback.py` | Rule-based fallback (grid + color split + contour) |

Legacy compatibility:
- `yolo_detector.py` is a thin import shim mapping old `YOLODetector` references to `FallbackGridDetector`. It exists only to avoid breaking any external scripts that imported it; the active detection path does not depend on YOLO.

### 3.3 Layer 3: OCR (`irpai.ocr`)

Purpose:
- extract text blocks and line-level confidence scores
- provide positional text evidence for layout and summary generation

Engine selection:
- OCR backend is created through `create_ocr_engine(OCRConfig)`.
- Currently supports `easyocr` as the primary backend.
- If OCR fails at runtime, the pipeline continues with an empty OCR result (no hard failure).

### 3.4 Layer 4: Layout Analysis (`irpai.layout.analyzer`)

Purpose:
- merge detection and OCR evidence into semantic regions
- classify roles: `heading`, `body`, `button`, `nav`, `icon`, `input`, `image`, …
- compute reading order (left-to-right, top-to-bottom by default)
- build parent → child hierarchy tree via spatial containment

Key config knobs: `merge_distance_x/y`, `min_region_area`, `heading_font_ratio`, `column_detection`.

### 3.5 Layer 5: Context Assembly (`irpai.context.assembler`)

Purpose:
- convert `LayoutResult` regions into the `ContextPack` output schema
- include `PipelineMetadata` (tier, detector name, timing, image size)
- enforce token budget (`max_context_tokens`)
- produce JSON / Markdown / system-prompt-ready formats via `format_for_llm`

### Optional: VLM (`irpai.vlm`)

The `irpai.vlm` package contains a Florence-2 backend (`florence.py`) and an abstract base interface (`base.py`). These modules exist in the codebase but are **not wired into the active pipeline**. The VLM stage is planned roadmap work (see section 5.3 and section 9).

---

## 4. Detection Design (RT-DETR + Fallback)

### 4.1 Why RT-DETR as Primary

RT-DETR provides stronger semantic detection behavior for UI-like layouts than pure heuristics, especially when elements are nested or visually dense. Being a transformer-based detector (no NMS re-tuning required), it handles overlapping elements more naturally.

Current default model id:
- `PekingU/rtdetr_r18vd_coco_o365`

`IrpAIConfig` auto-resolves to a local fine-tuned checkpoint if it exists at:
```
checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best
```

Current `DetectionConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `use_ml` | `True` | Enable RT-DETR as primary |
| `model_id` | auto | HF model id or local checkpoint path |
| `confidence_threshold` | `0.35` | Minimum ML detection confidence |
| `structural_boost` | `True` | Merge fallback structural regions |
| `nms_iou_threshold` | `0.35` | IoU threshold for final NMS merge |
| `max_detections` | `120` | Max detections returned |
| `device` | `"cpu"` | Torch device (`cpu`, `cuda`, `mps`, `auto`) |

### 4.2 Hybrid Merge Policy

1. Run RT-DETR if `use_ml=True`.
2. Run fallback detector when `structural_boost=True` or when ML path fails/is unavailable.
3. Merge both detection sets and apply NMS at `nms_iou_threshold`.
4. If the merged set is empty, emit one full-frame `content_area` region.

### 4.3 Fallback Strengths

The `FallbackGridDetector` remains useful for:
- minimal dependency environments (Lite tier)
- deterministic, reproducible structural regions
- survival path when model loading fails

---

## 5. Tiered Execution Model

Two tiers are implemented and stable.

### 5.1 Lite Tier

- `Tier.LITE`
- `DetectionConfig.use_ml=False`
- detection runs via rule fallback path only (no ML models loaded)
- OCR via EasyOCR, CPU-only
- **Use case:** environments without PyTorch / transformers, or when ultra-fast startup is required

### 5.2 Standard Tier

- `Tier.STANDARD`
- `DetectionConfig.use_ml=True`
- RT-DETR primary + fallback backup
- OCR via EasyOCR, CPU
- **Use case:** production screenshot understanding on CPU or GPU

Both tiers share the same preprocessing, layout, and context-assembly code paths — only detection differs.

---

## 6. Output Schema

The canonical output of the pipeline is a `ContextPack` (defined in `irpai.context.schema`):

```json
{
  "summary": {
    "description": "...",
    "app_type": "...",
    "key_elements": ["..."]
  },
  "regions": [
    {
      "role": "button",
      "text": "Submit",
      "bbox": [100, 220, 180, 250],
      "confidence": 0.92,
      "properties": {},
      "children": []
    }
  ],
  "hierarchy": {
    "roots": []
  },
  "meta": {
    "irpai_version": "0.1.0",
    "tier": "standard",
    "ocr_engine": "easyocr",
    "detection_model": "HybridDetector(ml=RTDETRDetector(...))",
    "total_ms": 0,
    "image_size": [1920, 1080]
  }
}
```

Key schema types:

| Type | Module | Role |
|------|--------|------|
| `ContextPack` | `irpai.context.schema` | Top-level pipeline output |
| `Region` | `irpai.context.schema` | Single screen region with role, text, bbox |
| `ScreenSummary` | `irpai.context.schema` | Natural-language summary + app type |
| `PipelineMetadata` | `irpai.context.schema` | Run provenance (tier, timing, model) |

---

## 7. Data Models

`irpai.models` provides rich domain types for external consumers of the pipeline:

| Model | Purpose |
|-------|---------|
| `BoundingBox` | Axis-aligned box with IoU, containment, and serialisation helpers |
| `UIElementType` | Enum of semantic UI element kinds (button, menu, icon, …) |
| `UIDetection` | Detected UI element + optional OCR text |
| `TextExtraction` | OCR text region with position and confidence |
| `SpatialRelationship` | Named spatial relationship between two elements |
| `LayoutStructure` | Screen regions (navbar, sidebar, footer, main_content, modal) |
| `StructuredContext` | Complete structured context with converter from `ContextPack` |
| `PipelineResult` | Wraps `StructuredContext` with timing and confidence fields |

All models are plain Python `dataclasses` — no external validation library required. Converters (`from_context_pack`, `from_detection_result`, etc.) allow bridging between the internal pipeline types and the public model API.

---

## 8. Training and Fine-Tuning Path

### 8.1 Training Docs and Script

The project includes dedicated training material:
- `docs/RTDETR_TRAINING.md` — full guide
- `examples/training/finetune_rtdetr_ui.py` — training script
- `examples/training/prepare_mobile_mixed_coco.py` — dataset preparation
- `examples/training/run_mobile_training.py` — single-command entry point

### 8.2 Intended Data Format

Training script expects COCO-style detection annotations for train/val splits.

### 8.3 Recommended Data Sources

- ScreenSpot-Pro (professional desktop screenshots)
- ScreenSpot-v2 mobile split (mobile screenshot coverage)
- Project-specific annotated screenshots

### 8.4 Deployment Strategy

1. Fine-tune RT-DETR checkpoint (see `docs/RTDETR_TRAINING.md`).
2. Save checkpoint to `checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best`.
3. `IrpAIConfig` auto-detects this path — no code change required.
4. Alternatively, set `DetectionConfig.model_id` explicitly to a HuggingFace Hub ID or any local path.
5. Keep `structural_boost=True` for resilience.

---

## 9. Feasibility and Licensing

### 9.1 Runtime Feasibility

- **Lite** is minimal and robust — runs on any CPU with Pillow + NumPy + OpenCV.
- **Standard** requires PyTorch and `transformers` for RT-DETR, but falls back gracefully.
- CPU-only use is fully supported for both tiers.
- EasyOCR requires PyTorch; the pipeline handles its absence gracefully.

### 9.2 Core Dependencies

| Dependency | Version | Required for |
|-----------|---------|-------------|
| `Pillow` | ≥10.0 | Image loading and manipulation |
| `numpy` | ≥1.24 | Array ops, image data |
| `opencv-python-headless` | ≥4.8 | Preprocessing, contour detection |
| `torch` | ≥2.0 | RT-DETR + EasyOCR (optional extras) |
| `transformers` | ≥4.40 | RT-DETR model (optional extras) |
| `easyocr` | ≥1.7 | OCR backend (optional extras) |

### 9.3 Licensing Snapshot

- RT-DETR via HuggingFace model/runtime stack (Apache 2.0)
- `FallbackGridDetector` implemented in-repo with OpenCV/NumPy (MIT)
- YOLO runtime is **not** part of the active detection path; `yolo_detector.py` is a compatibility shim only

---

## 10. Roadmap

### Completed

- ✅ Hybrid detector integrated into pipeline
- ✅ RT-DETR primary detector added
- ✅ Rule fallback preserved as backup with NMS merge
- ✅ Dataclass-based models (removed Pydantic dependency)
- ✅ Config and tests updated for new behavior
- ✅ Training docs and fine-tuning script added
- ✅ Auto-detection of local RT-DETR checkpoint
- ✅ PII detection and redaction utilities
- ✅ Research benchmark framework (6 benchmark suites)

### Near-term

- Stronger UI label mapping for generic RT-DETR COCO outputs → UI roles
- Benchmark harness for UI-specific datasets (ScreenSpot-Pro mAP)
- Optional parallel OCR + detection execution (async pipeline)

### Long-term

- Optional VLM enrichment pass (Florence-2 or equivalent) as a pluggable post-processor
- Richer region-level scene understanding and app-type detection
- Multi-language OCR support

---

This document should be kept aligned with the runtime implementation in:
- `src/irpai/pipeline.py`
- `src/irpai/config.py`
- `src/irpai/detection/`
- `src/irpai/models.py`
