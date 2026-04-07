# IrpAI - Technical Architecture

> Version: 0.1.x  
> Date: March 2026  
> Status: Active implementation + training path

---

## Table of Contents

1. System Overview
2. Runtime Architecture
3. Pipeline Layers
4. Detection Design (RT-DETR + Fallback)
5. Tiered Execution Model
6. Output Schema
7. Training and Fine-Tuning Path
8. Feasibility and Licensing
9. Roadmap

---

## 1. System Overview

### 1.1 Problem

General-purpose LLMs often misread screenshots because they miss spatial layout, fail on small text, and do not reliably distinguish interactive components from plain content.

### 1.2 Solution

IrpAI inserts a preprocessing and structure-extraction layer before the LLM:

Screenshot -> IrpAI Pipeline -> Context Pack -> LLM

The pipeline converts raw pixels into structured, token-efficient context containing:
- extracted text
- semantic regions
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
                    | Context Pack   |
                    +----------------+
```

### 2.2 Detection Sub-Flow

```
HybridDetector
   |
   +--> RT-DETR (primary, ML) -------------------+
   |                                              |
   +--> FallbackGridDetector (rules, backup) ----+--> Merge + NMS --> final detections
```

Behavior:
- If RT-DETR loads and inference succeeds, ML detections are used.
- Structural rule detections can be merged when enabled.
- If RT-DETR fails or dependencies are missing, fallback detector becomes the full detection path.

This guarantees detection output without hard pipeline failure.

---

## 3. Pipeline Layers

### 3.1 Layer 1: Preprocessing (`irpai.preprocessing.normalizer`)

Purpose:
- normalize image size and quality
- produce RGB and OCR-ready variants

Key operations:
- optional resize to max dimension
- denoising
- sharpening
- grayscale OCR image preparation

### 3.2 Layer 2: Detection (`irpai.detection`)

Active modules:
- `hybrid_detector.py`: orchestration and fusion
- `rt_detr_detector.py`: ML primary detector
- `rule_fallback.py`: rule-based fallback implementation

Legacy compatibility:
- `yolo_detector.py` is retained only as an import shim to avoid breaking old imports.
- Runtime detection path does not depend on YOLO.

### 3.3 Layer 3: OCR (`irpai.ocr`)

Purpose:
- extract text blocks and line-level confidence
- provide positional text evidence for layout and summary

Selection:
- OCR backend is created through `create_ocr_engine`.
- If OCR fails, fallback logic in the pipeline handles recovery.

### 3.4 Layer 4: Layout Analysis (`irpai.layout.analyzer`)

Purpose:
- merge detection and OCR evidence
- classify roles and region hierarchy
- compute reading order

### 3.5 Layer 5: Context Assembly (`irpai.context.assembler`)

Purpose:
- convert regions to output schema
- include metadata (tier, detector name, timing, image size)
- produce JSON/Markdown/system-prompt-ready formats

---

## 4. Detection Design (RT-DETR + Fallback)

### 4.1 Why RT-DETR as Primary

RT-DETR provides stronger semantic detection behavior for UI-like layouts than pure heuristics, especially when elements are nested or visually dense.

Current default model id:
- `PekingU/rtdetr_r18vd_coco_o365`

Current detector config fields:
- `use_ml`
- `model_id`
- `confidence_threshold`
- `structural_boost`
- `nms_iou_threshold`
- `max_detections`
- `device`

### 4.2 Hybrid Merge Policy

1. Run RT-DETR if enabled.
2. Run fallback path when:
- `structural_boost` is enabled, or
- ML path fails/unavailable.
3. Merge detections and apply NMS.
4. If empty, emit one full-frame `content_area` region.

### 4.3 Fallback Strengths

The fallback detector remains useful for:
- minimal dependency environments
- deterministic structural regions
- survival path when model loading fails

---

## 5. Tiered Execution Model

Current code implements two tiers.

### 5.1 Lite Tier

- `Tier.LITE`
- `DetectionConfig.use_ml=False`
- detection runs via rule fallback path
- OCR defaults remain lightweight

Use case:
- environments without ML detection dependencies

### 5.2 Standard Tier

- `Tier.STANDARD`
- `DetectionConfig.use_ml=True`
- RT-DETR primary + fallback backup

Use case:
- production-like screenshot understanding on CPU or accelerator

### 5.3 Planned Full Tier

A future full tier can add a VLM stage after layout assembly or as an enrichment pass. This is documented as roadmap work and is not the default runtime tier in current code.

---

## 6. Output Schema

Representative structure:

```json
{
  "summary": {
    "description": "...",
    "app_type": "...",
    "key_elements": ["..."],
    "vlm_caption": "..."
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
    "ocr_engine": "...",
    "detection_model": "HybridDetector(ml=RTDETRDetector(...))",
    "total_ms": 0,
    "image_size": [1920, 1080]
  }
}
```

---

## 7. Training and Fine-Tuning Path

### 7.1 Training Docs and Script

The project now includes dedicated training material:
- `docs/RTDETR_TRAINING.md`
- `examples/training/finetune_rtdetr_ui.py`

### 7.2 Intended Data Format

Training script expects COCO-style detection annotations for train/val.

### 7.3 Recommended Data Sources

- ScreenSpot-style UI screenshots
- Rico-like mobile UI element datasets
- project-specific annotated screenshots

### 7.4 Deployment Strategy

1. Fine-tune RT-DETR checkpoint.
2. Save checkpoint locally or publish model id.
3. Set `DetectionConfig.model_id` to the fine-tuned model.
4. Keep fallback enabled for resilience.

---

## 8. Feasibility and Licensing

### 8.1 Runtime Feasibility

- Lite remains minimal and robust.
- Standard provides semantic detection without sacrificing fallback safety.
- CPU-only use is still supported.

### 8.2 Licensing Snapshot

Primary detection stack:
- RT-DETR via HuggingFace model/runtime stack
- Fallback detector implemented in-repo with OpenCV/Numpy

Legacy YOLO runtime is not part of the active detection path.

---

## 9. Roadmap

### Phase 1 (Completed in this iteration)

- Hybrid detector integrated into pipeline
- RT-DETR primary detector added
- Rule fallback preserved as backup
- Config and tests updated for new behavior
- Training docs and fine-tuning script added

### Phase 2

- stronger UI label mapping for generic RT-DETR outputs
- benchmark harness for UI datasets
- optional parallel OCR + detection execution

### Phase 3

- add full-tier enrichment path with optional VLM stage
- richer region-level scene understanding

---

This document should be kept aligned with the runtime implementation in:
- `src/irpai/pipeline.py`
- `src/irpai/config.py`
- `src/irpai/detection/`
