# IrpAI — Technical Architecture Document

> **Version:** 0.1.0 · **Date:** February 2026 · **Status:** Prototype

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Pipeline Layers — Detailed Design](#3-pipeline-layers--detailed-design)
4. [Component Selection & Benchmarks](#4-component-selection--benchmarks)
5. [Tiered Execution Model](#5-tiered-execution-model)
6. [Output Schema Specification](#6-output-schema-specification)
7. [Feasibility Analysis](#7-feasibility-analysis)
8. [Improvements & Recommendations](#8-improvements--recommendations)
9. [Comparison with Existing Systems](#9-comparison-with-existing-systems)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. System Overview

### 1.1 Problem

Even strong LLMs (GPT-4V, Claude, Gemini) frequently misinterpret screenshots because they:
- Miss spatial relationships between UI elements
- Fail to read small or low-contrast text
- Don't infer layout structure (columns, headers, navigation)
- Cannot distinguish interactive elements (buttons, dropdowns) from static text

This forces users into multi-round prompting ("No, I meant the sidebar", "The error is in the bottom panel"), wasting time and tokens.

### 1.2 Solution

IrpAI is a **pre-processing pipeline** that sits between the user's screenshot and the LLM.  It extracts structured context automatically:

```
Screenshot → [IrpAI Pipeline] → Structured Context Pack → LLM
```

The LLM receives a compact JSON/Markdown representation of the screen rather than (or alongside) the raw image, enabling:
- **Faster responses** — the LLM doesn't need to "figure out" the UI
- **Higher accuracy** — explicit text + layout beats implicit vision
- **Lower compute** — structured context is smaller than image tokens
- **Fewer back-and-forth prompts** — context is already complete

### 1.3 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **80/20 accuracy** | Target ~80% "good enough" understanding with minimal compute |
| **Tiered execution** | Lite (CPU-only) → Standard (CPU+models) → Full (GPU+VLM) |
| **Graceful degradation** | Every component has a fallback; pipeline never hard-fails |
| **Token efficiency** | Output stays within configurable token budgets |
| **Pluggable backends** | Abstract interfaces for OCR, detection, VLM — swap freely |
| **Consumer hardware** | Standard tier runs on 8 GB RAM + any CPU |

---

## 2. Architecture Design

### 2.1 High-Level Data Flow

```
                    ┌─────────────┐
                    │  Screenshot  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Layer 1   │  PREPROCESSING
                    │  Normalizer │  Scale · Denoise · Sharpen
                    └──┬───────┬──┘
                       │       │
              ┌────────▼──┐ ┌──▼────────┐
              │  Layer 2   │ │  Layer 3   │  (run in parallel)
              │ DETECTION  │ │    OCR     │
              │   YOLO26   │ │ PaddleOCR  │
              └────────┬──┘ └──┬────────┘
                       │       │
                    ┌──▼───────▼──┐
                    │   Layer 4    │  LAYOUT ANALYSIS
                    │  Merge · Classify · Hierarchy
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Layer 4b   │  VLM CAPTIONING (GPU only)
                    │ Florence-2  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Layer 5    │  CONTEXT ASSEMBLY
                    │  Schema · Format · Token Budget
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Optional   │  RESEARCH HELPER
                    │  Web Search │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Context Pack │  → LLM Prompt
                    └─────────────┘
```

### 2.2 Key Design Decisions

1. **Detection and OCR run independently** on the preprocessed image, then results are merged in the layout layer.  This is more robust than sequential approaches because OCR doesn't depend on detection accuracy.

2. **VLM is optional** and runs only in the "full" tier.  Most value comes from OCR + layout alone; the VLM adds scene-level understanding for ambiguous cases.

3. **No embedding step**.  The original proposal included a vector embedding stage.  After analysis, this was deferred because:
   - For single-screenshot use, structured JSON is more useful to LLMs than embeddings
   - Embeddings add latency (~50-100 ms) without clear benefit for the primary use case
   - If multi-screenshot memory/RAG is needed later, embeddings can be added as a post-processing step

4. **CNN replaced with VLM**.  The original architecture proposed a custom CNN for "context, shapes, placement and app info."  A pretrained VLM (Florence-2, 0.23B params) handles all these tasks and more, without requiring custom training data or compute for training.

---

## 3. Pipeline Layers — Detailed Design

### 3.1 Layer 1: Preprocessing (`irpai.preprocessing.normalizer`)

**Purpose:** Normalize the input image for consistent downstream processing.

| Step | Method | Config |
|------|--------|--------|
| Resolution scaling | Lanczos downscale | `target_max_dimension=1920` |
| Denoising | OpenCV `fastNlMeansDenoisingColored` | `denoise_strength=3` |
| Sharpening | PIL `UnsharpMask(radius=1.5, percent=60)` | `sharpen=True` |
| OCR prep | Grayscale conversion | `grayscale_ocr=True` |
| Threshold (optional) | OpenCV adaptive Gaussian threshold | `adaptive_threshold=False` |

**Output:** `NormalizedImage` with `.color` (RGB) and `.ocr_ready` (grayscale) variants, plus scale metadata.

**Latency:** <10 ms for a 1920px image.

### 3.2 Layer 2: Detection (`irpai.detection.yolo_detector`)

**Purpose:** Identify UI elements (buttons, menus, headers, icons, text fields).

| Component | Details |
|-----------|---------|
| Model | YOLO26n (2.4M params) via Ultralytics |
| Input | RGB image from preprocessing |
| Output | Bounding boxes + class labels + confidence scores |
| Fallback | `FallbackGridDetector` — divides image into NxM grid |
| Classes | 19 UI element types (button, text_field, icon, menu, header, toolbar, sidebar, tab, dropdown, checkbox, radio, slider, card, modal, tooltip, nav_bar, status_bar, content_area, image) |

**Notes:**
- Out of the box, YOLO uses COCO classes mapped to approximate UI roles via `_COCO_TO_UI`
- For production use, fine-tune on UI-element datasets (Rico, WebUI, or custom annotations)
- ONNX export recommended for CPU inference consistency

**Latency:** ~40 ms CPU / ~2 ms GPU (YOLO26n).

### 3.3 Layer 3: OCR (`irpai.ocr`)

**Purpose:** Extract all readable text with positional metadata.

| Engine | Params | License | GPU | Best For |
|--------|--------|---------|-----|----------|
| PaddleOCR PP-OCRv5 | ~15M | Apache 2.0 | Optional | General screenshots, 111 languages |
| Tesseract 5.5 | ~20M/lang | Apache 2.0 | No | Clean printed text, fallback |

**Output hierarchy:** `OCRResult` → `OCRBlock[]` → `OCRLine[]` → `OCRWord[]`

Each word carries its bounding box, and lines are merged into paragraph-level blocks by vertical proximity (`y_gap=15px`).

**Engine selection:**
- `create_ocr_engine()` factory tries the configured primary engine first
- Falls back to secondary if import fails
- Raises `RuntimeError` only if neither is available

**Latency:** ~100-200 ms CPU (PaddleOCR), ~50-100 ms (Tesseract).

### 3.4 Layer 4: Layout Analysis (`irpai.layout`)

**Purpose:** Merge detection + OCR into semantic regions with roles and hierarchy.

**Steps:**

1. **Map OCR blocks → detection regions** — if a text block's centre falls inside a detection bounding box, they are associated.

2. **Classify unmatched text blocks** — heuristic analysis using:
   - Font size vs. median (≥1.4× median → heading)
   - Text length (short single-line → label)
   - ALL CAPS detection (→ heading)

3. **Merge nearby regions** — adjacent regions with the same role within `merge_distance_x/y` pixels are fused.

4. **Compute reading order** — column-aware sort:
   - Cluster region centres into columns (threshold: 15% of image width)
   - Sort within each column top-to-bottom
   - Read columns left-to-right (or RTL if configured)

5. **Build hierarchy** — `HierarchyBuilder` creates a parent→child tree:
   - Sort by area (largest first)
   - For each region, find the smallest existing region that contains it
   - Containment: child's centre inside parent's box AND parent ≥1.5× child area

**Output:** `LayoutResult` with `regions[]`, `reading_order[]`, and `image_size`.

### 3.5 Layer 4b: VLM Captioning (`irpai.vlm.florence`)

**Purpose:** Add natural-language scene understanding (full tier only).

| Model | Params | VRAM | License | Tasks |
|-------|--------|------|---------|-------|
| Florence-2-base | 0.23B | ~2 GB FP16 | MIT | Caption, Detailed Caption, OD, OCR with Region |

**Integration:**
- Lazy-loaded on first use via HuggingFace Transformers
- Runs up to 4 task prompts per image
- Outputs parsed into `VLMResult` with structured captions + objects + OCR text
- Errors are caught and logged — VLM failure never blocks the pipeline

**Latency:** ~300-500 ms GPU / ~2-5 s CPU.

### 3.6 Layer 5: Context Assembly (`irpai.context`)

**Purpose:** Package everything into a token-efficient `ContextPack`.

**Steps:**

1. **Convert regions** — `SemanticRegion` → `Region` (schema type) in reading order

2. **Build hierarchy** — serialise the `LayoutNode` tree to nested dicts

3. **Generate summary:**
   - App-type detection via regex patterns against OCR text (IDE, browser, terminal, chat, email, …)
   - Element count summary ("3× body, 2× heading, 1× button")
   - Natural-language description combining VLM caption + headings + stats

4. **Token budget enforcement:**
   - Estimate token count (chars / 4)
   - If over budget, truncate longest `body`/`unknown` regions first
   - Headings and labels are preserved intact

**Output formats:**
- `pack.to_json()` — compact or pretty JSON
- `pack.to_markdown()` — readable Markdown for chat injection
- `format_for_llm(pack, style="system_prompt")` — XML-tagged wrapper

### 3.7 Research Helper (`irpai.research.helper`)

**Purpose:** Fetch brief supporting context from the web when the screenshot contains error messages, URLs, or unfamiliar UI.

**Query extraction:**
- Error/exception text (first 80 chars)
- URLs visible on screen
- Heading text >5 chars

**Search backends:**
- DuckDuckGo Instant Answer API (no key required)
- Serper.dev (Google Search API, key required)

**Latency:** 1-3 s per query (network-dependent).

---

## 4. Component Selection & Benchmarks

### 4.1 OCR Engines

| Engine | Params | CPU Speed | GPU Speed | Accuracy (clean text) | Layout Understanding | License |
|--------|--------|-----------|-----------|-----------------------|---------------------|---------|
| **PaddleOCR PP-OCRv5** | ~15M | ~150 ms | ~50 ms | 95%+ | Paragraph grouping | Apache 2.0 |
| **Tesseract 5.5** | ~20M/lang | ~80 ms | N/A | 90%+ | Line-level only | Apache 2.0 |
| PaddleOCR-VL-1.5 | 0.9B | — | ~300 ms | 94.5% (OmniDocBench) | Full document parsing | Apache 2.0 |
| EasyOCR | ~30M | ~200 ms | ~80 ms | 88%+ | None | Apache 2.0 |

**Selected:** PaddleOCR (primary) + Tesseract (fallback).

### 4.2 UI Detection Models

| Model | Params | CPU (ms) | GPU (ms) | mAP (COCO) | License |
|-------|--------|----------|----------|------------|---------|
| **YOLO26n** | 2.4M | 39 | 1.7 | 40.9 | AGPL-3.0 |
| YOLO26s | 9.5M | 87 | 2.5 | 48.6 | AGPL-3.0 |
| YOLO26m | 20.4M | 220 | 4.7 | 53.1 | AGPL-3.0 |
| YOLOv8n | 3.2M | 45 | 2.0 | 37.3 | AGPL-3.0 |

**Selected:** YOLO26n for standard tier (fastest on CPU, smallest model).

> **Licensing note:** YOLO is AGPL-3.0 for open-source use or requires an Ultralytics Enterprise license for proprietary distribution.  If this is a concern, consider ONNX export + a custom wrapper, or use the MIT-licensed fallback grid detector.

### 4.3 Vision-Language Models

| Model | Params | Min VRAM | Screenshot Capability | License |
|-------|--------|----------|----------------------|---------|
| **Florence-2-base** | 0.23B | ~2 GB | Good (OD, OCR, captioning, grounding) | MIT |
| Florence-2-large | 0.77B | ~4 GB | Better | MIT |
| PaddleOCR-VL-1.5 | 0.9B | ~4 GB | Excellent (document-specific) | Apache 2.0 |
| BLIP-2 OPT-2.7B | 3.8B | ~7 GB (INT4) | Moderate | MIT |
| Moondream2 | 1.9B | ~4 GB | Moderate | Apache 2.0 |

**Selected:** Florence-2-base (smallest, MIT license, multi-task capable).

---

## 5. Tiered Execution Model

### 5.1 Tier Definitions

```
┌─────────────────────────────────────────────────────────┐
│ LITE (CPU-only, minimal dependencies)                    │
│                                                          │
│ ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐  │
│ │ Preproc  │→ │ Grid Det  │→ │Tesseract │→ │Context │  │
│ │ (PIL/CV2)│  │ (rule)    │  │ (CPU)    │  │Assembly│  │
│ └──────────┘  └───────────┘  └──────────┘  └────────┘  │
├─────────────────────────────────────────────────────────┤
│ STANDARD (CPU, ML models)                                │
│                                                          │
│ ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐  │
│ │ Preproc  │→ │ YOLO26n   │→ │PaddleOCR │→ │Layout  │→ │
│ │ (PIL/CV2)│  │ (ONNX/CPU)│  │ (CPU)    │  │+Context│  │
│ └──────────┘  └───────────┘  └──────────┘  └────────┘  │
├─────────────────────────────────────────────────────────┤
│ FULL (GPU, VLM-enhanced)                                 │
│                                                          │
│ ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐  │
│ │ Preproc  │→ │ YOLO26n   │→ │PaddleOCR │→ │Layout  │→ │
│ │ (PIL/CV2)│  │ (GPU)     │  │ (GPU)    │  │+VLM    │→ │
│ └──────────┘  └───────────┘  └──────────┘  │Florence │  │
│                                             │+Context │  │
│                                             └────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Hardware Requirements

| Tier | CPU | RAM | GPU VRAM | Disk (models) | OS |
|------|-----|-----|----------|---------------|-----|
| **Lite** | Any | 4 GB | — | ~30 MB | Any |
| **Standard** | 4+ cores | 8 GB | — | ~200 MB | Any |
| **Full** | 4+ cores | 16 GB | 4-6 GB | ~2 GB | Linux/macOS/Windows |

**Apple Silicon note:** All tiers work on M1/M2/M3 Macs.  The "full" tier uses MPS acceleration via PyTorch.

### 5.3 Latency Budget

| Layer | Lite | Standard | Full |
|-------|------|----------|------|
| Preprocessing | 5 ms | 5 ms | 5 ms |
| Detection | 1 ms | 40 ms | 2 ms |
| OCR | 80 ms | 150 ms | 80 ms |
| Layout | 2 ms | 5 ms | 5 ms |
| VLM | — | — | 400 ms |
| Context Assembly | 1 ms | 2 ms | 2 ms |
| **Total** | **~90 ms** | **~200 ms** | **~500 ms** |

---

## 6. Output Schema Specification

### 6.1 ContextPack JSON Schema

```json
{
  "summary": {
    "description": "string — one-paragraph natural-language summary",
    "app_type": "string — detected app: ide|browser|terminal|chat|email|…",
    "key_elements": ["string — e.g. '3× body', '2× heading'"],
    "vlm_caption": "string — raw Florence-2 caption (if available)"
  },
  "regions": [
    {
      "role": "string — heading|body|button|nav|icon|label|input|…",
      "text": "string — extracted text content",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "properties": {"font_size": 24},
      "children": [ /* nested regions */ ]
    }
  ],
  "hierarchy": {
    "roots": [
      {
        "role": "container",
        "bbox": [0, 0, 800, 600],
        "children": [ /* nested tree */ ]
      }
    ]
  },
  "meta": {
    "irpai_version": "0.1.0",
    "tier": "standard",
    "ocr_engine": "PaddleOCR",
    "detection_model": "YOLO (yolo26n)",
    "vlm_model": "none",
    "total_ms": 203.4,
    "image_size": [1920, 1080]
  }
}
```

### 6.2 Semantic Roles

| Role | Description | Typical Source |
|------|-------------|---------------|
| `heading` | Title or section header | OCR (large font) / Detection |
| `body` | Paragraph or content text | OCR |
| `button` | Clickable button | Detection |
| `label` | Short text label (<25 chars) | OCR heuristic |
| `nav` | Navigation bar or menu | Detection |
| `icon` | Graphical icon | Detection |
| `input` | Text input field | Detection |
| `image` | Non-text image area | Detection |
| `sidebar` | Side panel | Detection |
| `toolbar` | Toolbar row | Detection |
| `tab` | Tab element | Detection |
| `card` | Card-style container | Detection |
| `modal` | Overlay/dialog | Detection |
| `status_bar` | Status/info bar | Detection |
| `container` | Generic wrapper | Hierarchy builder |
| `unknown` | Unclassified | Fallback |

---

## 7. Feasibility Analysis

### 7.1 Accuracy Assessment

| Scenario | Expected Accuracy | Notes |
|----------|------------------|-------|
| Clean IDE screenshots | 90%+ | High-contrast, standard fonts |
| Browser pages | 85%+ | Variable layouts, web fonts |
| Mobile app screenshots | 80%+ | Smaller text, more icons |
| Terminal / dark themes | 85%+ | Good contrast, monospace fonts |
| Screenshots with overlays | 70%+ | Modals/tooltips may confuse detection |
| Low-resolution / blurry | 60-70% | OCR accuracy degrades |

**Target: 80% "good enough" accuracy** — IrpAI will correctly identify the major screen sections, read >80% of visible text, and produce a layout that lets an LLM understand the screen without additional user explanation.

### 7.2 Compute Feasibility

**CPU-only (Standard tier):**
- PaddleOCR: ~15M params, ~100 MB memory, CPU-optimised
- YOLO26n: 2.4M params, ~10 MB memory, 39 ms on CPU
- Total pipeline memory: <500 MB
- Total inference time: <300 ms
- **Verdict: Fully feasible on any modern laptop**

**GPU (Full tier):**
- Florence-2-base: 0.23B params, ~500 MB FP16 VRAM
- PaddleOCR + YOLO: ~50 MB additional
- Total VRAM: ~2-3 GB
- **Verdict: Fits comfortably on RTX 3060 (12 GB) or M1 (8 GB unified)**

### 7.3 Licensing

| Component | License | Commercial Use |
|-----------|---------|---------------|
| PaddleOCR | Apache 2.0 | ✅ |
| Tesseract | Apache 2.0 | ✅ |
| Florence-2 | MIT | ✅ |
| Ultralytics YOLO | AGPL-3.0 | ⚠️ Requires Enterprise license |
| Pillow / OpenCV | MIT / Apache | ✅ |
| PyTorch / Transformers | BSD / Apache | ✅ |

**Main concern:** YOLO's AGPL license requires either open-sourcing your full application or purchasing an Enterprise license.  If this is a blocker, alternatives include:
- Export YOLO weights to ONNX and wrap with custom inference code
- Use Florence-2's built-in object detection (`<OD>` task) instead of a separate YOLO model
- Fine-tune a custom detector with a permissive license

---

## 8. Improvements & Recommendations

### 8.1 Architecture Improvements over Original Proposal

| Original Proposal | Recommendation | Reason |
|-------------------|---------------|--------|
| Custom CNN for context/shapes | **Replace with Florence-2 VLM** | Pretrained 0.23B VLM handles detection + captioning + OCR in one model; no training needed |
| Explicit embedding step | **Defer — use structured JSON** | For single-screenshot use, JSON is more useful to LLMs than vector embeddings |
| Single-mode pipeline | **Tiered execution (3 tiers)** | Allows users with different hardware to use the best pipeline for their setup |
| Monolithic processing | **Modular layers with abstract interfaces** | Each layer can be independently upgraded, swapped, or optimised |

### 8.2 Short-Term Improvements

1. **Fine-tune YOLO on UI data** — Train on Rico (Android) + WebUI datasets for proper UI element classes instead of COCO mapping.  Expected improvement: +20-30% detection accuracy on screenshots.

2. **Parallel execution** — Run detection and OCR concurrently (they're independent).  Expected speedup: ~30% on Standard tier.

3. **Region-level VLM captioning** — Instead of captioning the full image, run Florence-2 on individual detected icons/images for richer descriptions.

4. **Confidence-based fallback** — If PaddleOCR confidence < 0.5 on a region, re-run that region with Tesseract as a cross-check.

### 8.3 Medium-Term Improvements

5. **PaddleOCR-VL-1.5 integration** — Replace both PaddleOCR + Florence-2 with the 0.9B VLM that handles OCR + layout + captioning in a single model (Apache 2.0).

6. **Multi-screenshot memory** — Add an embedding store (FAISS/ChromaDB) for RAG across multiple screenshots in a session.  This is where the originally proposed embedding step becomes valuable.

7. **Active-learning annotation tool** — Ship a simple UI that lets users correct IrpAI's output, generating training data for fine-tuning.

8. **ONNX Runtime optimisation** — Export all models to ONNX for consistent cross-platform performance and eliminate PyTorch dependency for inference.

### 8.4 Long-Term Vision

9. **Real-time screen monitoring** — Watch the screen continuously and maintain a live context window for the LLM.

10. **Multi-modal RAG** — Combine screenshot context with clipboard text, file contents, and web search for comprehensive LLM context.

---

## 9. Comparison with Existing Systems

| Feature | **IrpAI** | OmniParser v2 | MinerU | PaddleOCR-VL |
|---------|-----------|---------------|--------|--------------|
| **Target** | Screenshots → LLM context | Screenshots → GUI agents | PDFs → Markdown | Documents → structured |
| **OCR** | PaddleOCR + Tesseract | — | PaddleOCR | Built-in VLM |
| **Detection** | YOLO26n | YOLO + Florence | DocLayout-YOLO | Built-in VLM |
| **VLM** | Florence-2 (optional) | Florence-2 | Optional | PaddleOCR-VL (0.9B) |
| **Output** | JSON/Markdown context pack | Structured screen elements | Markdown | Markdown/JSON |
| **Min hardware** | CPU, 4 GB RAM | GPU, 8 GB VRAM | CPU, 16 GB RAM | GPU, 6 GB VRAM |
| **License** | MIT | CC-BY-4.0 / AGPL | AGPL-3.0 | Apache 2.0 |
| **Strengths** | Lightweight, LLM-optimised | GUI agent integration | PDF/document focus | Best accuracy |
| **Weaknesses** | Needs UI fine-tuning | Heavy, GPU-only | Not screenshot-focused | GPU required |

**IrpAI's differentiator:** Specifically optimised for **LLM context generation** from screenshots, with the lightest hardware requirements and a tiered model supporting CPU-only operation.

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Current — v0.1)
- [x] Core pipeline architecture with 5 layers
- [x] Tiered execution model (lite / standard / full)
- [x] PaddleOCR + Tesseract OCR backends
- [x] YOLO26 detection + grid fallback
- [x] Layout analysis with reading order
- [x] Context pack schema + JSON/Markdown output
- [x] Florence-2 VLM integration
- [x] Research helper module
- [x] CLI interface
- [x] Test suite
- [ ] Fine-tune YOLO on UI element dataset

### Phase 2: Optimisation (v0.2)
- [ ] Parallel detection + OCR execution
- [ ] ONNX Runtime model serving
- [ ] Region-level VLM captioning
- [ ] Confidence-based OCR cross-checking
- [ ] Token budget optimisation with importance scoring
- [ ] Benchmarking suite on ScreenSpot / OmniDocBench

### Phase 3: Intelligence (v0.3)
- [ ] PaddleOCR-VL-1.5 unified backend
- [ ] Multi-screenshot session memory (embeddings + RAG)
- [ ] Active learning annotation tool
- [ ] Real-time screen monitoring mode

### Phase 4: Integration (v1.0)
- [ ] SDK for popular LLM frameworks (LangChain, LlamaIndex)
- [ ] Browser extension for instant context
- [ ] VS Code extension
- [ ] API server mode (FastAPI)

---

*This document is maintained alongside the codebase and should be updated as the architecture evolves.*
