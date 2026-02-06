# IrpAI - Image Resolution Pipeline AI

**Machine Vision Preprocessing for Enhanced LLM-Screenshot Interactions**

IrpAI is a lightweight machine vision preprocessing pipeline that extracts structured context from screenshots, enabling any text-based LLM to "understand" visual UI content without native vision capabilities.

## Problem

Even advanced multimodal LLMs struggle with screenshots—users must manually describe UI layout, text content, and spatial relationships. IrpAI eliminates this friction by automatically converting screenshots into structured context.

## Architecture

```
Screenshot (PNG/JPG)
    ↓
[Preprocessing] → Normalized image (orientation, contrast, denoising)
    ↓
[Vision Branch] → UI element bounding boxes + element types
    ↓
[Text Branch]   → OCR text strings + coordinates
    ↓
[Fusion Module] → Structured context JSON (layout, hierarchy, platform)
    ↓
[LLM Interface] → Enhanced prompt → LLM Response
```

## Installation

### From Wheel (Recommended)

```bash
# Install build tools
pip install build

# Build the wheel
python -m build --wheel

# Install the built wheel
pip install dist/irpai-0.1.0-py3-none-any.whl
```

### Development (Editable)

```bash
# Modern editable install (PEP 660 wheel-based)
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Direct Installation

```bash
# Install from source directory
pip install .
```

### Core Dependencies

| Component | Package | Purpose |
|-----------|---------|---------|
| Computer Vision | `opencv-python` | Image preprocessing |
| Deep Learning | `torch`, `torchvision` | UI element detection (Faster R-CNN) |
| OCR | `paddleocr`, `paddlepaddle` | Text extraction |
| Embeddings | `sentence-transformers` | Semantic retrieval (optional) |
| LLM Interface | `openai` / `anthropic` | LLM response generation |
| CLI | `typer`, `rich` | Command-line interface |
| API | `fastapi`, `uvicorn` | REST API server |

## Quick Start

### Python API

```python
from irpai.pipeline import IrpAIPipeline

# Initialize pipeline
pipeline = IrpAIPipeline()

# Analyze a screenshot with a query
result = pipeline.analyze(
    "screenshot.png",
    query="What error is shown?",
    use_llm=True  # Requires OPENAI_API_KEY
)

print(result.llm_response)
print(result.structured_context.to_summary())
print(f"Processed in {result.processing_time_ms:.0f}ms")
```

### Extract Context Only (No LLM)

```python
context = pipeline.extract_context("screenshot.png")
print(context.platform)        # "web_browser"
print(context.app_title)       # "GitHub - user/repo"
print(context.all_text)        # All extracted text
print(context.layout.navbar)   # Navbar elements
```

### CLI

```bash
# Analyze a single screenshot
irpai analyze screenshot.png --query "What button should I click?"

# Batch process a directory
irpai batch screenshots/ --output results/

# Interactive mode (multiple questions, one screenshot)
irpai interactive screenshot.png

# Generate default config
irpai init-config --output my_config.json

# Check system info
irpai info
```

### REST API

```bash
# Start the server
uvicorn irpai.api:app --host 0.0.0.0 --port 8000

# Or use Python
python -m irpai.api
```

```bash
# Upload and analyze
curl -X POST http://localhost:8000/api/v1/analyze/upload \
  -F "file=@screenshot.png" \
  -F "query=What error is shown?"

# JSON mode (base64 image)
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "query": "Describe this screen"}'
```

## Configuration

```python
from irpai.config import PipelineConfig

config = PipelineConfig(
    # Device selection
    device="auto",  # "cpu", "gpu", "auto"
    
    # Preprocessing
    preprocessing={"max_dimension": 1920, "enable_denoising": True},
    
    # Vision (UI Detection)
    vision={"confidence_threshold": 0.5, "use_quantized": False},
    
    # OCR
    ocr={"engine": "paddleocr", "language": "en", "use_selective_ocr": True},
    
    # Fusion
    fusion={"enable_platform_detection": True, "enable_spatial_relationships": True},
    
    # LLM
    llm={
        "approach": "direct_injection",  # or "embedding_retrieval"
        "llm_provider": "openai",        # or "anthropic", "local"
        "llm_model": "gpt-4",
        "enable_pii_redaction": True,
    },
    
    # Pipeline
    enable_caching=True,
    enable_async=True,
    verbose=False,
)
```

## Pipeline Stages

### 1. Preprocessing
- Orientation detection & correction
- Resolution normalization (max 1920px)
- Bilateral filter denoising
- CLAHE contrast enhancement

### 2. Vision Branch (UI Detection)
- **Neural**: Faster R-CNN with FPN (ResNet-50 backbone)
- **Heuristic**: Contour analysis fallback for CPU-only environments
- Detects: buttons, text inputs, text blocks, images, menus, modals, containers, icons

### 3. Text Branch (OCR)
- **Primary**: PaddleOCR (97%+ accuracy, 30MB model)
- **Fallback**: EasyOCR, Tesseract
- Selective OCR: Uses UI detections as region proposals (3-5x faster)
- Ensemble mode: Multi-OCR voting for maximum accuracy
- PII redaction: Automatic detection and masking of sensitive data

### 4. Fusion Module
- Spatial-text alignment (assigns text to UI elements)
- Layout region detection (navbar, sidebar, main content, footer, modal)
- Platform & application detection
- Spatial relationship inference
- Context-aware post-processing

### 5. LLM Interface
- **Direct Injection**: Full context serialized into natural language prompt
- **Embedding Retrieval**: Semantic search for query-relevant UI elements
- Supports: OpenAI, Anthropic, local models (via Ollama/LangChain)

## Performance Targets

| Metric | MVP Target | Production Target |
|--------|-----------|------------------|
| End-to-End Accuracy | >70% | >85% |
| Latency (CPU) | <5s | <3s |
| Latency (GPU) | <2s | <1s |
| Memory Usage | <2GB | <1GB |
| OCR Accuracy | >90% | >95% |
| UI Detection mAP@0.5 | >0.60 | >0.68 |

## Project Structure

```
IrpAI/
├── src/irpai/
│   ├── __init__.py          # Package metadata
│   ├── config.py            # Pydantic configuration models
│   ├── models.py            # Data models (BoundingBox, UIDetection, etc.)
│   ├── preprocessing.py     # Stage 1: Image normalization
│   ├── vision.py            # Stage 2: UI element detection
│   ├── ocr.py               # Stage 3: Text extraction
│   ├── fusion.py            # Stage 4: Context structuring
│   ├── llm_interface.py     # Stage 5: LLM prompt building
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── cli.py               # CLI (Typer)
│   └── api.py               # REST API (FastAPI)
├── tests/
│   └── test_pipeline.py     # Test suite
├── pyproject.toml            # Project config & dependencies
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=irpai --cov-report=html
```

## License

MIT
