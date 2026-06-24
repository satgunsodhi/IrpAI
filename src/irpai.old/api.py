"""
IrpAI REST API Server

FastAPI-based REST API for screenshot analysis.

Endpoints:
- POST /api/v1/analyze - Analyze screenshot with optional query
- POST /api/v1/context - Extract structured context only
- GET  /api/v1/health  - Health check
- GET  /api/v1/info    - System info
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from irpai import __version__
from irpai.config import PipelineConfig
from irpai.pipeline import IrpAIPipeline

logger = logging.getLogger(__name__)

# ── App Setup ───────────────────────────────────────────────

app = FastAPI(
    title="IrpAI API",
    description="Image Resolution Pipeline AI - Screenshot analysis for LLM understanding",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy-loaded)
_pipeline: Optional[IrpAIPipeline] = None


def get_pipeline() -> IrpAIPipeline:
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        config = PipelineConfig()
        _pipeline = IrpAIPipeline(config)
    return _pipeline


# ── Request/Response Models ─────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for analyze endpoint (JSON mode)."""

    image: str = Field(..., description="Base64-encoded screenshot image")
    query: Optional[str] = Field(None, description="Question about the screenshot")
    return_structured_context: bool = Field(
        False, description="Include full structured context in response"
    )
    use_llm: bool = Field(True, description="Generate LLM response")


class AnalyzeResponse(BaseModel):
    """Response from analyze endpoint."""

    answer: Optional[str] = Field(None, description="LLM-generated answer")
    enhanced_prompt: Optional[str] = Field(None, description="Enhanced prompt sent to LLM")
    platform: str = Field(..., description="Detected platform")
    app_title: str = Field(..., description="Detected application title")
    summary: str = Field(..., description="Screenshot summary")
    all_text: str = Field(..., description="All extracted text")
    total_elements: int = Field(..., description="Number of detected UI elements")
    total_text_regions: int = Field(..., description="Number of detected text regions")
    processing_time_ms: float = Field(..., description="Total processing time in ms")
    stage_timings: dict[str, float] = Field(
        default_factory=dict, description="Per-stage timing breakdown"
    )
    confidence: float = Field(..., description="Overall confidence score")
    warnings: list[str] = Field(default_factory=list, description="Processing warnings")
    structured_context: Optional[dict] = Field(
        None, description="Full structured context (if requested)"
    )


class ContextResponse(BaseModel):
    """Response from context-only endpoint."""

    platform: str
    app_title: str
    layout: dict
    all_text: str
    metadata: dict
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float


# ── Track startup time ──────────────────────────────────────

_start_time = time.time()


# ── Endpoints ───────────────────────────────────────────────


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/api/v1/info")
async def system_info():
    """Return system information and available components."""
    components = {}

    try:
        import torch

        components["pytorch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
        }
    except ImportError:
        components["pytorch"] = {"available": False}

    try:
        import cv2

        components["opencv"] = {"available": True, "version": cv2.__version__}
    except ImportError:
        components["opencv"] = {"available": False}

    try:
        import paddleocr

        components["paddleocr"] = {"available": True}
    except ImportError:
        components["paddleocr"] = {"available": False}

    return {
        "name": "IrpAI",
        "version": __version__,
        "components": components,
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_screenshot(request: AnalyzeRequest):
    """
    Analyze a screenshot and optionally answer a query.

    Accepts base64-encoded image and optional question.
    """
    pipeline = get_pipeline()

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(request.image)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Process
    try:
        result = pipeline.analyze(
            img,
            query=request.query,
            use_llm=request.use_llm and request.query is not None,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    ctx = result.structured_context

    response = AnalyzeResponse(
        answer=result.llm_response,
        enhanced_prompt=result.enhanced_prompt,
        platform=ctx.platform,
        app_title=ctx.app_title,
        summary=ctx.to_summary(),
        all_text=ctx.all_text,
        total_elements=len(ctx.ui_detections),
        total_text_regions=len(ctx.text_extractions),
        processing_time_ms=result.processing_time_ms,
        stage_timings=result.stage_timings,
        confidence=result.confidence,
        warnings=result.warnings,
        structured_context=ctx.model_dump() if request.return_structured_context else None,
    )

    return response


@app.post("/api/v1/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(
    file: UploadFile = File(..., description="Screenshot image file"),
    query: Optional[str] = Form(None, description="Question about the screenshot"),
    return_structured_context: bool = Form(False),
    use_llm: bool = Form(True),
):
    """
    Analyze a screenshot uploaded as a file.

    Accepts multipart form data with image file.
    """
    pipeline = get_pipeline()

    # Read uploaded file
    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Process
    try:
        result = pipeline.analyze(
            img,
            query=query,
            use_llm=use_llm and query is not None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    ctx = result.structured_context

    response = AnalyzeResponse(
        answer=result.llm_response,
        enhanced_prompt=result.enhanced_prompt,
        platform=ctx.platform,
        app_title=ctx.app_title,
        summary=ctx.to_summary(),
        all_text=ctx.all_text,
        total_elements=len(ctx.ui_detections),
        total_text_regions=len(ctx.text_extractions),
        processing_time_ms=result.processing_time_ms,
        stage_timings=result.stage_timings,
        confidence=result.confidence,
        warnings=result.warnings,
        structured_context=ctx.model_dump() if return_structured_context else None,
    )

    return response


@app.post("/api/v1/context", response_model=ContextResponse)
async def extract_context(
    file: UploadFile = File(..., description="Screenshot image file"),
):
    """
    Extract structured context without LLM interaction.

    Returns the full structured analysis of the screenshot.
    """
    pipeline = get_pipeline()

    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        result = pipeline.analyze(img, query=None, use_llm=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context extraction failed: {e}")

    ctx = result.structured_context

    return ContextResponse(
        platform=ctx.platform,
        app_title=ctx.app_title,
        layout=ctx.layout.model_dump(),
        all_text=ctx.all_text,
        metadata=ctx.metadata,
        processing_time_ms=result.processing_time_ms,
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
