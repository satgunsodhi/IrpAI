"""
IrpAI — Advanced Usage Example

Demonstrates:
    - Custom per-layer configuration
    - Using the VLM tier with Florence-2
    - Accessing individual layer outputs
    - Injecting context into an LLM API call
    - Using the research helper
"""

from pathlib import Path

from irpai import IrpAIConfig, Tier
from irpai.config import (
    DetectionConfig,
    LayoutConfig,
    OCRConfig,
    OutputConfig,
    PreprocessingConfig,
    ResearchConfig,
    VLMConfig,
)
from irpai.pipeline import IrpAIPipeline
from irpai.context.formatter import format_for_llm


def custom_config_example() -> None:
    """Show how to override individual layer configs."""
    config = IrpAIConfig(
        tier=Tier.STANDARD,
        preprocessing=PreprocessingConfig(
            target_max_dimension=1280,  # smaller for faster processing
            denoise_strength=5,
            sharpen=True,
        ),
        ocr=OCRConfig(
            engine="paddleocr",
            fallback_engine="tesseract",
            paragraph_merge=True,
            rec_batch_num=8,
        ),
        detection=DetectionConfig(
            model_name="yolo26s",  # slightly larger model for better accuracy
            confidence_threshold=0.3,
            device="cpu",
        ),
        layout=LayoutConfig(
            heading_font_ratio=1.3,
            column_detection=True,
            reading_order="ltr-ttb",
        ),
        output=OutputConfig(
            format="json",
            include_bboxes=True,
            include_hierarchy=True,
            max_context_tokens=2048,  # tight budget
            compact=True,
        ),
        verbose=True,
    )

    pipeline = IrpAIPipeline(config)
    pack = pipeline.run("screenshot.png")

    print(f"Custom config result: {len(pack.regions)} regions, ~{pack.token_estimate()} tokens")


def full_tier_example() -> None:
    """Show the GPU tier with Florence-2 VLM captioning."""
    config = IrpAIConfig(
        tier=Tier.FULL,
        vlm=VLMConfig(
            model_name="microsoft/Florence-2-base",
            device="cuda",
            dtype="float16",
            tasks=("<CAPTION>", "<DETAILED_CAPTION>", "<OD>"),
        ),
    )

    pipeline = IrpAIPipeline(config)
    pipeline.warmup()  # pre-load models

    pack = pipeline.run("screenshot.png")

    # VLM caption is included in the summary
    print(f"VLM caption: {pack.summary.vlm_caption}")
    print(f"Description: {pack.summary.description}")


def research_example() -> None:
    """Show the research helper fetching context from the web."""
    config = IrpAIConfig(
        tier=Tier.STANDARD,
        research=ResearchConfig(
            enabled=True,
            search_provider="duckduckgo",
            max_results=3,
        ),
    )

    pipeline = IrpAIPipeline(config)
    pack = pipeline.run("screenshot.png")

    # Research results are noted in key_elements
    for elem in pack.summary.key_elements:
        if "Research" in elem:
            print(f"  {elem}")


def llm_injection_example() -> None:
    """Show how to inject context into an LLM API call."""
    pipeline = IrpAIPipeline(IrpAIConfig(tier=Tier.STANDARD))
    pack = pipeline.run("screenshot.png")

    # Format as system prompt
    context = format_for_llm(pack, style="system_prompt")

    # Example: OpenAI-style API call (pseudocode)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.  The user has shared a screenshot.  "
                "Here is the structured context extracted from it:\n\n"
                + context
            ),
        },
        {
            "role": "user",
            "content": "What's happening in this screenshot?  Help me fix the error.",
        },
    ]

    print("LLM messages prepared:")
    print(f"  System prompt: {len(messages[0]['content'])} chars")
    print(f"  User message:  {len(messages[1]['content'])} chars")

    # You would then pass `messages` to your LLM API:
    # response = openai.ChatCompletion.create(model="gpt-4", messages=messages)


def step_by_step_example() -> None:
    """Show how to run individual layers for debugging."""
    from irpai.preprocessing import Normalizer
    from irpai.ocr.engine import create_ocr_engine
    from irpai.detection.yolo_detector import FallbackGridDetector
    from irpai.layout import LayoutAnalyzer
    from irpai.context import ContextAssembler

    # Step 1: Preprocess
    normalizer = Normalizer()
    normalized = normalizer.normalize("screenshot.png")
    print(f"Preprocessed: {normalized}")

    # Step 2: OCR
    ocr_engine = create_ocr_engine()
    ocr_result = ocr_engine.recognize(normalized.ocr_ready)
    print(f"OCR: {len(ocr_result.all_lines)} lines, confidence={ocr_result.mean_confidence:.2f}")

    # Step 3: Detection
    detector = FallbackGridDetector(rows=3, cols=4)
    detections = detector.detect(normalized.color)
    print(f"Detection: {len(detections.detections)} regions")

    # Step 4: Layout analysis
    analyzer = LayoutAnalyzer()
    layout = analyzer.analyze(ocr_result, detections)
    print(f"Layout: {len(layout.regions)} semantic regions")

    # Step 5: Assemble
    assembler = ContextAssembler()
    pack = assembler.assemble(layout, ocr=ocr_result, detections=detections)
    print(f"Context pack: {pack.token_estimate()} tokens")
    print(pack.to_json(compact=False))


if __name__ == "__main__":
    import sys

    examples = {
        "custom": custom_config_example,
        "full": full_tier_example,
        "research": research_example,
        "llm": llm_injection_example,
        "steps": step_by_step_example,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Available examples:")
        for name, fn in examples.items():
            print(f"  python -m examples.advanced_usage {name}  — {fn.__doc__.strip()}")
