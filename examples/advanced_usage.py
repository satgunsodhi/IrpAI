"""
IrpAI — Advanced Usage Example

Demonstrates:
    - Custom per-layer configuration
    - Using a fine-tuned RT-DETR checkpoint
    - Running individual pipeline layers (step-by-step)
    - Injecting context into an LLM API call
    - Controlling token budget and output format
"""

from pathlib import Path

from irpai import IrpAIConfig, Tier
from irpai.config import (
    DetectionConfig,
    LayoutConfig,
    OCRConfig,
    OutputConfig,
    PreprocessingConfig,
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
            engine="easyocr",
            paragraph_merge=True,
        ),
        detection=DetectionConfig(
            use_ml=True,
            confidence_threshold=0.3,
            structural_boost=True,
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


def fine_tuned_checkpoint_example() -> None:
    """Show how to use a locally fine-tuned RT-DETR checkpoint."""
    # After running examples/training/run_mobile_training.py, a best checkpoint
    # is saved to checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best.
    # IrpAIConfig auto-detects this path; you can also set it explicitly:
    checkpoint = Path("checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best")

    config = IrpAIConfig(
        tier=Tier.STANDARD,
        detection=DetectionConfig(
            use_ml=True,
            model_id=str(checkpoint) if checkpoint.exists() else "PekingU/rtdetr_r18vd_coco_o365",
            confidence_threshold=0.35,
            structural_boost=True,
            device="cpu",
        ),
    )

    pipeline = IrpAIPipeline(config)
    pipeline.warmup()  # pre-load models to avoid first-call latency

    pack = pipeline.run("screenshot.png")
    print(f"Fine-tuned model result: {len(pack.regions)} regions, ~{pack.token_estimate()} tokens")
    print(f"Detection model: {pack.meta.detection_model}")


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
                "Here is the structured context extracted from it:\n\n" + context
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


def token_budget_example() -> None:
    """Show how to control token budget and bbox/hierarchy inclusion."""
    pipeline = IrpAIPipeline(IrpAIConfig(tier=Tier.LITE))
    pack = pipeline.run("screenshot.png")

    print(f"Full output:          ~{pack.token_estimate()} tokens")

    # Omitting bboxes saves 7–13% tokens
    json_no_bbox = pack.to_json(include_bboxes=False)
    print(f"Without bboxes:       ~{len(json_no_bbox) // 4} tokens")

    # Omitting hierarchy saves 35–56% tokens for complex layouts
    json_no_hier = pack.to_json(include_hierarchy=False)
    print(f"Without hierarchy:    ~{len(json_no_hier) // 4} tokens")

    # Markdown format is the most compact
    md = pack.to_markdown()
    print(f"Markdown format:      ~{len(md) // 4} tokens")


def step_by_step_example() -> None:
    """Show how to run individual layers for debugging or custom orchestration."""
    from irpai.config import (
        PreprocessingConfig,
        OCRConfig,
        DetectionConfig,
        LayoutConfig,
        OutputConfig,
    )
    from irpai.preprocessing.normalizer import Normalizer
    from irpai.ocr.engine import create_ocr_engine
    from irpai.detection.hybrid_detector import HybridDetector
    from irpai.layout.analyzer import LayoutAnalyzer
    from irpai.context.assembler import ContextAssembler
    from irpai.context.schema import PipelineMetadata

    # Step 1: Preprocess
    normalizer = Normalizer(PreprocessingConfig())
    normalized = normalizer.normalize("screenshot.png")
    print(f"Preprocessed: {normalized.processed_size}")

    # Step 2: OCR
    ocr_engine = create_ocr_engine(OCRConfig())
    ocr_result = ocr_engine.recognize(normalized.ocr_ready)
    print(f"OCR: {len(ocr_result.all_lines)} lines, confidence={ocr_result.mean_confidence:.2f}")

    # Step 3: Detection (lite path — rule-based fallback only)
    detector = HybridDetector(use_ml=False)
    detections = detector.detect(normalized.color)
    print(f"Detection: {len(detections.detections)} regions")

    # Step 4: Layout analysis
    analyzer = LayoutAnalyzer(LayoutConfig())
    layout = analyzer.analyze(ocr_result, detections)
    print(f"Layout: {len(layout.regions)} semantic regions")

    # Step 5: Assemble
    assembler = ContextAssembler(OutputConfig())
    metadata = PipelineMetadata(
        tier="lite", ocr_engine=ocr_engine.name, detection_model=detector.name
    )
    pack = assembler.assemble(layout, ocr=ocr_result, detections=detections, metadata=metadata)
    print(f"Context pack: {pack.token_estimate()} tokens")
    print(pack.to_json(compact=False))


if __name__ == "__main__":
    import sys

    examples = {
        "custom": custom_config_example,
        "checkpoint": fine_tuned_checkpoint_example,
        "llm": llm_injection_example,
        "tokens": token_budget_example,
        "steps": step_by_step_example,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Available examples:")
        for name, fn in examples.items():
            print(f"  python examples/advanced_usage.py {name}  — {fn.__doc__.strip()}")
