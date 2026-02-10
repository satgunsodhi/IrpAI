"""
IrpAI CLI

Simple command-line interface for running the pipeline on screenshots.

Usage::

    irpai screenshot.png
    irpai screenshot.png --tier full --format markdown
    irpai screenshot.png --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from irpai.config import IrpAIConfig, Tier
from irpai.pipeline import IrpAIPipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="irpai",
        description="IrpAI — Convert screenshots into structured LLM context.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the screenshot image file.",
    )
    parser.add_argument(
        "--tier",
        choices=["lite", "standard", "full"],
        default="standard",
        help="Execution tier (default: standard).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "system_prompt"],
        default="json",
        dest="output_format",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write output to file instead of stdout.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="Compact JSON output (default).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum context tokens (default: 4096).",
    )

    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Build config
    tier = Tier(args.tier)
    config = IrpAIConfig(tier=tier, verbose=args.verbose)

    # Override output settings
    from irpai.config import OutputConfig
    config.output = OutputConfig(
        format=args.output_format if args.output_format != "system_prompt" else "json",
        compact=not args.pretty,
        max_context_tokens=args.max_tokens,
    )

    # Run pipeline
    pipeline = IrpAIPipeline(config)
    pack = pipeline.run(str(args.image))

    # Format output
    from irpai.context.formatter import format_for_llm
    output = format_for_llm(pack, style=args.output_format)

    if args.pretty and args.output_format == "json":
        output = json.dumps(json.loads(output), indent=2, ensure_ascii=False)

    # Write output
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
