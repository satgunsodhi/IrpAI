"""
IrpAI Command-Line Interface

Provides CLI access to the IrpAI pipeline for:
- Single screenshot analysis
- Batch processing
- Interactive mode
- Configuration management
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="irpai",
    help="IrpAI - Image Resolution Pipeline AI: Analyze screenshots for LLM understanding",
    add_completion=False,
)
console = Console()


def _get_pipeline(config_path: Optional[str] = None, verbose: bool = False):
    """Create pipeline instance with optional config."""
    from irpai.config import PipelineConfig
    from irpai.pipeline import IrpAIPipeline

    if config_path:
        config = PipelineConfig.from_file(config_path)
    else:
        config = PipelineConfig()

    config.verbose = verbose
    return IrpAIPipeline(config)


@app.command()
def analyze(
    image_path: str = typer.Argument(..., help="Path to screenshot image"),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Question about the screenshot"
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON file"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save structured context to JSON file"
    ),
    visualize: Optional[str] = typer.Option(
        None, "--visualize", "-v", help="Save annotated image to file"
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM response generation"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """
    Analyze a single screenshot.

    Examples:
        irpai analyze screenshot.png --query "What error is shown?"
        irpai analyze screenshot.png --output context.json
        irpai analyze screenshot.png --visualize annotated.png
    """
    path = Path(image_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {image_path}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing screenshot...", total=None)

        pipeline = _get_pipeline(config, verbose)
        result = pipeline.analyze(
            str(path),
            query=query,
            use_llm=not no_llm and query is not None,
        )

        progress.update(task, completed=True)

    # Display results
    _display_result(result, query)

    # Save outputs
    if output:
        output_data = result.structured_context.model_dump()
        output_data["pipeline_result"] = {
            "processing_time_ms": result.processing_time_ms,
            "stage_timings": result.stage_timings,
            "confidence": result.confidence,
            "warnings": result.warnings,
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        console.print(f"\n[green]Context saved to {output}[/green]")

    if visualize:
        pipeline.visualize(str(path), result.structured_context, visualize)
        console.print(f"[green]Visualization saved to {visualize}[/green]")


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing screenshots"),
    output_dir: str = typer.Option(
        "irpai_results", "--output", "-o", help="Output directory for results"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Apply same query to all screenshots"
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON file"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """
    Batch process multiple screenshots.

    Example:
        irpai batch screenshots/ --output results/ --query "Describe this screen"
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        console.print(f"[red]Error: Directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    # Find image files
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        console.print(f"[yellow]No image files found in {input_dir}[/yellow]")
        raise typer.Exit(1)

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pipeline = _get_pipeline(config, verbose)

    console.print(f"Processing {len(image_files)} screenshots...")

    results_summary = []

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(image_files))

        for img_file in image_files:
            try:
                result = pipeline.analyze(str(img_file), query=query, use_llm=False)

                # Save individual result
                result_file = out_path / f"{img_file.stem}_context.json"
                with open(result_file, "w") as f:
                    json.dump(
                        result.structured_context.model_dump(),
                        f,
                        indent=2,
                        default=str,
                    )

                results_summary.append(
                    {
                        "file": img_file.name,
                        "status": "success",
                        "elements": len(result.structured_context.ui_detections),
                        "text_regions": len(result.structured_context.text_extractions),
                        "time_ms": result.processing_time_ms,
                        "confidence": result.confidence,
                    }
                )

            except Exception as e:
                results_summary.append(
                    {
                        "file": img_file.name,
                        "status": f"error: {e}",
                        "elements": 0,
                        "text_regions": 0,
                        "time_ms": 0,
                        "confidence": 0,
                    }
                )

            progress.advance(task)

    # Display summary table
    table = Table(title="Batch Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Elements", justify="right")
    table.add_column("Text Regions", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Confidence", justify="right")

    for r in results_summary:
        status_style = "green" if r["status"] == "success" else "red"
        table.add_row(
            r["file"],
            f"[{status_style}]{r['status']}[/{status_style}]",
            str(r["elements"]),
            str(r["text_regions"]),
            f"{r['time_ms']:.0f}",
            f"{r['confidence']:.2f}",
        )

    console.print(table)

    # Save summary
    summary_file = out_path / "batch_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    console.print(f"\n[green]Results saved to {output_dir}/[/green]")


@app.command()
def interactive(
    image_path: str = typer.Argument(..., help="Path to screenshot image"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config JSON file"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """
    Interactive mode - ask multiple questions about a screenshot.

    Example:
        irpai interactive screenshot.png
    """
    path = Path(image_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {image_path}[/red]")
        raise typer.Exit(1)

    pipeline = _get_pipeline(config, verbose)

    # Pre-process the screenshot
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing screenshot...", total=None)
        context = pipeline.extract_context(str(path))
        progress.update(task, completed=True)

    console.print(
        Panel(
            context.to_summary(),
            title=f"Screenshot: {path.name}",
            style="cyan",
        )
    )

    console.print(
        "\n[bold green]Interactive mode.[/bold green] "
        "Type your questions (type 'quit' or 'exit' to stop)\n"
    )

    while True:
        try:
            query = console.input("[bold blue]> [/bold blue]")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("quit", "exit", "q"):
            break

        if not query.strip():
            continue

        # Build prompt and show context-enhanced response
        prompt = pipeline.prompt_builder.build_prompt(query, context)

        try:
            _, response = pipeline.llm_interface.generate_response(query, context)
            console.print(Panel(response, title="IrpAI", style="green"))
        except Exception:
            # Show the enhanced prompt instead
            console.print(
                Panel(
                    f"[yellow]LLM unavailable. Here's the enhanced prompt you can use:[/yellow]\n\n{prompt}",
                    title="IrpAI (Offline)",
                    style="yellow",
                )
            )

    console.print("\n[dim]Goodbye![/dim]")


@app.command()
def init_config(
    output: str = typer.Option(
        "irpai_config.json", "--output", "-o", help="Output config file path"
    ),
):
    """
    Generate a default configuration file.

    Example:
        irpai init-config --output my_config.json
    """
    from irpai.config import PipelineConfig

    config = PipelineConfig()
    config.to_file(output)
    console.print(f"[green]Default config saved to {output}[/green]")
    console.print("[dim]Edit the file to customize pipeline behavior.[/dim]")


@app.command()
def info():
    """Show IrpAI system information and available components."""
    from irpai import __version__

    console.print(
        Panel(
            f"[bold]IrpAI - Image Resolution Pipeline AI[/bold]\n"
            f"Version: {__version__}\n"
            f"Machine Vision Preprocessing for LLM-Screenshot Interactions",
            title="IrpAI",
            style="cyan",
        )
    )

    # Check component availability
    table = Table(title="Component Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # OpenCV
    try:
        import cv2

        table.add_row("OpenCV", "[green]✓ Available[/green]", f"v{cv2.__version__}")
    except ImportError:
        table.add_row("OpenCV", "[red]✗ Missing[/red]", "pip install opencv-python")

    # PyTorch
    try:
        import torch

        gpu = "CUDA" if torch.cuda.is_available() else "CPU only"
        table.add_row("PyTorch", "[green]✓ Available[/green]", f"v{torch.__version__} ({gpu})")
    except ImportError:
        table.add_row("PyTorch", "[red]✗ Missing[/red]", "pip install torch")

    # PaddleOCR
    try:
        import paddleocr

        table.add_row("PaddleOCR", "[green]✓ Available[/green]", "Primary OCR engine")
    except ImportError:
        table.add_row("PaddleOCR", "[yellow]○ Optional[/yellow]", "pip install paddleocr")

    # EasyOCR
    try:
        import easyocr

        table.add_row("EasyOCR", "[green]✓ Available[/green]", "Fallback OCR engine")
    except ImportError:
        table.add_row("EasyOCR", "[yellow]○ Optional[/yellow]", "pip install easyocr")

    # SentenceTransformers
    try:
        import sentence_transformers

        table.add_row("SentenceTransformers", "[green]✓ Available[/green]", "Embedding retrieval")
    except ImportError:
        table.add_row(
            "SentenceTransformers",
            "[yellow]○ Optional[/yellow]",
            "pip install sentence-transformers",
        )

    # OpenAI
    try:
        import openai

        table.add_row("OpenAI SDK", "[green]✓ Available[/green]", "LLM provider")
    except ImportError:
        table.add_row("OpenAI SDK", "[yellow]○ Optional[/yellow]", "pip install openai")

    console.print(table)


def _display_result(result, query: Optional[str] = None):
    """Display pipeline result in a formatted way."""
    ctx = result.structured_context

    # Summary panel
    console.print(
        Panel(
            ctx.to_summary(),
            title=f"{ctx.platform} - {ctx.app_title}",
            style="cyan",
        )
    )

    # Timing info
    timing_table = Table(title="Processing Timings")
    timing_table.add_column("Stage", style="cyan")
    timing_table.add_column("Time (ms)", justify="right")

    for stage, ms in result.stage_timings.items():
        timing_table.add_row(stage, f"{ms:.1f}")

    timing_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{result.processing_time_ms:.1f}[/bold]",
    )
    console.print(timing_table)

    # Confidence
    conf_color = (
        "green" if result.confidence > 0.7 else "yellow" if result.confidence > 0.4 else "red"
    )
    console.print(f"\nConfidence: [{conf_color}]{result.confidence:.1%}[/{conf_color}]")

    # Warnings
    if result.warnings:
        for warning in result.warnings:
            console.print(f"[yellow]⚠ {warning}[/yellow]")

    # LLM response
    if result.llm_response:
        console.print(
            Panel(
                result.llm_response,
                title="IrpAI Response",
                style="green",
            )
        )
    elif query and result.enhanced_prompt:
        console.print(
            Panel(
                result.enhanced_prompt,
                title="Enhanced Prompt (LLM not called)",
                style="yellow",
            )
        )


if __name__ == "__main__":
    app()
