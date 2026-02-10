"""
IrpAI — Basic Usage Example

Demonstrates the simplest way to use IrpAI: take a screenshot,
run the pipeline, and get structured context for an LLM.

Requirements:
    pip install irpai[ocr-tesseract]   # Lite tier (CPU only)
    pip install irpai[all]             # Full install
"""

from pathlib import Path

from irpai import IrpAIPipeline, IrpAIConfig, Tier


def main() -> None:
    # ── 1. Configure ─────────────────────────────────────────────────
    # Choose a tier based on your hardware:
    #   Tier.LITE     → CPU only, Tesseract OCR, no ML models
    #   Tier.STANDARD → CPU, PaddleOCR + YOLO detection
    #   Tier.FULL     → GPU, PaddleOCR + YOLO + Florence-2 VLM

    config = IrpAIConfig(tier=Tier.STANDARD)
    pipeline = IrpAIPipeline(config)

    # ── 2. Run on a screenshot ───────────────────────────────────────
    screenshot_path = Path("screenshot.png")

    if not screenshot_path.exists():
        print("No screenshot.png found — creating a demo image for testing.")
        _create_demo_image(screenshot_path)

    pack = pipeline.run(screenshot_path)

    # ── 3. Use the output ────────────────────────────────────────────
    # As JSON (best for programmatic use):
    print("═══ JSON Output ═══")
    print(pack.to_json(compact=False))
    print()

    # As Markdown (good for pasting into chat):
    print("═══ Markdown Output ═══")
    print(pack.to_markdown())
    print()

    # Key metrics:
    print("═══ Metrics ═══")
    print(f"  Regions:  {len(pack.regions)}")
    print(f"  Tokens:   ~{pack.token_estimate()}")
    print(f"  App type: {pack.summary.app_type}")
    print(f"  Time:     {pack.meta.total_ms:.0f} ms")
    print()

    # ── 4. Inject into an LLM prompt ────────────────────────────────
    from irpai.context.formatter import format_for_llm

    context = format_for_llm(pack, style="system_prompt")
    print("═══ System Prompt Injection ═══")
    print(context[:500])
    print("...")


def _create_demo_image(path: Path) -> None:
    """Create a realistic IDE-like demo screenshot with proper fonts."""
    from PIL import Image, ImageDraw, ImageFont
    import platform

    W, H = 1200, 900
    img = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # --- Find a usable TrueType font ---
    def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            # macOS
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            # Windows
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for fp in candidates:
            try:
                return ImageFont.truetype(fp, size)
            except (OSError, IOError):
                continue
        # Last resort: Pillow default
        try:
            return ImageFont.truetype("DejaVuSansMono.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()

    font_title = _load_font(20)
    font_sidebar = _load_font(18)
    font_code = _load_font(16)
    font_status = _load_font(15)

    TITLE_H = 48
    SIDEBAR_W = 240
    STATUS_H = 32
    EDITOR_L = SIDEBAR_W
    EDITOR_T = TITLE_H
    EDITOR_B = H - STATUS_H

    # ── Title bar ─────────────────────────────────────────────────────
    draw.rectangle([0, 0, W, TITLE_H], fill=(50, 50, 50))
    draw.text((16, 12), "My Application — main.py", fill=(220, 220, 220), font=font_title)
    # Window buttons
    for i, colour in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([W - 90 + i * 28, 14, W - 70 + i * 28, 34], fill=colour)

    # ── Sidebar ───────────────────────────────────────────────────────
    draw.rectangle([0, TITLE_H, SIDEBAR_W, EDITOR_B], fill=(37, 37, 38))
    sidebar_items = ["Explorer", "Search", "Source Control", "Extensions", "Settings"]
    for i, item in enumerate(sidebar_items):
        y = TITLE_H + 16 + i * 36
        draw.text((18, y), item, fill=(190, 190, 190), font=font_sidebar)

    # ── Editor area ───────────────────────────────────────────────────
    draw.rectangle([EDITOR_L, EDITOR_T, W, EDITOR_B], fill=(30, 30, 30))
    code_lines = [
        ("import irpai", (86, 156, 214)),
        ("", (212, 212, 212)),
        ("def main():", (86, 156, 214)),
        ("    config = irpai.IrpAIConfig()", (212, 212, 212)),
        ("    pipeline = irpai.IrpAIPipeline(config)", (212, 212, 212)),
        ('    result = pipeline.run("screenshot.png")', (206, 145, 120)),
        ("    print(result.to_json())", (212, 212, 212)),
        ("", (212, 212, 212)),
        ('if __name__ == "__main__":', (86, 156, 214)),
        ("    main()", (212, 212, 212)),
    ]
    for i, (line, colour) in enumerate(code_lines):
        y = EDITOR_T + 16 + i * 28
        draw.text((EDITOR_L + 24, y), line, fill=colour, font=font_code)

    # ── Status bar ────────────────────────────────────────────────────
    draw.rectangle([0, EDITOR_B, W, H], fill=(0, 122, 204))
    draw.text(
        (14, EDITOR_B + 7),
        "Python 3.12   UTF-8   LF   Ln 1, Col 1",
        fill=(255, 255, 255),
        font=font_status,
    )

    # ── Save with 150 DPI so Tesseract sees large-enough glyphs ──────
    img.save(path, dpi=(150, 150))
    print(f"Demo image saved to {path}  ({W}×{H} @150 DPI)")


if __name__ == "__main__":
    main()
