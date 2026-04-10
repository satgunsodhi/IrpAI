"""
Synthetic Scene Generator
=========================

Creates realistic synthetic screenshots with **known ground truth** for
benchmarking every layer of the IrpAI pipeline.  Each scene returns:

- A PIL ``Image`` at 150 DPI
- A ``GroundTruth`` dataclass with expected regions, roles, text, bounding
  boxes, reading order, and app_type.

Scene catalogue
---------------
- IDE (1200×900)
- Browser (1000×800)
- Mobile (390×844)
- Chat (800×600)
- Dashboard (1440×900)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Ground-truth types
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthRegion:
    """Expected region in a synthetic screenshot."""
    role: str
    text: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    children: list[GroundTruthRegion] = field(default_factory=list)


@dataclass
class GroundTruth:
    """Complete ground truth for one synthetic screenshot."""
    scene_name: str
    app_type: str
    image_size: tuple[int, int]  # (w, h)
    regions: list[GroundTruthRegion] = field(default_factory=list)
    reading_order: list[int] = field(default_factory=list)
    all_text: str = ""
    pii_texts: list[str] = field(default_factory=list)  # injected PII strings
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticScene:
    """A synthetic screenshot and its ground truth."""
    image: Image.Image
    ground_truth: GroundTruth


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort font loader — falls back gracefully."""
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------

def make_ide_scene() -> SyntheticScene:
    """1200×900 IDE-like screenshot (title bar, sidebar, editor, status bar)."""
    W, H = 1200, 900
    img = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    font_title = _load_font(20)
    font_sidebar = _load_font(16)
    font_code = _load_font(14)
    font_status = _load_font(14)

    TITLE_H = 48
    SIDEBAR_W = 240
    STATUS_H = 32

    regions: list[GroundTruthRegion] = []

    # Title bar
    draw.rectangle([0, 0, W, TITLE_H], fill=(50, 50, 50))
    title_text = "My Application — main.py"
    draw.text((16, 12), title_text, fill=(220, 220, 220), font=font_title)
    regions.append(GroundTruthRegion(role="heading", text=title_text, bbox=(0, 0, W, TITLE_H)))

    # Sidebar
    draw.rectangle([0, TITLE_H, SIDEBAR_W, H - STATUS_H], fill=(37, 37, 38))
    sidebar_items = ["Explorer", "Search", "Source Control", "Extensions", "Settings"]
    sidebar_text = "\n".join(sidebar_items)
    for i, item in enumerate(sidebar_items):
        draw.text((18, TITLE_H + 16 + i * 32), item, fill=(190, 190, 190), font=font_sidebar)
    regions.append(GroundTruthRegion(
        role="nav", text=sidebar_text,
        bbox=(0, TITLE_H, SIDEBAR_W, H - STATUS_H),
    ))

    # Editor
    code_lines = [
        "import irpai",
        "",
        "def main():",
        "    config = irpai.IrpAIConfig()",
        "    pipeline = irpai.IrpAIPipeline(config)",
        '    result = pipeline.run("screenshot.png")',
        "    print(result.to_json())",
        "",
        'if __name__ == "__main__":',
        "    main()",
    ]
    code_text = "\n".join(code_lines)
    code_colors = [
        (86, 156, 214), (212, 212, 212), (86, 156, 214),
        (212, 212, 212), (212, 212, 212), (206, 145, 120),
        (212, 212, 212), (212, 212, 212), (86, 156, 214),
        (212, 212, 212),
    ]
    for i, (line, colour) in enumerate(zip(code_lines, code_colors)):
        draw.text((SIDEBAR_W + 24, TITLE_H + 16 + i * 24), line, fill=colour, font=font_code)
    regions.append(GroundTruthRegion(
        role="body", text=code_text,
        bbox=(SIDEBAR_W, TITLE_H, W, H - STATUS_H),
    ))

    # Status bar
    draw.rectangle([0, H - STATUS_H, W, H], fill=(0, 122, 204))
    status_text = "Python 3.12   UTF-8   LF   Ln 1, Col 1"
    draw.text((14, H - STATUS_H + 8), status_text, fill=(255, 255, 255), font=font_status)
    regions.append(GroundTruthRegion(
        role="status_bar", text=status_text,
        bbox=(0, H - STATUS_H, W, H),
    ))

    all_text = "\n".join(r.text for r in regions)
    gt = GroundTruth(
        scene_name="ide",
        app_type="ide",
        image_size=(W, H),
        regions=regions,
        reading_order=[0, 1, 2, 3],
        all_text=all_text,
    )
    img.info["dpi"] = (150, 150)
    return SyntheticScene(image=img, ground_truth=gt)


def make_browser_scene() -> SyntheticScene:
    """1000×800 browser-like screenshot (nav, search, content cards)."""
    W, H = 1000, 800
    img = Image.new("RGB", (W, H), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    font_nav = _load_font(16)
    font_heading = _load_font(22)
    font_body = _load_font(14)

    NAV_H = 52
    regions: list[GroundTruthRegion] = []

    # Nav bar
    draw.rectangle([0, 0, W, NAV_H], fill=(33, 37, 41))
    nav_text = "Home    About    Products    Contact"
    draw.text((20, 16), nav_text, fill=(255, 255, 255), font=font_nav)
    regions.append(GroundTruthRegion(role="nav", text=nav_text, bbox=(0, 0, W, NAV_H)))

    # Search bar
    draw.rectangle([200, NAV_H + 20, 800, NAV_H + 56], fill=(255, 255, 255), outline=(200, 200, 200))
    search_text = "Search products..."
    draw.text((212, NAV_H + 28), search_text, fill=(160, 160, 160), font=font_body)
    regions.append(GroundTruthRegion(role="input", text=search_text, bbox=(200, NAV_H + 20, 800, NAV_H + 56)))

    # Heading
    heading_y = NAV_H + 80
    heading_text = "Featured Products"
    draw.text((40, heading_y), heading_text, fill=(33, 37, 41), font=font_heading)
    regions.append(GroundTruthRegion(role="heading", text=heading_text, bbox=(40, heading_y, 400, heading_y + 30)))

    # Content cards
    cards = [
        ("Premium Widget", "$49.99 — High quality widget for professionals"),
        ("Smart Gadget", "$29.99 — Next-gen smart home device"),
        ("Ultra Tool", "$19.99 — Essential tool for everyday use"),
    ]
    card_y = heading_y + 50
    for i, (title, desc) in enumerate(cards):
        cx = 40 + i * 310
        draw.rectangle([cx, card_y, cx + 290, card_y + 180], fill=(255, 255, 255), outline=(220, 220, 220))
        draw.text((cx + 16, card_y + 16), title, fill=(33, 37, 41), font=font_nav)
        draw.text((cx + 16, card_y + 48), desc, fill=(100, 100, 100), font=font_body)
        # Buy button
        draw.rectangle([cx + 16, card_y + 130, cx + 120, card_y + 160], fill=(0, 123, 255))
        draw.text((cx + 30, card_y + 135), "Buy Now", fill=(255, 255, 255), font=font_body)
        regions.append(GroundTruthRegion(
            role="card", text=f"{title}\n{desc}\nBuy Now",
            bbox=(cx, card_y, cx + 290, card_y + 180),
        ))

    # Footer
    footer_y = H - 60
    draw.rectangle([0, footer_y, W, H], fill=(33, 37, 41))
    footer_text = "© 2026 Example Corp. All rights reserved."
    draw.text((20, footer_y + 20), footer_text, fill=(180, 180, 180), font=font_body)
    regions.append(GroundTruthRegion(role="footer", text=footer_text, bbox=(0, footer_y, W, H)))

    all_text = "\n".join(r.text for r in regions)
    gt = GroundTruth(
        scene_name="browser",
        app_type="browser",
        image_size=(W, H),
        regions=regions,
        reading_order=list(range(len(regions))),
        all_text=all_text,
    )
    img.info["dpi"] = (150, 150)
    return SyntheticScene(image=img, ground_truth=gt)


def make_mobile_scene() -> SyntheticScene:
    """390×844 mobile app screenshot (header, list, bottom nav)."""
    W, H = 390, 844
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_header = _load_font(18)
    font_item = _load_font(14)
    font_small = _load_font(12)

    HEADER_H = 56
    BOTTOM_NAV_H = 50
    regions: list[GroundTruthRegion] = []

    # Status bar + header
    draw.rectangle([0, 0, W, HEADER_H], fill=(98, 0, 238))
    header_text = "My Tasks"
    draw.text((16, 18), header_text, fill=(255, 255, 255), font=font_header)
    regions.append(GroundTruthRegion(role="heading", text=header_text, bbox=(0, 0, W, HEADER_H)))

    # List items
    items = [
        ("Review pull requests", "Due today — 3 pending"),
        ("Update documentation", "Due tomorrow — 1 page"),
        ("Fix login bug", "Due Fri — Critical priority"),
        ("Deploy v2.1.0", "Due next week — Staging ready"),
        ("Write unit tests", "Due next week — 5 modules"),
    ]
    item_y = HEADER_H + 12
    for i, (title, subtitle) in enumerate(items):
        y = item_y + i * 72
        draw.rectangle([0, y, W, y + 70], fill=(255, 255, 255), outline=(240, 240, 240))
        draw.text((20, y + 12), title, fill=(33, 33, 33), font=font_item)
        draw.text((20, y + 36), subtitle, fill=(130, 130, 130), font=font_small)
        regions.append(GroundTruthRegion(
            role="label", text=f"{title}\n{subtitle}",
            bbox=(0, y, W, y + 70),
        ))

    # Bottom nav
    draw.rectangle([0, H - BOTTOM_NAV_H, W, H], fill=(255, 255, 255))
    draw.line([0, H - BOTTOM_NAV_H, W, H - BOTTOM_NAV_H], fill=(220, 220, 220))
    nav_items = ["Home", "Tasks", "Profile"]
    nav_text = "  ".join(nav_items)
    for i, item in enumerate(nav_items):
        x = 30 + i * 130
        draw.text((x, H - 35), item, fill=(98, 0, 238), font=font_item)
    regions.append(GroundTruthRegion(
        role="nav", text=nav_text,
        bbox=(0, H - BOTTOM_NAV_H, W, H),
    ))

    all_text = "\n".join(r.text for r in regions)
    gt = GroundTruth(
        scene_name="mobile",
        app_type="mobile_app",
        image_size=(W, H),
        regions=regions,
        reading_order=list(range(len(regions))),
        all_text=all_text,
    )
    img.info["dpi"] = (150, 150)
    return SyntheticScene(image=img, ground_truth=gt)


def make_chat_scene() -> SyntheticScene:
    """800×600 chat interface (header, message bubbles, input)."""
    W, H = 800, 600
    img = Image.new("RGB", (W, H), color=(18, 18, 18))
    draw = ImageDraw.Draw(img)

    font_header = _load_font(18)
    font_msg = _load_font(14)
    font_input = _load_font(14)

    HEADER_H = 50
    INPUT_H = 50
    regions: list[GroundTruthRegion] = []

    # Header
    draw.rectangle([0, 0, W, HEADER_H], fill=(30, 30, 30))
    header_text = "Team Chat — #general"
    draw.text((20, 14), header_text, fill=(255, 255, 255), font=font_header)
    regions.append(GroundTruthRegion(role="heading", text=header_text, bbox=(0, 0, W, HEADER_H)))

    # Messages
    messages = [
        ("Alice", "Hey team, the deployment is ready for review!", True),
        ("Bob", "Looks good — I'll check the staging server now.", False),
        ("Alice", "Make sure to test the login flow and dashboard.", True),
        ("Charlie", "I found a minor bug in the settings page, creating a ticket.", False),
        ("Alice", "Thanks Charlie. Let's aim to merge by EOD.", True),
    ]
    msg_y = HEADER_H + 16
    for sender, text, is_left in messages:
        full_text = f"{sender}: {text}"
        if is_left:
            bx1, bx2 = 20, 480
        else:
            bx1, bx2 = 300, 780
        draw.rectangle([bx1, msg_y, bx2, msg_y + 44], fill=(45, 45, 48) if is_left else (0, 92, 175))
        draw.text((bx1 + 12, msg_y + 4), sender, fill=(160, 160, 255), font=font_msg)
        draw.text((bx1 + 12, msg_y + 22), text[:60], fill=(220, 220, 220), font=font_msg)
        regions.append(GroundTruthRegion(
            role="body", text=full_text,
            bbox=(bx1, msg_y, bx2, msg_y + 44),
        ))
        msg_y += 56

    # Input
    draw.rectangle([0, H - INPUT_H, W, H], fill=(30, 30, 30))
    draw.rectangle([16, H - INPUT_H + 8, W - 80, H - 8], fill=(45, 45, 48))
    input_text = "Type a message..."
    draw.text((28, H - INPUT_H + 16), input_text, fill=(120, 120, 120), font=font_input)
    draw.rectangle([W - 70, H - INPUT_H + 8, W - 16, H - 8], fill=(0, 123, 255))
    draw.text((W - 58, H - INPUT_H + 16), "Send", fill=(255, 255, 255), font=font_input)
    regions.append(GroundTruthRegion(role="input", text=input_text, bbox=(16, H - INPUT_H + 8, W - 80, H - 8)))
    regions.append(GroundTruthRegion(role="button", text="Send", bbox=(W - 70, H - INPUT_H + 8, W - 16, H - 8)))

    all_text = "\n".join(r.text for r in regions)
    gt = GroundTruth(
        scene_name="chat",
        app_type="chat",
        image_size=(W, H),
        regions=regions,
        reading_order=list(range(len(regions))),
        all_text=all_text,
    )
    img.info["dpi"] = (150, 150)
    return SyntheticScene(image=img, ground_truth=gt)


def make_dashboard_scene() -> SyntheticScene:
    """1440×900 analytics dashboard (sidebar nav, KPI cards, table)."""
    W, H = 1440, 900
    img = Image.new("RGB", (W, H), color=(240, 242, 245))
    draw = ImageDraw.Draw(img)

    font_heading = _load_font(22)
    font_nav = _load_font(16)
    font_kpi_value = _load_font(28)
    font_kpi_label = _load_font(13)
    font_table = _load_font(13)

    SIDEBAR_W = 220
    regions: list[GroundTruthRegion] = []

    # Sidebar
    draw.rectangle([0, 0, SIDEBAR_W, H], fill=(30, 41, 59))
    nav_items = ["Dashboard", "Analytics", "Users", "Reports", "Settings"]
    nav_text = "\n".join(nav_items)
    for i, item in enumerate(nav_items):
        draw.text((20, 30 + i * 40), item, fill=(180, 195, 220), font=font_nav)
    regions.append(GroundTruthRegion(role="nav", text=nav_text, bbox=(0, 0, SIDEBAR_W, H)))

    # Heading
    heading_text = "Analytics Overview"
    draw.text((SIDEBAR_W + 40, 30), heading_text, fill=(30, 41, 59), font=font_heading)
    regions.append(GroundTruthRegion(
        role="heading", text=heading_text,
        bbox=(SIDEBAR_W + 40, 30, SIDEBAR_W + 400, 60),
    ))

    # KPI cards
    kpis = [
        ("Total Users", "12,847", "+14.2%"),
        ("Revenue", "$284,930", "+8.7%"),
        ("Conversion", "3.24%", "+0.8%"),
        ("Avg Session", "4m 32s", "-2.1%"),
    ]
    card_y = 80
    kpi_texts = []
    for i, (label, value, change) in enumerate(kpis):
        cx = SIDEBAR_W + 40 + i * 295
        draw.rectangle([cx, card_y, cx + 275, card_y + 120], fill=(255, 255, 255))
        draw.text((cx + 20, card_y + 16), label, fill=(100, 110, 130), font=font_kpi_label)
        draw.text((cx + 20, card_y + 42), value, fill=(30, 41, 59), font=font_kpi_value)
        color = (34, 197, 94) if change.startswith("+") else (239, 68, 68)
        draw.text((cx + 20, card_y + 85), change, fill=color, font=font_kpi_label)
        kpi_texts.append(f"{label}: {value} ({change})")
    regions.append(GroundTruthRegion(
        role="card", text="\n".join(kpi_texts),
        bbox=(SIDEBAR_W + 40, card_y, W - 40, card_y + 120),
    ))

    # Data table
    table_y = card_y + 150
    draw.rectangle([SIDEBAR_W + 40, table_y, W - 40, table_y + 30], fill=(240, 242, 245))
    headers = ["Name", "Email", "Plan", "Revenue", "Status"]
    header_text = "  ".join(headers)
    for j, h in enumerate(headers):
        draw.text((SIDEBAR_W + 60 + j * 230, table_y + 6), h, fill=(80, 90, 110), font=font_table)
    regions.append(GroundTruthRegion(
        role="heading", text=header_text,
        bbox=(SIDEBAR_W + 40, table_y, W - 40, table_y + 30),
    ))

    # Table rows — include PII for the privacy benchmark
    rows = [
        ("John Smith", "john@example.com", "Enterprise", "$12,400", "Active"),
        ("Jane Doe", "jane.doe@corp.io", "Pro", "$8,200", "Active"),
        ("Bob Wilson", "bob@startup.dev", "Starter", "$1,900", "Churned"),
        ("Alice Chen", "alice@bigco.org", "Enterprise", "$15,800", "Active"),
    ]
    pii_texts = [row[1] for row in rows]
    row_texts = []
    for i, row in enumerate(rows):
        ry = table_y + 34 + i * 32
        bg = (255, 255, 255) if i % 2 == 0 else (248, 249, 250)
        draw.rectangle([SIDEBAR_W + 40, ry, W - 40, ry + 30], fill=bg)
        for j, cell in enumerate(row):
            draw.text((SIDEBAR_W + 60 + j * 230, ry + 6), cell, fill=(50, 50, 50), font=font_table)
        row_texts.append("  ".join(row))
    regions.append(GroundTruthRegion(
        role="body", text="\n".join(row_texts),
        bbox=(SIDEBAR_W + 40, table_y + 34, W - 40, table_y + 34 + len(rows) * 32),
    ))

    all_text = "\n".join(r.text for r in regions)
    gt = GroundTruth(
        scene_name="dashboard",
        app_type="settings",  # closest match from assembler patterns (Settings keyword)
        image_size=(W, H),
        regions=regions,
        reading_order=list(range(len(regions))),
        all_text=all_text,
        pii_texts=pii_texts,
    )
    img.info["dpi"] = (150, 150)
    return SyntheticScene(image=img, ground_truth=gt)


# ---------------------------------------------------------------------------
# Scene catalogue
# ---------------------------------------------------------------------------

ALL_SCENE_BUILDERS: dict[str, Any] = {
    "ide": make_ide_scene,
    "browser": make_browser_scene,
    "mobile": make_mobile_scene,
    "chat": make_chat_scene,
    "dashboard": make_dashboard_scene,
}


def build_all_scenes() -> list[SyntheticScene]:
    """Build every synthetic scene and return as a list."""
    return [builder() for builder in ALL_SCENE_BUILDERS.values()]
