"""
Tests for the context assembly layer and output schema.
"""

import json

from irpai.context.assembler import ContextAssembler
from irpai.context.formatter import format_for_llm
from irpai.context.schema import ContextPack, PipelineMetadata, Region, ScreenSummary
from irpai.layout.analyzer import LayoutResult, SemanticRegion
from irpai.ocr.base import OCRBlock, OCRLine, OCRResult


def _make_layout() -> LayoutResult:
    regions = [
        SemanticRegion(role="heading", bbox=(10, 10, 300, 40), text="Dashboard"),
        SemanticRegion(role="body", bbox=(10, 50, 700, 400), text="Welcome to the app.\nHere is your data."),
        SemanticRegion(role="button", bbox=(600, 450, 700, 480), text="Save"),
    ]
    return LayoutResult(
        regions=regions,
        image_size=(800, 600),
        reading_order=[0, 1, 2],
    )


def _make_ocr() -> OCRResult:
    blocks = [
        OCRBlock(
            lines=[OCRLine(text="Dashboard", bbox=(10, 10, 300, 40), confidence=0.95)],
            bbox=(10, 10, 300, 40),
            confidence=0.95,
        ),
    ]
    return OCRResult(blocks=blocks, image_size=(800, 600), engine="test")


class TestContextPack:
    def test_to_json(self) -> None:
        pack = ContextPack(
            summary=ScreenSummary(description="A test screen", app_type="ide"),
            regions=[
                Region(role="heading", text="Title", bbox=(0, 0, 100, 30)),
                Region(role="body", text="Content", bbox=(0, 35, 100, 200)),
            ],
        )
        j = pack.to_json(compact=False)
        data = json.loads(j)

        assert "summary" in data
        assert "regions" in data
        assert len(data["regions"]) == 2
        assert data["summary"]["app_type"] == "ide"

    def test_to_json_compact(self) -> None:
        pack = ContextPack(
            regions=[Region(role="heading", text="Title")],
        )
        compact = pack.to_json(compact=True)
        pretty = pack.to_json(compact=False)

        assert len(compact) < len(pretty)

    def test_to_markdown(self) -> None:
        pack = ContextPack(
            summary=ScreenSummary(description="Test", app_type="browser"),
            regions=[
                Region(role="heading", text="Title"),
                Region(role="body", text="Body content"),
            ],
        )
        md = pack.to_markdown()

        assert "## Screenshot Context" in md
        assert "[HEADING]" in md
        assert "[BODY]" in md

    def test_token_estimate(self) -> None:
        pack = ContextPack(
            regions=[Region(role="body", text="x" * 400)],
        )
        estimate = pack.token_estimate()
        assert estimate > 0

    def test_exclude_bboxes(self) -> None:
        pack = ContextPack(
            regions=[Region(role="heading", text="Title", bbox=(0, 0, 100, 30))],
        )
        d = pack.to_dict(include_bboxes=False)
        assert "bbox" not in d["regions"][0]

    def test_nested_regions(self) -> None:
        pack = ContextPack(
            regions=[
                Region(
                    role="card",
                    text="",
                    bbox=(0, 0, 300, 200),
                    children=[
                        Region(role="heading", text="Card Title", bbox=(10, 10, 280, 40)),
                    ],
                ),
            ],
        )
        d = pack.to_dict()
        assert len(d["regions"][0]["children"]) == 1


class TestContextAssembler:
    def test_assemble_basic(self) -> None:
        layout = _make_layout()
        ocr = _make_ocr()
        assembler = ContextAssembler()
        pack = assembler.assemble(layout, ocr=ocr)

        assert len(pack.regions) == 3
        assert pack.summary.description  # should not be empty

    def test_app_type_detection(self) -> None:
        layout = LayoutResult(
            regions=[
                SemanticRegion(
                    role="body",
                    bbox=(0, 0, 800, 600),
                    text="Visual Studio Code\nimport numpy\ndef main():\n    pass",
                ),
            ],
            image_size=(800, 600),
            reading_order=[0],
        )
        assembler = ContextAssembler()
        pack = assembler.assemble(layout)

        assert pack.summary.app_type == "ide"

    def test_token_budget_enforcement(self) -> None:
        from irpai.config import OutputConfig

        long_text = "x " * 5000  # ~10,000 chars
        layout = LayoutResult(
            regions=[SemanticRegion(role="body", bbox=(0, 0, 800, 600), text=long_text)],
            image_size=(800, 600),
            reading_order=[0],
        )
        assembler = ContextAssembler(OutputConfig(max_context_tokens=500))
        pack = assembler.assemble(layout)

        # Should be truncated
        assert pack.token_estimate() < 10000  # much less than original


class TestFormatter:
    def test_json_format(self) -> None:
        pack = ContextPack(regions=[Region(role="body", text="test")])
        output = format_for_llm(pack, style="json")
        data = json.loads(output)
        assert "regions" in data

    def test_markdown_format(self) -> None:
        pack = ContextPack(regions=[Region(role="body", text="test")])
        output = format_for_llm(pack, style="markdown")
        assert "## Screenshot Context" in output

    def test_system_prompt_format(self) -> None:
        pack = ContextPack(regions=[Region(role="body", text="test")])
        output = format_for_llm(pack, style="system_prompt")
        assert "<screenshot_context>" in output
        assert "</screenshot_context>" in output

    def test_max_chars(self) -> None:
        pack = ContextPack(regions=[Region(role="body", text="x" * 1000)])
        output = format_for_llm(pack, style="json", max_chars=200)
        assert len(output) <= 200
