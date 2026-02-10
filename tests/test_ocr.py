"""
Tests for OCR base types and engine factory.
"""

from irpai.ocr.base import OCRBlock, OCRLine, OCRResult, OCRWord


class TestOCRTypes:
    def test_ocr_word(self) -> None:
        word = OCRWord(text="hello", bbox=(10, 20, 50, 35), confidence=0.95)
        assert word.text == "hello"
        assert word.confidence == 0.95

    def test_ocr_line(self) -> None:
        words = [
            OCRWord(text="hello", bbox=(10, 20, 50, 35), confidence=0.95),
            OCRWord(text="world", bbox=(55, 20, 100, 35), confidence=0.9),
        ]
        line = OCRLine(
            text="hello world",
            bbox=(10, 20, 100, 35),
            words=words,
            confidence=0.92,
        )
        assert line.text == "hello world"
        assert line.height == 15

    def test_ocr_block(self) -> None:
        line1 = OCRLine(text="Title", bbox=(10, 10, 200, 30), confidence=0.95)
        line2 = OCRLine(text="Subtitle", bbox=(10, 35, 200, 50), confidence=0.9)
        block = OCRBlock(
            lines=[line1, line2],
            bbox=(10, 10, 200, 50),
            confidence=0.92,
        )
        assert block.text == "Title\nSubtitle"
        assert block.estimated_font_size == 20.0  # median line height

    def test_ocr_result(self) -> None:
        line = OCRLine(text="test", bbox=(0, 0, 100, 20), confidence=0.88)
        block = OCRBlock(lines=[line], bbox=(0, 0, 100, 20), confidence=0.88)
        result = OCRResult(blocks=[block], image_size=(800, 600), engine="test")

        assert result.full_text == "test"
        assert len(result.all_lines) == 1
        assert abs(result.mean_confidence - 0.88) < 0.01

    def test_empty_result(self) -> None:
        result = OCRResult()
        assert result.full_text == ""
        assert result.all_lines == []
        assert result.mean_confidence == 0.0
