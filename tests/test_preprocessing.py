"""
Tests for the preprocessing layer.
"""

import numpy as np
from PIL import Image

from irpai.config import PreprocessingConfig
from irpai.preprocessing.normalizer import Normalizer, NormalizedImage


def _make_test_image(w: int = 800, h: int = 600) -> Image.Image:
    """Create a simple test image."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class TestNormalizer:
    def test_basic_normalize(self) -> None:
        img = _make_test_image(800, 600)
        normalizer = Normalizer()
        result = normalizer.normalize(img)

        assert isinstance(result, NormalizedImage)
        assert result.color.mode == "RGB"
        assert result.processed_size == (800, 600)
        assert result.scale_factor == 1.0

    def test_downscale_large_image(self) -> None:
        img = _make_test_image(3840, 2160)
        normalizer = Normalizer(PreprocessingConfig(target_max_dimension=1920))
        result = normalizer.normalize(img)

        assert max(result.processed_size) <= 1920
        assert result.scale_factor < 1.0

    def test_no_upscale_small_image(self) -> None:
        img = _make_test_image(640, 480)
        normalizer = Normalizer(PreprocessingConfig(target_max_dimension=1920))
        result = normalizer.normalize(img)

        assert result.processed_size == (640, 480)
        assert result.scale_factor == 1.0

    def test_grayscale_ocr_image(self) -> None:
        img = _make_test_image(400, 300)
        normalizer = Normalizer(PreprocessingConfig(grayscale_ocr=True))
        result = normalizer.normalize(img)

        assert result.ocr_ready.mode == "L"

    def test_adaptive_threshold(self) -> None:
        img = _make_test_image(400, 300)
        normalizer = Normalizer(
            PreprocessingConfig(
                grayscale_ocr=True,
                adaptive_threshold=True,
            )
        )
        result = normalizer.normalize(img)

        assert result.ocr_ready.mode == "L"
        arr = np.array(result.ocr_ready)
        # Adaptive threshold produces binary image
        unique = set(arr.flatten().tolist())
        assert unique.issubset({0, 255})

    def test_numpy_input(self) -> None:
        arr = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        normalizer = Normalizer()
        result = normalizer.normalize(arr)

        assert isinstance(result, NormalizedImage)
        assert result.color.mode == "RGB"

    def test_denoise_disabled(self) -> None:
        img = _make_test_image(400, 300)
        normalizer = Normalizer(PreprocessingConfig(denoise_strength=0))
        result = normalizer.normalize(img)

        assert isinstance(result, NormalizedImage)

    def test_sharpen_disabled(self) -> None:
        img = _make_test_image(400, 300)
        normalizer = Normalizer(PreprocessingConfig(sharpen=False))
        result = normalizer.normalize(img)

        assert isinstance(result, NormalizedImage)
