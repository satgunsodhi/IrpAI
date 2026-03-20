"""
Image Normalizer

Handles all preprocessing steps before the image enters the detection
and OCR layers:
    1. Resolution scaling (cap at target_max_dimension)
    2. Optional denoising (non-local means)
    3. Optional sharpening (unsharp mask)
    4. Optional adaptive thresholding (for low-contrast screens)
    5. Grayscale conversion for OCR pathway
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from irpai.config import PreprocessingConfig
from irpai.utils.image import pil_to_cv2, cv2_to_pil, load_image
from irpai.utils.logging import get_logger

log = get_logger("preprocessing")


class Normalizer:
    """Stateless image normalizer driven by :class:`PreprocessingConfig`."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.cfg = config or PreprocessingConfig()

    # -- public API ---------------------------------------------------------

    def normalize(
        self,
        image: str | Image.Image | np.ndarray,
    ) -> NormalizedImage:
        """
        Run the full preprocessing pipeline and return a
        :class:`NormalizedImage` bundle.
        """
        img = load_image(image)
        original_size = img.size  # (w, h)

        log.debug("Input size: %s", original_size)

        # 1. Scale
        img = self._scale(img)

        # 2. Denoise (operates in OpenCV space)
        #    Skip for synthetic / clean screenshots — denoising blurs
        #    sharp edges and hurts OCR accuracy.
        if self.cfg.denoise_strength > 0 and self._is_noisy(img):
            img = self._denoise(img)

        # 3. Sharpen
        if self.cfg.sharpen:
            img = self._sharpen(img)

        # 4. Prepare OCR variant (grayscale + optional threshold)
        ocr_img = self._prepare_ocr_image(img)

        scale_factor = img.size[0] / original_size[0]

        log.info(
            "Preprocessed %s → %s (scale=%.2f)",
            original_size,
            img.size,
            scale_factor,
        )

        return NormalizedImage(
            color=img,
            ocr_ready=ocr_img,
            original_size=original_size,
            processed_size=img.size,
            scale_factor=scale_factor,
        )

    # -- private steps ------------------------------------------------------

    def _scale(self, img: Image.Image) -> Image.Image:
        """Down-scale so the longest side ≤ target_max_dimension."""
        max_dim = self.cfg.target_max_dimension
        w, h = img.size
        longest = max(w, h)
        if longest <= max_dim:
            return img

        ratio = max_dim / longest
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _denoise(self, img: Image.Image) -> Image.Image:
        """Apply non-local-means denoising."""
        arr = pil_to_cv2(img)
        denoised = cv2.fastNlMeansDenoisingColored(
            arr,
            None,
            self.cfg.denoise_strength,
            self.cfg.denoise_strength,
            7,
            21,
        )
        return cv2_to_pil(denoised)

    @staticmethod
    def _is_noisy(img: Image.Image, threshold: float = 8.0) -> bool:
        """Estimate if an image is noisy (high Laplacian variance = edges, not noise).

        Clean screenshots (solid backgrounds, crisp text) have very *low*
        high-frequency noise variance.  We measure noise in smooth regions
        by checking variance of the Laplacian on a small center crop.
        """
        arr = np.array(img.convert("L"))
        # Sample center 25 % to avoid UI edges
        ch, cw = arr.shape
        y1, y2 = ch // 4, 3 * ch // 4
        x1, x2 = cw // 4, 3 * cw // 4
        crop = arr[y1:y2, x1:x2]
        lap = cv2.Laplacian(crop, cv2.CV_64F)
        variance = float(lap.var())
        log.debug("Noise variance: %.1f (threshold %.1f)", variance, threshold)
        return variance > threshold

    def _sharpen(self, img: Image.Image) -> Image.Image:
        """Mild unsharp-mask sharpening to recover edge detail after resize."""
        return img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))

    def _prepare_ocr_image(self, img: Image.Image) -> Image.Image:
        """
        Produce an OCR-optimised image:
        - Convert to grayscale (if configured)
        - Apply adaptive thresholding for low-contrast input (if configured)
        """
        ocr = img.copy()

        if self.cfg.grayscale_ocr:
            ocr = ocr.convert("L")

        if self.cfg.adaptive_threshold:
            arr = np.array(ocr)
            if arr.ndim == 2:
                thresh = cv2.adaptiveThreshold(
                    arr,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=15,
                    C=8,
                )
                ocr = Image.fromarray(thresh)

        # Ensure DPI metadata is present (Tesseract defaults to 72 DPI
        # without it, making text appear tiny and unreadable).
        if not ocr.info.get("dpi"):
            ocr.info["dpi"] = (150, 150)

        return ocr


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class NormalizedImage:
    """
    Bundle of preprocessed image variants produced by :class:`Normalizer`.

    Attributes
    ----------
    color:
        Full-colour RGB image after scaling, denoising, and sharpening.
    ocr_ready:
        Grayscale (and optionally thresholded) image optimised for OCR.
    original_size:
        ``(width, height)`` of the input image.
    processed_size:
        ``(width, height)`` after preprocessing.
    scale_factor:
        ``processed_width / original_width``.
    """

    __slots__ = ("color", "ocr_ready", "original_size", "processed_size", "scale_factor")

    def __init__(
        self,
        color: Image.Image,
        ocr_ready: Image.Image,
        original_size: tuple[int, int],
        processed_size: tuple[int, int],
        scale_factor: float,
    ) -> None:
        self.color = color
        self.ocr_ready = ocr_ready
        self.original_size = original_size
        self.processed_size = processed_size
        self.scale_factor = scale_factor

    def __repr__(self) -> str:
        return (
            f"NormalizedImage(size={self.processed_size}, "
            f"scale={self.scale_factor:.2f})"
        )
