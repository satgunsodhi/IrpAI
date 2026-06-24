"""
Image Normalizer

Handles all preprocessing steps before the image enters the detection
and OCR layers:
    1. Orientation detection & correction (Hough-line based)
    2. Resolution scaling (cap at target_max_dimension)
    3. Optional denoising (non-local means)
    4. Optional sharpening (unsharp mask)
    5. Optional adaptive thresholding (for low-contrast screens)
    6. Grayscale conversion for OCR pathway
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageFilter

from irpai.config import PreprocessingConfig
from irpai.utils.image import pil_to_cv2, cv2_to_pil, load_image
from irpai.utils.logging import get_logger

log = get_logger("preprocessing")


# ---------------------------------------------------------------------------
# Orientation detection (ported from irpai.old)
# ---------------------------------------------------------------------------


class OrientationDetector:
    """
    Detect and correct image orientation using Hough-line analysis.

    Uses edge detection to find dominant line directions and infers
    whether the image is rotated 0°, 90°, 180°, or 270°.
    """

    @staticmethod
    def detect_orientation(img: np.ndarray) -> int:
        """
        Detect image orientation in degrees.

        Parameters
        ----------
        img : np.ndarray
            Input image (BGR or RGB, uint8).

        Returns
        -------
        int
            Rotation angle: 0, 90, 180, or 270 degrees.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=50,
            maxLineGap=10,
        )

        if lines is None or len(lines) < 5:
            return 0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        angles_arr = np.array(angles)

        horizontal = int(np.sum(np.abs(angles_arr) < 15)) + int(
            np.sum(np.abs(angles_arr - 180) < 15)
        )
        vertical = int(np.sum(np.abs(angles_arr - 90) < 15)) + int(
            np.sum(np.abs(angles_arr + 90) < 15)
        )

        h, w = img.shape[:2]
        if vertical > horizontal * 2 and w > h:
            return 90
        elif vertical > horizontal * 2 and h > w:
            return 270

        # Check for upside-down: text content usually at the top
        top_third = gray[: h // 3, :]
        bottom_third = gray[2 * h // 3 :, :]

        top_edges = cv2.Canny(top_third, 50, 150)
        bottom_edges = cv2.Canny(bottom_third, 50, 150)

        top_density = np.sum(top_edges > 0) / max(top_edges.size, 1)
        bottom_density = np.sum(bottom_edges > 0) / max(bottom_edges.size, 1)

        if bottom_density > top_density * 2.0:
            return 180

        return 0

    @staticmethod
    def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by given angle (0, 90, 180, 270)."""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class Normalizer:
    """
    Stateless image normalizer driven by :class:`PreprocessingConfig`.

    Produces a :class:`NormalizedImage` with:
    - Full-colour image (scaled, denoised, sharpened)
    - Grayscale OCR-ready image (optionally thresholded)
    - Image quality metrics (brightness, contrast, sharpness)
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.cfg = config or PreprocessingConfig()
        self._orientation_detector = OrientationDetector()

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

        # 0. Orientation correction
        img = self._correct_orientation(img)

        # 1. Scale
        img = self._scale(img)

        # 2. Denoise (skip for clean screenshots)
        if self.cfg.denoise_strength > 0 and self._is_noisy(img):
            img = self._denoise(img)

        # 3. Sharpen
        if self.cfg.sharpen:
            img = self._sharpen(img)

        # 4. Prepare OCR variant (grayscale + optional threshold)
        ocr_img = self._prepare_ocr_image(img)

        # 5. Quality metrics
        quality = self.compute_quality_metrics(img)

        scale_factor = img.size[0] / original_size[0]

        log.info(
            "Preprocessed %s → %s (scale=%.2f, sharpness=%.0f)",
            original_size,
            img.size,
            scale_factor,
            quality["sharpness"],
        )

        return NormalizedImage(
            color=img,
            ocr_ready=ocr_img,
            original_size=original_size,
            processed_size=img.size,
            scale_factor=scale_factor,
            quality_metrics=quality,
        )

    # -- quality metrics (ported from irpai.old) ----------------------------

    @staticmethod
    def compute_quality_metrics(img: Image.Image | np.ndarray) -> dict:
        """
        Compute image quality metrics for diagnostics.

        Returns
        -------
        dict
            Keys: ``brightness``, ``contrast``, ``sharpness``,
            ``resolution``, ``is_low_resolution``.
        """
        if isinstance(img, Image.Image):
            arr = pil_to_cv2(img)
        else:
            arr = img

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr

        return {
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray)),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "resolution": f"{arr.shape[1]}x{arr.shape[0]}",
            "is_low_resolution": max(arr.shape[:2]) < 800,
        }

    # -- private steps ------------------------------------------------------

    def _correct_orientation(self, img: Image.Image) -> Image.Image:
        """Auto-rotate if the image appears to be in the wrong orientation."""
        arr = pil_to_cv2(img)
        angle = self._orientation_detector.detect_orientation(arr)
        if angle != 0:
            log.info("Detected orientation: %d°, applying correction", angle)
            arr = self._orientation_detector.rotate_image(arr, angle)
            return cv2_to_pil(arr)
        return img

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
        """Estimate if an image is noisy."""
        arr = np.array(img.convert("L"))
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

        # Ensure DPI metadata is present
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
    quality_metrics:
        Dict with brightness, contrast, sharpness, resolution info.
    """

    __slots__ = (
        "color",
        "ocr_ready",
        "original_size",
        "processed_size",
        "scale_factor",
        "quality_metrics",
    )

    def __init__(
        self,
        color: Image.Image,
        ocr_ready: Image.Image,
        original_size: tuple[int, int],
        processed_size: tuple[int, int],
        scale_factor: float,
        quality_metrics: dict | None = None,
    ) -> None:
        self.color = color
        self.ocr_ready = ocr_ready
        self.original_size = original_size
        self.processed_size = processed_size
        self.scale_factor = scale_factor
        self.quality_metrics = quality_metrics or {}

    def __repr__(self) -> str:
        return f"NormalizedImage(size={self.processed_size}, scale={self.scale_factor:.2f})"
