"""
IrpAI Stage 1: Preprocessing Module

Normalizes input screenshots for optimal downstream performance:
- Orientation detection & correction
- Resolution normalization
- Denoising (bilateral filter)
- Contrast enhancement (CLAHE)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from irpai.config import PreprocessingConfig

logger = logging.getLogger(__name__)


class OrientationDetector:
    """
    Detect and correct image orientation.

    Uses heuristic-based approach for screenshots:
    - Screenshots are typically landscape or portrait
    - Text line detection to infer orientation
    """

    @staticmethod
    def detect_orientation(img: np.ndarray) -> int:
        """
        Detect image orientation in degrees.

        Args:
            img: Input image (BGR)

        Returns:
            Rotation angle: 0, 90, 180, or 270 degrees
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use edge detection to find dominant line directions
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

        if lines is None or len(lines) < 5:
            return 0

        # Calculate dominant angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        angles = np.array(angles)

        # Count lines near 0°, 90°, 180°, 270°
        horizontal = np.sum(np.abs(angles) < 15) + np.sum(np.abs(angles - 180) < 15)
        vertical = np.sum(np.abs(angles - 90) < 15) + np.sum(np.abs(angles + 90) < 15)

        # If more vertical lines than horizontal, image might be rotated 90°
        h, w = img.shape[:2]
        if vertical > horizontal * 2 and w > h:
            return 90
        elif vertical > horizontal * 2 and h > w:
            return 270

        # Check for upside-down using text density (text usually at top)
        top_third = gray[:h // 3, :]
        bottom_third = gray[2 * h // 3:, :]

        top_edges = cv2.Canny(top_third, 50, 150)
        bottom_edges = cv2.Canny(bottom_third, 50, 150)

        top_density = np.sum(top_edges > 0) / top_edges.size
        bottom_density = np.sum(bottom_edges > 0) / bottom_edges.size

        # If significantly more content at bottom, might be upside-down
        if bottom_density > top_density * 2.0:
            return 180

        return 0

    @staticmethod
    def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by given angle (0, 90, 180, 270)."""
        if angle == 0:
            return img
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            logger.warning(f"Unsupported rotation angle: {angle}. Skipping rotation.")
            return img


class Preprocessor:
    """
    Screenshot preprocessing pipeline.

    Normalizes images for optimal downstream processing through:
    1. Orientation correction
    2. Resolution normalization
    3. Bilateral filter denoising
    4. CLAHE contrast enhancement
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.orientation_detector = OrientationDetector()
        logger.info("Preprocessor initialized with config: %s", self.config.model_dump())

    def process(self, image: np.ndarray | str | Path) -> np.ndarray:
        """
        Run full preprocessing pipeline on screenshot.

        Args:
            image: Input image as numpy array (BGR), file path, or Path object

        Returns:
            Preprocessed image (H, W, 3) uint8 array

        Raises:
            FileNotFoundError: If image path doesn't exist
            ValueError: If image cannot be loaded
        """
        start_time = time.perf_counter()

        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = self._load_image(str(image))
        else:
            img = image.copy()

        original_shape = img.shape[:2]
        logger.debug(f"Input image shape: {original_shape}")

        # Stage 1: Orientation correction
        if self.config.enable_orientation_correction:
            img = self._correct_orientation(img)

        # Stage 2: Resolution normalization
        img = self._normalize_resolution(img)

        # Stage 3: Denoising
        if self.config.enable_denoising:
            img = self._denoise(img)

        # Stage 4: Contrast enhancement
        if self.config.enable_contrast_enhancement:
            img = self._enhance_contrast(img)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Preprocessing complete: {original_shape} → {img.shape[:2]} in {elapsed_ms:.1f}ms"
        )

        return img

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return img

    def _correct_orientation(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct image orientation."""
        angle = self.orientation_detector.detect_orientation(img)
        if angle != 0:
            logger.info(f"Detected orientation: {angle}°, applying correction")
            img = self.orientation_detector.rotate_image(img, angle)
        return img

    def _normalize_resolution(self, img: np.ndarray) -> np.ndarray:
        """Resize image to standard dimensions while preserving aspect ratio."""
        h, w = img.shape[:2]
        max_dim = self.config.max_dimension

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized: ({w}, {h}) → ({new_w}, {new_h})")

        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to reduce noise while preserving edges."""
        return cv2.bilateralFilter(
            img,
            self.config.bilateral_d,
            self.config.bilateral_sigma_color,
            self.config.bilateral_sigma_space,
        )

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement on the L channel of LAB color space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def compute_quality_metrics(img: np.ndarray) -> dict:
        """
        Compute image quality metrics for diagnostics.

        Returns:
            Dict with brightness, contrast, sharpness metrics
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return {
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray)),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "resolution": f"{img.shape[1]}x{img.shape[0]}",
            "is_low_resolution": max(img.shape[:2]) < 800,
        }
