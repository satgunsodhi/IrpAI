"""
Image utility functions.

Shared helpers for loading, converting, and cropping images used
across every pipeline layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


# -- Type aliases -----------------------------------------------------------

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)


# -- Loading ----------------------------------------------------------------

def load_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """
    Normalise any common image source into a PIL ``Image`` in RGB mode.

    Accepts file paths (str / Path), PIL Images, or NumPy arrays (BGR or RGB).
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, np.ndarray):
        # OpenCV uses BGR — convert to RGB before wrapping
        if source.ndim == 3 and source.shape[2] == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        return Image.fromarray(source).convert("RGB")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


# -- Conversion -------------------------------------------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR ``ndarray``."""
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ``ndarray`` to a PIL RGB ``Image``."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# -- Cropping ---------------------------------------------------------------

def crop_region(
    img: Image.Image | np.ndarray,
    bbox: BBox,
    *,
    padding: int = 0,
) -> Image.Image:
    """
    Crop a rectangular region from an image.

    Parameters
    ----------
    img:
        Source image (PIL or NumPy).
    bbox:
        ``(x1, y1, x2, y2)`` coordinates of the region.
    padding:
        Extra pixels to include around the box (clamped to image edges).
    """
    if isinstance(img, np.ndarray):
        img = cv2_to_pil(img) if img.ndim == 3 else Image.fromarray(img)

    w, h = img.size
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)
    return img.crop((x1, y1, x2, y2))


def compute_iou(a: BBox, b: BBox) -> float:
    """Compute Intersection-over-Union between two bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def boxes_overlap(a: BBox, b: BBox, threshold: float = 0.1) -> bool:
    """Return True if two boxes overlap beyond the given IoU threshold."""
    return compute_iou(a, b) > threshold


def merge_boxes(boxes: Sequence[BBox]) -> BBox:
    """Return the smallest bounding box enclosing all given boxes."""
    if not boxes:
        raise ValueError("Cannot merge an empty list of boxes")
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def box_area(bbox: BBox) -> int:
    """Return the area of a bounding box in pixels."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def box_center(bbox: BBox) -> tuple[float, float]:
    """Return the centre point ``(cx, cy)`` of a bounding box."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
