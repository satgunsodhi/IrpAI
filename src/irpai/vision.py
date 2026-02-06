"""
IrpAI Stage 2: Vision Branch - UI Element Detection

Detects and localizes UI components in screenshots using:
- Faster R-CNN with FPN (primary, when detectron2 available)
- TorchVision Faster R-CNN (fallback)
- Heuristic-based detection (lightweight fallback for CPU-only)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from irpai.config import VisionConfig
from irpai.models import BoundingBox, UIDetection, UIElementType

logger = logging.getLogger(__name__)


# Mapping from COCO classes to UI element types (for transfer learning)
COCO_TO_UI_MAPPING = {
    1: UIElementType.CONTAINER,   # person → container (placeholder)
    73: UIElementType.TEXT_BLOCK,  # book → text_block
    74: UIElementType.IMAGE,      # clock → image
    77: UIElementType.BUTTON,     # cell phone → button (placeholder)
    84: UIElementType.TEXT_BLOCK,  # book → text_block
}


class HeuristicUIDetector:
    """
    Lightweight, heuristic-based UI element detector.

    Uses contour analysis and edge detection to identify UI elements
    without requiring a trained neural network. Suitable as a fallback
    or for CPU-constrained environments.
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()

    def detect(self, img: np.ndarray) -> list[UIDetection]:
        """
        Detect UI elements using contour-based heuristics.

        Args:
            img: Preprocessed screenshot (BGR)

        Returns:
            List of UIDetection objects
        """
        start_time = time.perf_counter()
        detections = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Step 1: Detect rectangular regions (buttons, inputs, containers)
        rect_detections = self._detect_rectangles(gray, h, w)
        detections.extend(rect_detections)

        # Step 2: Detect text-heavy regions
        text_detections = self._detect_text_regions(gray, h, w)
        detections.extend(text_detections)

        # Step 3: Non-maximum suppression to remove overlaps
        detections = self._nms(detections, self.config.nms_threshold)

        # Mark text-containing regions
        for det in detections:
            det.text_region = det.element_type in [
                UIElementType.BUTTON,
                UIElementType.TEXT_INPUT,
                UIElementType.TEXT_BLOCK,
                UIElementType.MENU,
                UIElementType.TAB,
                UIElementType.LINK,
            ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Heuristic detection: {len(detections)} elements in {elapsed_ms:.1f}ms")

        return detections[:self.config.max_detections]

    def _detect_rectangles(self, gray: np.ndarray, h: int, w: int) -> list[UIDetection]:
        """Detect rectangular UI elements using edge detection and contour analysis."""
        detections = []

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size relative to image
            min_area = (h * w) * 0.0005  # Min 0.05% of image
            max_area = (h * w) * 0.8     # Max 80% of image

            if area < min_area or area > max_area:
                continue

            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Rectangles have 4 vertices
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                aspect_ratio = bw / max(bh, 1)

                # Classify based on size and aspect ratio
                elem_type = self._classify_rectangle(bw, bh, aspect_ratio, w, h)

                detections.append(
                    UIDetection(
                        bbox=BoundingBox(x1=x, y1=y, x2=x + bw, y2=y + bh),
                        element_type=elem_type,
                        confidence=0.6,
                        text_region=True,
                    )
                )

        return detections

    def _classify_rectangle(
        self, bw: int, bh: int, aspect_ratio: float, img_w: int, img_h: int
    ) -> UIElementType:
        """Classify a rectangular region based on size and aspect ratio."""
        rel_w = bw / img_w
        rel_h = bh / img_h

        # Button: small-medium, wider than tall
        if 0.03 < rel_w < 0.25 and 0.02 < rel_h < 0.08 and 1.5 < aspect_ratio < 8:
            return UIElementType.BUTTON

        # Text input: wider than tall, medium size
        if 0.1 < rel_w < 0.6 and 0.02 < rel_h < 0.06 and aspect_ratio > 4:
            return UIElementType.TEXT_INPUT

        # Large horizontal bar (navbar/menu)
        if rel_w > 0.7 and rel_h < 0.1:
            return UIElementType.MENU

        # Modal: centered, medium-large
        if 0.2 < rel_w < 0.7 and 0.2 < rel_h < 0.7:
            return UIElementType.MODAL

        # Container: large area
        if rel_w > 0.3 or rel_h > 0.3:
            return UIElementType.CONTAINER

        # Small square: likely icon or checkbox
        if 0.8 < aspect_ratio < 1.2 and rel_w < 0.05:
            return UIElementType.ICON

        return UIElementType.CONTAINER

    def _detect_text_regions(self, gray: np.ndarray, h: int, w: int) -> list[UIDetection]:
        """Detect text-heavy regions using connected components analysis."""
        detections = []

        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological operations to connect text components
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        
        # Dilate to connect nearby text
        dilated = cv2.dilate(binary, kernel_horizontal, iterations=1)
        dilated = cv2.dilate(dilated, kernel_vertical, iterations=1)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )

        # Extract bounding boxes from components
        bboxes = []
        min_area = 100
        max_area = int(h * w * 0.3)

        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, bw, bh, area = stats[i]
            
            if area < min_area or area > max_area:
                continue
            
            # Filter by aspect ratio (text regions are typically wider than tall)
            aspect_ratio = bw / max(bh, 1)
            if aspect_ratio < 0.2 or aspect_ratio > 50:
                continue
            
            bboxes.append([x, y, x + bw, y + bh])

        if not bboxes:
            return detections

        # Cluster nearby boxes into text blocks
        text_blocks = self._cluster_text_regions(bboxes, h, w)

        for block in text_blocks:
            x1, y1, x2, y2 = block
            detections.append(
                UIDetection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    element_type=UIElementType.TEXT_BLOCK,
                    confidence=0.5,
                    text_region=True,
                )
            )

        return detections

    def _cluster_text_regions(
        self, bboxes: list[list[int]], h: int, w: int
    ) -> list[list[int]]:
        """Cluster nearby text bounding boxes into text block regions."""
        if not bboxes:
            return []

        bboxes_arr = np.array(bboxes)

        # Simple grid-based clustering
        merge_dist_x = w * 0.05  # 5% of image width
        merge_dist_y = h * 0.02  # 2% of image height

        clusters = []
        used = set()

        for i in range(len(bboxes_arr)):
            if i in used:
                continue

            cluster = [bboxes_arr[i]]
            used.add(i)

            for j in range(i + 1, len(bboxes_arr)):
                if j in used:
                    continue

                # Check if boxes are close enough to merge
                bi = bboxes_arr[i]
                bj = bboxes_arr[j]

                dx = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
                dy = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))

                if dx < merge_dist_x and dy < merge_dist_y:
                    cluster.append(bj)
                    used.add(j)

            # Compute bounding box of cluster
            cluster = np.array(cluster)
            merged = [
                int(cluster[:, 0].min()),
                int(cluster[:, 1].min()),
                int(cluster[:, 2].max()),
                int(cluster[:, 3].max()),
            ]

            # Filter tiny clusters
            area = (merged[2] - merged[0]) * (merged[3] - merged[1])
            if area > (h * w) * 0.001:  # At least 0.1% of image
                clusters.append(merged)

        return clusters

    def _nms(self, detections: list[UIDetection], threshold: float) -> list[UIDetection]:
        """Non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        for i, det in enumerate(detections):
            should_keep = True
            for kept in keep:
                iou = det.bbox.iou(kept.bbox)
                if iou > threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)

        return keep


class NeuralUIDetector:
    """
    Neural network-based UI element detector using Faster R-CNN with FPN.

    Uses TorchVision's pretrained Faster R-CNN model, fine-tuned for
    UI element detection. Falls back to COCO-pretrained model if no
    custom weights are available.
    """

    def __init__(self, config: Optional[VisionConfig] = None, device: Optional[str] = None):
        self.config = config or VisionConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        logger.info(f"NeuralUIDetector initialized on {self.device}")

    def _load_model(self) -> nn.Module:
        """Load Faster R-CNN model with appropriate weights."""
        num_classes = len(self.config.detection_classes) + 1  # +1 for background

        if self.config.custom_model_path:
            # Load custom-trained model
            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
            checkpoint = torch.load(self.config.custom_model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded custom model from {self.config.custom_model_path}")
        else:
            # Use COCO-pretrained model (transfer learning base)
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            # Replace the classifier head for UI detection
            in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore[union-attr]
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            logger.info("Using COCO-pretrained Faster R-CNN (transfer learning base)")

        model.to(self.device)
        model.eval()

        # Optional: quantize for speed
        if self.config.use_quantized and self.device == "cpu":
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            logger.info("Model quantized to INT8")

        return model

    def detect(self, img: np.ndarray) -> list[UIDetection]:
        """
        Detect UI elements in screenshot using Faster R-CNN.

        Args:
            img: Preprocessed screenshot (BGR, uint8)

        Returns:
            List of UIDetection objects
        """
        start_time = time.perf_counter()

        # Convert BGR to RGB and to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor: torch.Tensor = self.transform(img_rgb).to(self.device)  # type: ignore[assignment]

        # Run inference
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]

        # Parse predictions
        detections = []
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < self.config.confidence_threshold:
                continue

            # Map label to UI element type
            elem_type = self._map_label_to_type(int(label))

            detections.append(
                UIDetection(
                    bbox=BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    ),
                    element_type=elem_type,
                    confidence=float(score),
                    text_region=elem_type
                    in [
                        UIElementType.BUTTON,
                        UIElementType.TEXT_INPUT,
                        UIElementType.TEXT_BLOCK,
                        UIElementType.MENU,
                        UIElementType.TAB,
                        UIElementType.LINK,
                    ],
                )
            )

        # Apply NMS
        detections = self._nms(detections)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Neural detection: {len(detections)} elements in {elapsed_ms:.1f}ms")

        return detections[:self.config.max_detections]

    def _map_label_to_type(self, label: int) -> UIElementType:
        """Map model output label to UIElementType."""
        if label < len(self.config.detection_classes):
            try:
                return UIElementType(self.config.detection_classes[label])
            except ValueError:
                return UIElementType.UNKNOWN
        return UIElementType.UNKNOWN

    def _nms(self, detections: list[UIDetection]) -> list[UIDetection]:
        """Non-maximum suppression."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        for det in detections:
            should_keep = True
            for kept in keep:
                if det.bbox.iou(kept.bbox) > self.config.nms_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)

        return keep


class UIDetector:
    """
    Unified UI element detector.

    Selects the best available detection strategy:
    1. Neural (Faster R-CNN) - if GPU or custom model available
    2. Heuristic - lightweight fallback
    """

    def __init__(self, config: Optional[VisionConfig] = None, device: Optional[str] = None):
        self.config = config or VisionConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Try neural detector, fall back to heuristic
        self._use_neural = False
        self._neural_detector: Optional[NeuralUIDetector] = None
        self._heuristic_detector = HeuristicUIDetector(self.config)

        if self.config.custom_model_path or self.device == "cuda":
            try:
                self._neural_detector = NeuralUIDetector(self.config, self.device)
                self._use_neural = True
                logger.info("Using neural UI detector")
            except Exception as e:
                logger.warning(f"Neural detector init failed: {e}. Using heuristic fallback.")
        else:
            logger.info("Using heuristic UI detector (no custom model or GPU)")

    def detect(self, img: np.ndarray) -> list[UIDetection]:
        """
        Detect UI elements using best available method.

        Args:
            img: Preprocessed screenshot (BGR)

        Returns:
            List of UIDetection objects
        """
        if self._use_neural and self._neural_detector is not None:
            try:
                return self._neural_detector.detect(img)
            except Exception as e:
                logger.warning(f"Neural detection failed: {e}. Falling back to heuristic.")

        return self._heuristic_detector.detect(img)
