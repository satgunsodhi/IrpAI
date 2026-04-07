"""RT-DETR based detector used as the primary ML backend for UI detection."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from irpai.detection.base import BaseDetector, DetectionOutput, DetectionResult
from irpai.utils.logging import get_logger

log = get_logger("detection.rt_detr")


class RTDETRDetector(BaseDetector):
    """
    Primary ML detector backed by HuggingFace RT-DETR.

    The detector lazily imports heavy dependencies (torch/transformers),
    so environments without ML extras can still run with rule fallback.
    """

    MODEL_ID = "PekingU/rtdetr_r18vd_coco_o365"

    def __init__(
        self,
        model_id: str | None = None,
        confidence_threshold: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id or self.MODEL_ID
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._processor = None
        self._model = None
        self._torch = None

    @property
    def name(self) -> str:
        short_id = self.model_id.split("/")[-1]
        return f"RTDETRDetector({short_id})"

    def warmup(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        except Exception as exc:  # pragma: no cover - dependency availability
            raise RuntimeError(
                "RTDETRDetector requires 'torch' and 'transformers'. "
                "Install with: pip install -e .[detection-ml]"
            ) from exc

        resolved_device = _resolve_device(self.device, torch)
        self._torch = torch

        loaded_from: str | None = None
        last_exc: Exception | None = None
        for candidate in _candidate_model_ids(self.model_id, self.MODEL_ID):
            if _looks_like_local_path(candidate) and not Path(candidate).exists():
                log.warning("RT-DETR local checkpoint missing: %s", candidate)
                continue

            try:
                self._processor = RTDetrImageProcessor.from_pretrained(candidate)
                self._model = RTDetrForObjectDetection.from_pretrained(candidate)
                loaded_from = candidate
                break
            except Exception as exc:  # pragma: no cover - model availability
                last_exc = exc
                log.warning("Failed to load RT-DETR model from %s", candidate, exc_info=True)

        if self._model is None or self._processor is None:
            raise RuntimeError(
                "Unable to load RT-DETR model from local or fallback sources. "
                f"Requested='{self.model_id}', fallback='{self.MODEL_ID}'."
            ) from last_exc

        assert loaded_from is not None
        if loaded_from != self.model_id:
            log.warning("Using RT-DETR fallback model: %s (requested: %s)", loaded_from, self.model_id)
            self.model_id = loaded_from

        self._model.to(resolved_device)
        self._model.eval()
        self.device = resolved_device

    def detect(self, image: Image.Image | np.ndarray) -> DetectionOutput:
        self._load_model()
        assert self._processor is not None and self._model is not None and self._torch is not None

        t0 = time.perf_counter()
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        w, h = pil.size

        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=[(h, w)],
            threshold=self.confidence_threshold,
        )[0]

        detections: list[DetectionResult] = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"],
        ):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            raw_label = self._model.config.id2label.get(int(label_id.item()), "object")
            ui_label = _map_coco_to_ui(raw_label)

            detections.append(
                DetectionResult(
                    label=ui_label,
                    bbox=(x1, y1, x2, y2),
                    confidence=round(float(score.item()), 3),
                    meta={"source": "rt_detr", "raw_label": raw_label},
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return DetectionOutput(
            detections=detections,
            image_size=(w, h),
            model_name=self.name,
            inference_ms=round(elapsed_ms, 1),
        )


def _resolve_device(device: str, torch: object) -> str:
    if device != "auto":
        return device

    if bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()):
        return "cuda"

    mps = getattr(torch, "backends", None)
    if mps and hasattr(mps, "mps") and bool(mps.mps.is_available()):
        return "mps"

    return "cpu"


def _candidate_model_ids(requested_model_id: str, fallback_model_id: str) -> list[str]:
    candidates = [requested_model_id]
    if requested_model_id != fallback_model_id:
        candidates.append(fallback_model_id)
    return candidates


def _looks_like_local_path(model_id: str) -> bool:
    if model_id.startswith("."):
        return True
    if model_id.startswith("/"):
        return True
    if "\\" in model_id:
        return True
    if ":" in model_id[:3]:
        return True
    # HF model ids are typically owner/repo (single slash).
    if model_id.count("/") > 1:
        return True
    return False


def _map_coco_to_ui(coco_label: str) -> str:
    """Map COCO labels to IrpAI UI labels for generic checkpoints."""
    label = coco_label.lower()
    mapping = {
        "remote": "button",
        "mouse": "button",
        "keyboard": "text_input",
        "tv": "content_area",
        "laptop": "content_area",
        "cell phone": "content_area",
        "book": "text_block",
        "clock": "icon",
        "traffic light": "icon",
        "stop sign": "icon",
        "person": "container",
        "chair": "container",
        "couch": "container",
        "table": "container",
        "dining table": "container",
        "potted plant": "container",
        "window": "container",
        "door": "container",
    }
    return mapping.get(label, "container")
