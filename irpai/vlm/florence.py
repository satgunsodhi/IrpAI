"""
Florence-2 Vision-Language Model integration.

Uses Microsoft's Florence-2-base (0.23B params, MIT license) for:
    - Image captioning (scene understanding)
    - Object detection with labels
    - OCR with region localisation
    - Dense captioning

This model runs comfortably on a 4-6 GB GPU or (more slowly) on CPU.

Install
-------
    pip install torch transformers einops timm
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from irpai.config import VLMConfig
from irpai.vlm.base import BaseVLM, VLMResult
from irpai.utils.logging import get_logger

log = get_logger("vlm.florence")


class FlorenceVLM(BaseVLM):
    """
    Florence-2 wrapper implementing the :class:`BaseVLM` interface.

    Lazily loads the model on first call to :meth:`run`.
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.cfg = config or VLMConfig()
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return f"Florence-2 ({self.cfg.model_name.split('/')[-1]})"

    # -- lazy init ----------------------------------------------------------

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor

        try:
            import torch  # type: ignore[import-untyped]
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForCausalLM,
                AutoProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "Florence-2 requires: pip install torch transformers einops timm"
            ) from exc

        log.info(
            "Loading Florence-2: %s (device=%s, dtype=%s)",
            self.cfg.model_name,
            self.cfg.device,
            self.cfg.dtype,
        )

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.cfg.dtype, torch.float16)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.cfg.device)

        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True,
        )

        log.info("Florence-2 loaded successfully")
        return self._model, self._processor

    def warmup(self) -> None:
        """Pre-load the model and run a dummy inference."""
        model, processor = self._ensure_model()
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        self._run_task(model, processor, dummy, "<CAPTION>")
        log.info("Florence-2 warmup complete")

    # -- inference ----------------------------------------------------------

    def run(
        self,
        image: Image.Image | np.ndarray,
        tasks: tuple[str, ...] | None = None,
    ) -> VLMResult:
        model, processor = self._ensure_model()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        task_list = tasks or self.cfg.tasks
        t0 = time.perf_counter()

        raw_outputs: dict[str, object] = {}
        caption = ""
        detailed_caption = ""
        objects: list[dict[str, object]] = []
        ocr_text: list[dict[str, object]] = []

        for task_prompt in task_list:
            try:
                result = self._run_task(model, processor, image, task_prompt)
                raw_outputs[task_prompt] = result

                if task_prompt == "<CAPTION>":
                    caption = result.get("<CAPTION>", "")
                elif task_prompt == "<DETAILED_CAPTION>":
                    detailed_caption = result.get("<DETAILED_CAPTION>", "")
                elif task_prompt == "<OD>":
                    od_data = result.get("<OD>", {})
                    if isinstance(od_data, dict):
                        labels = od_data.get("labels", [])
                        bboxes = od_data.get("bboxes", [])
                        for lbl, bbox in zip(labels, bboxes):
                            objects.append({"label": lbl, "bbox": bbox})
                elif task_prompt == "<OCR_WITH_REGION>":
                    ocr_data = result.get("<OCR_WITH_REGION>", {})
                    if isinstance(ocr_data, dict):
                        labels = ocr_data.get("labels", [])
                        bboxes = ocr_data.get("quad_boxes", [])
                        for lbl, bbox in zip(labels, bboxes):
                            ocr_text.append({"text": lbl, "bbox": bbox})

            except Exception:
                log.exception("Florence-2 task '%s' failed", task_prompt)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        log.info(
            "Florence-2: %d tasks in %.1f ms, caption=%d chars",
            len(task_list),
            elapsed_ms,
            len(caption),
        )

        return VLMResult(
            caption=caption,
            detailed_caption=detailed_caption,
            objects=objects,
            ocr_text=ocr_text,
            raw_outputs=raw_outputs,
            model_name=self.name,
            inference_ms=elapsed_ms,
        )

    def _run_task(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        task_prompt: str,
    ) -> dict[str, Any]:
        """Run a single Florence-2 task prompt and return parsed output."""
        import torch  # type: ignore[import-untyped]

        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=3,
            )

        # Decode and post-process
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image.size,
        )
        return result
