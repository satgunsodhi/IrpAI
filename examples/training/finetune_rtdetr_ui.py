"""Fine-tune RT-DETR on COCO-format UI element annotations.

Example:
    python examples/training/finetune_rtdetr_ui.py \
    --train-images data/screenspot_mobile_mixed_coco/train/images \
    --train-annotations data/screenspot_mobile_mixed_coco/train/annotations.json \
    --val-images data/screenspot_mobile_mixed_coco/val/images \
    --val-annotations data/screenspot_mobile_mixed_coco/val/annotations.json \
    --output-dir checkpoints/rtdetr-ui-mobile-mixed-complete
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


@dataclass(frozen=True)
class TrainPaths:
    train_images: Path
    train_annotations: Path
    val_images: Path
    val_annotations: Path


class CocoUIDataset(Dataset):
    """COCO detection dataset adapted for RT-DETR image processor."""

    def __init__(
        self,
        image_dir: Path,
        annotations_file: Path,
        processor: Any,
    ) -> None:
        self._base = CocoDetection(root=str(image_dir), annFile=str(annotations_file))
        self.processor = processor

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, object]:
        image, annotations = self._base[idx]

        if not isinstance(image, Image.Image):
            raise TypeError("CocoDetection must return PIL.Image")

        image_id = self._base.ids[idx]
        encoding = self.processor(
            images=image,
            annotations={"image_id": image_id, "annotations": annotations},
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    pixel_values: list[torch.Tensor] = [sample["pixel_values"] for sample in batch]  # type: ignore[list-item]
    labels: list[dict[str, object]] = [sample["labels"] for sample in batch]  # type: ignore[list-item]

    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on UI data")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="PekingU/rtdetr_r18vd_coco_o365")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-total-limit", type=int, default=6)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def _validate_paths(paths: TrainPaths) -> None:
    for p in [
        paths.train_images,
        paths.train_annotations,
        paths.val_images,
        paths.val_annotations,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")


def _supports_kwarg(training_args_cls: Any, kwarg: str) -> bool:
    return kwarg in inspect.signature(training_args_cls.__init__).parameters


def _build_training_args_kwargs(args: argparse.Namespace, training_args_cls: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "dataloader_num_workers": args.num_workers,
        "remove_unused_columns": False,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "fp16": bool(args.fp16),
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }

    if _supports_kwarg(training_args_cls, "eval_strategy"):
        kwargs["eval_strategy"] = "steps"
    elif _supports_kwarg(training_args_cls, "evaluation_strategy"):
        kwargs["evaluation_strategy"] = "steps"

    if _supports_kwarg(training_args_cls, "save_total_limit"):
        kwargs["save_total_limit"] = args.save_total_limit

    if _supports_kwarg(training_args_cls, "seed"):
        kwargs["seed"] = args.seed

    if _supports_kwarg(training_args_cls, "use_cpu"):
        kwargs["use_cpu"] = bool(args.use_cpu)

    return kwargs


def main() -> None:
    transformers = importlib.import_module("transformers")
    RTDetrForObjectDetection = transformers.RTDetrForObjectDetection
    RTDetrImageProcessor = transformers.RTDetrImageProcessor
    Trainer = transformers.Trainer
    TrainingArguments = transformers.TrainingArguments

    args = parse_args()
    paths = TrainPaths(
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
    )
    _validate_paths(paths)

    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint to resume: {args.resume_from_checkpoint}")

    processor = RTDetrImageProcessor.from_pretrained(args.model_id)
    model = RTDetrForObjectDetection.from_pretrained(args.model_id)

    train_dataset = CocoUIDataset(paths.train_images, paths.train_annotations, processor)
    val_dataset = CocoUIDataset(paths.val_images, paths.val_annotations, processor)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        **_build_training_args_kwargs(args=args, training_args_cls=TrainingArguments),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    train_kwargs: dict[str, Any] = {}
    if args.resume_from_checkpoint is not None:
        train_kwargs["resume_from_checkpoint"] = str(args.resume_from_checkpoint)

    trainer.train(**train_kwargs)
    trainer.save_model(str(args.output_dir / "checkpoint-best"))
    processor.save_pretrained(str(args.output_dir / "checkpoint-best"))


if __name__ == "__main__":
    main()
