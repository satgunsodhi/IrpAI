"""Run the full mobile-mixed training workflow in one command.

This script orchestrates:
1) Dataset preparation (mobile-inclusive COCO export)
2) RT-DETR fine-tuning
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_RTDETR_HF_MODEL_ID = "PekingU/rtdetr_r18vd_coco_o365"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mobile-mixed data prep + RT-DETR training")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--output-root", type=Path, default=Path("data/screenspot_mobile_mixed_coco"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/rtdetr-ui-mobile-mixed-complete"),
    )
    parser.add_argument("--total-samples", type=int, default=240)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--mobile-min-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_RTDETR_HF_MODEL_ID,
        help=(
            "Base model id/path for fine-tuning. "
            "Use a local checkpoint path if you want to continue from prior weights."
        ),
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--export-best-zip",
        action="store_true",
        help="Export checkpoint-best as a zip archive for sharing/upload.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for exported zip archives.",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def _export_best_checkpoint_zip(best_checkpoint_dir: Path, export_dir: Path, checkpoint_dir: Path) -> Path:
    if not best_checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {best_checkpoint_dir}")

    export_dir.mkdir(parents=True, exist_ok=True)
    archive_base = export_dir / f"{checkpoint_dir.name}-checkpoint-best"
    archive_path = shutil.make_archive(
        base_name=str(archive_base),
        format="zip",
        root_dir=str(best_checkpoint_dir.parent),
        base_dir=best_checkpoint_dir.name,
    )
    return Path(archive_path)


def main() -> None:
    args = parse_args()

    python_cmd = str(args.python)
    output_root = args.output_root
    checkpoint_dir = args.checkpoint_dir
    model_id = args.model_id

    if not args.skip_prepare:
        _run(
            [
                python_cmd,
                "examples/training/prepare_mobile_mixed_coco.py",
                "--output-root",
                str(output_root),
                "--total-samples",
                str(args.total_samples),
                "--val-ratio",
                str(args.val_ratio),
                "--mobile-min-ratio",
                str(args.mobile_min_ratio),
                "--seed",
                str(args.seed),
            ]
        )

    train_cmd = [
        python_cmd,
        "examples/training/finetune_rtdetr_ui.py",
        "--train-images",
        str(output_root / "train" / "images"),
        "--train-annotations",
        str(output_root / "train" / "annotations.json"),
        "--val-images",
        str(output_root / "val" / "images"),
        "--val-annotations",
        str(output_root / "val" / "annotations.json"),
        "--output-dir",
        str(checkpoint_dir),
        "--model-id",
        model_id,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--seed",
        str(args.seed),
    ]
    if args.resume_from_checkpoint is not None:
        train_cmd.extend(["--resume-from-checkpoint", str(args.resume_from_checkpoint)])

    _run(train_cmd)

    best_checkpoint_dir = checkpoint_dir / "checkpoint-best"
    if args.export_best_zip:
        archive_path = _export_best_checkpoint_zip(best_checkpoint_dir, args.export_dir, checkpoint_dir)
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"\nExported best checkpoint zip: {archive_path} ({size_mb:.1f} MB)")

    print(f"\nDone. Best checkpoint should be at: {checkpoint_dir / 'checkpoint-best'}")


if __name__ == "__main__":
    main()
