"""Prepare a mixed COCO dataset for RT-DETR fine-tuning.

Primary goal: guarantee mobile screenshots are present in train and val.

Sources:
- ScreenSpot-Pro (desktop/professional screenshots)
- ScreenSpot-v2 mobile split (mobile screenshots)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import zipfile
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image


PRO_DATASET_ID = "likaixin/ScreenSpot-Pro"
V2_DATASET_ID = "OS-Copilot/ScreenSpot-v2"
SUPPORTED_CLASSES = ("icon", "text")


@dataclass(frozen=True)
class Record:
    source: str
    image_rel: str
    ui_type: str
    bbox: tuple[float, float, float, float]
    img_size: tuple[int, int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mixed COCO subset with mobile coverage")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/screenspot_mobile_mixed_coco")
    )
    parser.add_argument("--total-samples", type=int, default=240)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--mobile-min-ratio",
        type=float,
        default=0.4,
        help="Minimum mobile fraction in sampled records (0.0 to 1.0).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_pro_records(raw_root: Path) -> list[Record]:
    ann_dir = raw_root / "annotations"
    files = sorted(ann_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation files found in {ann_dir}")

    records: list[Record] = []
    for ann_file in files:
        with ann_file.open("r", encoding="utf-8") as f:
            items: list[dict[str, Any]] = json.load(f)

        for item in items:
            ui_type = str(item.get("ui_type", "")).strip().lower()
            if ui_type not in SUPPORTED_CLASSES:
                continue

            image_rel = str(item.get("img_filename", "")).strip()
            bbox_raw = item.get("bbox", [])
            if len(bbox_raw) != 4 or not image_rel:
                continue

            try:
                b0 = float(bbox_raw[0])
                b1 = float(bbox_raw[1])
                b2 = float(bbox_raw[2])
                b3 = float(bbox_raw[3])
                bbox = (b0, b1, b2, b3)
            except (TypeError, ValueError):
                continue

            size = item.get("img_size")
            img_size: tuple[int, int] | None = None
            if isinstance(size, list) and len(size) == 2:
                try:
                    img_size = (int(size[0]), int(size[1]))
                except (TypeError, ValueError):
                    img_size = None

            records.append(
                Record(
                    source="pro",
                    image_rel=image_rel,
                    ui_type=ui_type,
                    bbox=bbox,
                    img_size=img_size,
                )
            )

    return records


def _load_v2_records(raw_root: Path, filename: str, source_name: str) -> list[Record]:
    ann_file = raw_root / filename
    if not ann_file.exists():
        raise FileNotFoundError(f"Missing ScreenSpot-v2 annotations: {ann_file}")

    with ann_file.open("r", encoding="utf-8") as f:
        items: list[dict[str, Any]] = json.load(f)

    records: list[Record] = []
    for item in items:
        ui_type = str(item.get("data_type", "")).strip().lower()
        if ui_type not in SUPPORTED_CLASSES:
            continue

        image_rel = str(item.get("img_filename", "")).strip()
        bbox_raw = item.get("bbox", [])
        if len(bbox_raw) != 4 or not image_rel:
            continue

        try:
            b0 = float(bbox_raw[0])
            b1 = float(bbox_raw[1])
            b2 = float(bbox_raw[2])
            b3 = float(bbox_raw[3])
            bbox = (b0, b1, b2, b3)
        except (TypeError, ValueError):
            continue

        records.append(
            Record(
                source=source_name,
                image_rel=image_rel,
                ui_type=ui_type,
                bbox=bbox,
                img_size=None,
            )
        )

    return records


def _sample_balanced(records: list[Record], target: int, seed: int) -> list[Record]:
    if target <= 0 or not records:
        return []
    if target >= len(records):
        return records.copy()

    rng = random.Random(seed)
    by_class: dict[str, list[Record]] = {cls: [] for cls in SUPPORTED_CLASSES}
    for rec in records:
        by_class[rec.ui_type].append(rec)

    for cls in SUPPORTED_CLASSES:
        rng.shuffle(by_class[cls])

    sampled: list[Record] = []
    class_counts = {cls: len(items) for cls, items in by_class.items()}
    total = sum(class_counts.values())

    for cls in SUPPORTED_CLASSES:
        take = max(1, round(target * class_counts[cls] / max(total, 1)))
        sampled.extend(by_class[cls][:take])

    if len(sampled) > target:
        rng.shuffle(sampled)
        sampled = sampled[:target]
    elif len(sampled) < target:
        remainder = [r for r in records if r not in sampled]
        rng.shuffle(remainder)
        sampled.extend(remainder[: target - len(sampled)])

    rng.shuffle(sampled)
    return sampled


def _sample_mixed_records(
    pro_records: list[Record],
    mobile_records: list[Record],
    total_samples: int,
    mobile_min_ratio: float,
    seed: int,
) -> list[Record]:
    if total_samples <= 0:
        return []

    mobile_min_ratio = max(0.0, min(mobile_min_ratio, 1.0))
    mobile_target = min(len(mobile_records), max(1, int(round(total_samples * mobile_min_ratio))))

    sampled_mobile = _sample_balanced(mobile_records, mobile_target, seed=seed)
    remaining_target = max(0, total_samples - len(sampled_mobile))
    sampled_pro = _sample_balanced(pro_records, remaining_target, seed=seed + 1)

    sampled = sampled_mobile + sampled_pro
    if len(sampled) < total_samples:
        rng = random.Random(seed + 2)
        combined = pro_records + mobile_records
        remainder = [r for r in combined if r not in sampled]
        rng.shuffle(remainder)
        sampled.extend(remainder[: total_samples - len(sampled)])

    rng = random.Random(seed + 3)
    rng.shuffle(sampled)
    return sampled[:total_samples]


def _split_records(
    records: list[Record], val_ratio: float, seed: int
) -> tuple[list[Record], list[Record]]:
    rng = random.Random(seed)
    by_bucket: dict[tuple[str, str], list[Record]] = {}
    for rec in records:
        key = (rec.source, rec.ui_type)
        by_bucket.setdefault(key, []).append(rec)

    train: list[Record] = []
    val: list[Record] = []

    for bucket_records in by_bucket.values():
        rng.shuffle(bucket_records)
        val_count = max(1, int(round(len(bucket_records) * val_ratio)))
        val.extend(bucket_records[:val_count])
        train.extend(bucket_records[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _bbox_to_xywh(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox

    # ScreenSpot-Pro stores [x1, y1, x2, y2]. If not, fall back to xywh.
    if x2 > x1 and y2 > y1:
        return x1, y1, x2 - x1, y2 - y1
    return x1, y1, x2, y2


def _xywh_from_record(rec: Record) -> tuple[float, float, float, float]:
    if rec.source in ("mobile_v2", "desktop_v2", "web_v2"):
        return rec.bbox
    return _bbox_to_xywh(rec.bbox)


def _build_mobile_image_index(extract_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in extract_root.rglob("*.png"):
        key = path.name
        if key not in index:
            index[key] = str(path)
    return index


def _ensure_mobile_v2_images(out_root: Path, needed_names: set[str]) -> dict[str, str]:
    if not needed_names:
        return {}

    raw_root = out_root / "raw" / "screenspot_v2"
    raw_root.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id=V2_DATASET_ID,
        repo_type="dataset",
        filename="screenspotv2_image.zip",
        local_dir=str(raw_root),
    )
    extract_root = raw_root / "images"
    extract_root.mkdir(parents=True, exist_ok=True)

    index = _build_mobile_image_index(extract_root)
    missing = {name for name in needed_names if name not in index}
    if missing:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            member_lookup: dict[str, str] = {}
            for member in members:
                key = Path(member).name
                if key in needed_names and key not in member_lookup:
                    member_lookup[key] = member

            for name in missing:
                member = member_lookup.get(name)
                if member is None:
                    continue
                zf.extract(member, path=extract_root)

        index = _build_mobile_image_index(extract_root)

    unresolved = [name for name in needed_names if name not in index]
    if unresolved:
        preview = unresolved[:8]
        print(
            f"warning: missing {len(unresolved)} mobile images in ScreenSpot-v2 archive; "
            f"skipping examples like: {preview}"
        )

    return index


def _clamp_bbox(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float] | None:
    x = max(0.0, min(x, float(width - 1)))
    y = max(0.0, min(y, float(height - 1)))
    w = max(1.0, min(w, float(width) - x))
    h = max(1.0, min(h, float(height) - y))
    if w <= 0.0 or h <= 0.0:
        return None
    return x, y, w, h


def _build_split(
    split_name: str,
    split_records: list[Record],
    out_root: Path,
    mobile_index: dict[str, str],
) -> None:
    images_dir = out_root / split_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "icon"},
            {"id": 2, "name": "text"},
        ],
    }

    image_id = 1
    annotation_id = 1
    skipped_mobile_missing = 0

    for rec in split_records:
        if rec.source in ("mobile_v2", "desktop_v2", "web_v2"):
            local_file = mobile_index.get(rec.image_rel)
            if local_file is None:
                skipped_mobile_missing += 1
                continue
        else:
            remote_file = f"images/{rec.image_rel}"
            local_file = hf_hub_download(
                repo_id=PRO_DATASET_ID,
                repo_type="dataset",
                filename=remote_file,
                local_dir=str(out_root / "raw" / "screenspot_pro"),
            )

        local_name = f"{rec.source}__{rec.image_rel.replace('/', '__')}"
        dst_file = images_dir / local_name
        if not dst_file.exists():
            with Image.open(local_file) as source_img:
                source_img.save(dst_file)

        with Image.open(dst_file) as img:
            width, height = img.size

        x, y, w, h = _xywh_from_record(rec)
        clamped = _clamp_bbox(x, y, w, h, width, height)
        if clamped is None:
            continue
        x, y, w, h = clamped

        coco["images"].append(
            {
                "id": image_id,
                "file_name": local_name,
                "width": width,
                "height": height,
            }
        )
        coco["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1 if rec.ui_type == "icon" else 2,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            }
        )

        image_id += 1
        annotation_id += 1

    ann_file = out_root / split_name / "annotations.json"
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    with ann_file.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    source_counts: dict[str, int] = {}
    for rec in split_records:
        source_counts[rec.source] = source_counts.get(rec.source, 0) + 1

    print(
        f"{split_name}: images={len(coco['images'])} annotations={len(coco['annotations'])} "
        f"sources={source_counts} skipped_mobile_missing={skipped_mobile_missing}"
    )


def main() -> None:
    args = parse_args()
    out_root = args.output_root

    snapshot_download(
        repo_id=PRO_DATASET_ID,
        repo_type="dataset",
        local_dir=str(out_root / "raw" / "screenspot_pro"),
        allow_patterns=["annotations/*.json", "README.md", "eval.yaml"],
    )

    # Download all v2 splits
    v2_splits = {
        "mobile_v2": "screenspot_mobile_v2.json",
        "desktop_v2": "screenspot_desktop_v2.json",
        "web_v2": "screenspot_web_v2.json",
    }
    for filename in v2_splits.values():
        hf_hub_download(
            repo_id=V2_DATASET_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(out_root / "raw" / "screenspot_v2"),
        )

    pro_records = _load_pro_records(out_root / "raw" / "screenspot_pro")

    v2_records = []
    for source_name, filename in v2_splits.items():
        v2_records.extend(
            _load_v2_records(out_root / "raw" / "screenspot_v2", filename, source_name)
        )

    if not pro_records and not v2_records:
        raise RuntimeError("No usable records found")

    if args.total_samples <= 0:
        sampled = pro_records + v2_records
        # Shuffle sampled records to ensure uniform distribution
        random.Random(args.seed).shuffle(sampled)
    else:
        # For mixed sampling, treat all v2_records as 'mobile' or partition accordingly.
        # Since we want mixed records, we can pass v2_records as the second list to _sample_mixed_records.
        sampled = _sample_mixed_records(
            pro_records=pro_records,
            mobile_records=v2_records,
            total_samples=args.total_samples,
            mobile_min_ratio=args.mobile_min_ratio,
            seed=args.seed,
        )

    train_records, val_records = _split_records(sampled, val_ratio=args.val_ratio, seed=args.seed)

    needed_v2 = {r.image_rel for r in sampled if r.source in v2_splits}
    mobile_index = _ensure_mobile_v2_images(out_root, needed_names=needed_v2)

    _build_split("train", train_records, out_root, mobile_index=mobile_index)
    _build_split("val", val_records, out_root, mobile_index=mobile_index)

    v2_count = sum(1 for r in sampled if r.source in v2_splits)
    print(
        f"done: total={len(sampled)} train={len(train_records)} val={len(val_records)} "
        f"v2={v2_count}"
    )


if __name__ == "__main__":
    main()
