import os
import json
import shutil
import random
from pathlib import Path
import fiftyone as fo
import fiftyone.utils.huggingface as fouh


def main():
    print("=== 1. Preparing ScreenSpot Suite ===")
    os.system(
        "python examples/training/prepare_mobile_mixed_coco.py --output-root data/screenspot --total-samples -1 --val-ratio 0.15"
    )

    output_dir = Path("data/merged_coco")
    train_img_dir = output_dir / "train" / "images"
    val_img_dir = output_dir / "val" / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    merged_train = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "icon"}, {"id": 2, "name": "text"}],
    }
    merged_val = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "icon"}, {"id": 2, "name": "text"}],
    }

    img_idx = 1
    ann_idx = 1

    def add_coco_dataset(img_src_dir, ann_path, val_ratio=0.15, dataset_prefix="dataset"):
        nonlocal img_idx, ann_idx
        with open(ann_path, "r") as f:
            data = json.load(f)

        cat_map = {}
        for cat in data["categories"]:
            name = cat["name"].lower()
            if "text" in name or "label" in name or "heading" in name:
                cat_map[cat["id"]] = 2
            else:
                cat_map[cat["id"]] = 1

        img_anns = {}
        for ann in data["annotations"]:
            img_anns.setdefault(ann["image_id"], []).append(ann)

        images = data["images"]
        # Seeded shuffle for reproducibility
        random.Random(42).shuffle(images)

        val_count = int(len(images) * val_ratio)
        val_images = set(img["id"] for img in images[:val_count])

        for img in images:
            old_id = img["id"]
            filename = img["file_name"]
            src_path = Path(img_src_dir) / filename
            if not src_path.exists():
                continue

            is_val = old_id in val_images
            dst_dir = val_img_dir if is_val else train_img_dir
            merged_data = merged_val if is_val else merged_train

            new_filename = f"{dataset_prefix}_{filename}"
            shutil.copy(src_path, dst_dir / new_filename)

            merged_data["images"].append(
                {
                    "id": img_idx,
                    "file_name": new_filename,
                    "width": img["width"],
                    "height": img["height"],
                }
            )

            anns = img_anns.get(old_id, [])
            for ann in anns:
                new_cat = cat_map.get(ann["category_id"], 1)
                merged_data["annotations"].append(
                    {
                        "id": ann_idx,
                        "image_id": img_idx,
                        "category_id": new_cat,
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )
                ann_idx += 1

            img_idx += 1

    print("=== 2. Merging ScreenSpot Dataset ===")
    add_coco_dataset(
        "data/screenspot/train/images",
        "data/screenspot/train/annotations.json",
        val_ratio=0.0,
        dataset_prefix="screenspot",
    )
    add_coco_dataset(
        "data/screenspot/val/images",
        "data/screenspot/val/annotations.json",
        val_ratio=1.0,
        dataset_prefix="screenspot",
    )
    print(f"ScreenSpot merged. Total images: {img_idx - 1}")

    print("=== 3. Merging GroundUI-18k Dataset ===")
    try:
        groundui = fouh.load_from_hub("Voxel51/GroundUI-18k")
        groundui.export(
            export_dir="data/groundui_coco",
            dataset_type=fo.types.COCODetectionDataset,
            label_field="detections",
        )
        add_coco_dataset(
            "data/groundui_coco/data",
            "data/groundui_coco/labels.json",
            val_ratio=0.15,
            dataset_prefix="groundui",
        )
        print(f"GroundUI-18k merged. Total images: {img_idx - 1}")
    except Exception as e:
        print(f"Error importing GroundUI-18k: {e}")

    print("=== 4. Merging Rico Dataset (10,000 Subset) ===")
    try:
        rico = fouh.load_from_hub("Voxel51/rico")
        rico_subset = rico.limit(10000)
        rico_subset.export(
            export_dir="data/rico_coco",
            dataset_type=fo.types.COCODetectionDataset,
            label_field="detections",
        )
        add_coco_dataset(
            "data/rico_coco/data",
            "data/rico_coco/labels.json",
            val_ratio=0.15,
            dataset_prefix="rico",
        )
        print(f"Rico subset merged. Total images: {img_idx - 1}")
    except Exception as e:
        print(f"Error importing Rico: {e}")

    with open(output_dir / "train" / "annotations.json", "w") as f:
        json.dump(merged_train, f)
    with open(output_dir / "val" / "annotations.json", "w") as f:
        json.dump(merged_val, f)

    print(
        f"=== Merge Finished! Train: {len(merged_train['images'])} | Val: {len(merged_val['images'])} ==="
    )


if __name__ == "__main__":
    main()
