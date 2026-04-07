# RT-DETR UI Detector Training Guide

This guide explains how to train a UI-focused RT-DETR checkpoint and use it in IrpAI.

## 1. Goal

Train an RT-DETR detector with mobile screenshot coverage using a consolidated dataset pipeline.

Current prepared dataset classes in this repo:

- icon
- text

If you need additional UI classes, extend the mapping and source labels in:

- examples/training/prepare_mobile_mixed_coco.py

The runtime pipeline already supports this via `HybridDetector`:
- primary: `RTDETRDetector`
- fallback: `FallbackGridDetector`

## 2. Data Sources

The consolidated prep flow combines:

- ScreenSpot-Pro (professional desktop screenshots)
- ScreenSpot-v2 mobile split (mobile screenshots)

The prep script enforces a minimum mobile ratio and exports COCO format.

## 3. Directory Layout

Prepared dataset output:

```text
data/
  screenspot_mobile_mixed_coco/
    raw/
    train/
      images/
      annotations.json
    val/
      images/
      annotations.json
```

Each `annotations.json` must contain:
- `images`
- `annotations`
- `categories`

## 4. Install Training Dependencies

```bash
pip install -e ".[training]"
```

Recommended environment setup:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

pip install -e ".[training]"
```

## 5. Start Fine-Tuning

Recommended (single command, cross-platform):

```bash
python examples/training/run_mobile_training.py
```

To also export the best checkpoint as a zip suitable for GitHub Releases:

```bash
python examples/training/run_mobile_training.py --export-best-zip
```

Manual flow (advanced):

Step 1: prepare the mixed COCO dataset with mobile coverage.

```bash
python examples/training/prepare_mobile_mixed_coco.py \
  --output-root data/screenspot_mobile_mixed_coco \
  --total-samples 240 \
  --val-ratio 0.2 \
  --mobile-min-ratio 0.5 \
  --seed 42
```

Step 2: run RT-DETR fine-tuning.

```bash
python examples/training/finetune_rtdetr_ui.py \
  --train-images data/screenspot_mobile_mixed_coco/train/images \
  --train-annotations data/screenspot_mobile_mixed_coco/train/annotations.json \
  --val-images data/screenspot_mobile_mixed_coco/val/images \
  --val-annotations data/screenspot_mobile_mixed_coco/val/annotations.json \
  --output-dir checkpoints/rtdetr-ui-mobile-mixed-complete \
  --model-id PekingU/rtdetr_r18vd_coco_o365 \
  --epochs 10 \
  --batch-size 4 \
  --lr 3e-5
```

## 6. Validate and Export

After training, the best checkpoint folder should include:

- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- `preprocessor_config.json`

You can keep it local or upload it to HuggingFace.

Consolidated local best checkpoint:

```text
checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best
```

If `--export-best-zip` is used, this archive is created:

```text
artifacts/rtdetr-ui-mobile-mixed-complete-checkpoint-best.zip
```

## 7. Use Trained Model in IrpAI

Configure the detector model ID/path:

```python
from irpai.config import IrpAIConfig, DetectionConfig, Tier
from irpai.pipeline import IrpAIPipeline

config = IrpAIConfig(tier=Tier.STANDARD)
config.detection = DetectionConfig(
    use_ml=True,
    model_id="checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best",
    confidence_threshold=0.35,
    structural_boost=True,
    device="cpu",
)

pipeline = IrpAIPipeline(config)
```

## 8. Labeling Guidance

For the current two-class setup:

- icon: tight box around the clickable icon glyph
- text: tight box around visible target text only

## 9. Practical Tips

- Start from RT-DETR-R18 and train 10+ epochs.
- Keep mobile coverage >= 40% for mobile-oriented use-cases.
- Use resolution close to real screenshots (1080-1920 width).
- Track class-level precision/recall for icon vs text separately.
- Keep `FallbackGridDetector` enabled as a structural safety net.

## 10. Recommended Evaluation Metrics

Use COCO metrics and report:

- mAP@0.50:0.95
- mAP@0.50
- AP per UI class
- Inference latency on CPU

For pipeline quality, also track:

- OCR-to-region assignment accuracy
- Heading/body role consistency after layout merge
- End-to-end context usefulness on real screenshots

## 11. What To Commit

- Commit scripts and markdown docs.
- Do not commit generated `data/` folders.
- Do not commit generated `checkpoints/` folders.

The repository `.gitignore` is configured accordingly.

## 12. Runtime Fallback Behavior

At inference time, RT-DETR loading uses this order:

1. configured model id/path (local checkpoint or Hub id)
2. fallback Hub base model: `PekingU/rtdetr_r18vd_coco_o365`
3. if RT-DETR cannot load, `HybridDetector` falls back to `FallbackGridDetector`

This keeps the pipeline operational even if local weights are missing.

## 13. GitHub Upload Recommendation

- Prefer uploading the zip archive as a GitHub Release asset.
- Avoid committing checkpoint files directly to the repo.
- If you must version model weights in Git, use Git LFS.
