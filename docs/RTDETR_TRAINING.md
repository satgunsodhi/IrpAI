# RT-DETR UI Detector Training Guide

This guide covers how to train a UI-focused RT-DETR checkpoint and use it in IrpAI.

## 1. Scope

Current runtime detection integration:
- primary detector: RT-DETR (standard tier)
- fallback detector: rule-based structural detector

Current prepared dataset classes in this repo:
- `icon`
- `text`

If you need more UI classes, extend mapping logic in:
- `examples/training/prepare_mobile_mixed_coco.py`

## 2. Data Sources

The prep pipeline consolidates data from:
- ScreenSpot-Pro
- ScreenSpot-v2 mobile split

Prepared output is COCO format under:

```text
data/screenspot_mobile_mixed_coco/
  train/images/
  train/annotations.json
  val/images/
  val/annotations.json
```

## 3. Environment Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install training dependencies:

```bash
pip install -e ".[training]"
```

## 4. Recommended Training Command

```bash
python examples/training/run_mobile_training.py
```

To export best checkpoint as a zip artifact:

```bash
python examples/training/run_mobile_training.py --export-best-zip
```

## 5. Manual Training Flow (Advanced)

Step 1: prepare mixed COCO dataset.

```bash
python examples/training/prepare_mobile_mixed_coco.py \
  --output-root data/screenspot_mobile_mixed_coco \
  --total-samples 240 \
  --val-ratio 0.2 \
  --mobile-min-ratio 0.5 \
  --seed 42
```

Step 2: fine-tune RT-DETR.

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

## 6. Checkpoints and Runtime Loading

Expected best-checkpoint folder:

```text
checkpoints/rtdetr-ui-mobile-mixed-complete/checkpoint-best
```

Expected files include:
- `config.json`
- `preprocessor_config.json`
- `model.safetensors` or `pytorch_model.bin`

Runtime default behavior in `src/irpai/config.py`:
1. if local `checkpoint-best` exists, use it as detector `model_id`
2. otherwise use Hub model `PekingU/rtdetr_r18vd_coco_o365`

## 7. Use a Trained Checkpoint in IrpAI

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
pack = pipeline.run("screenshot.png")
```

## 8. Practical Labeling Guidance

For the current two-class setup:
- `icon`: box tightly around interactive icon glyph
- `text`: box tightly around visible target text

## 9. Troubleshooting

### 9.1 Windows safetensors write-lock (os error 1224)

Symptom:
- save/export fails because training is reading and writing the same checkpoint path simultaneously

Safe approach:
- train from Hub base model id
- write outputs into a checkpoint directory
- only point runtime to local `checkpoint-best` after training/export completes

### 9.2 ML dependencies missing

Install required extras:

```bash
pip install -e ".[training]"
```

### 9.3 Inference too slow on CPU

Options:
- lower input resolution in preprocessing config
- tune confidence threshold / max detections
- keep fallback enabled for resilience

## 10. Model Artifact Sharing

Recommended:
- export `checkpoint-best` to zip in `artifacts/`
- upload zip as a GitHub Release asset
- do not commit checkpoint folders to git

Repository hygiene:
- generated datasets in `data/` are git-ignored
- generated checkpoints in `checkpoints/` are git-ignored
