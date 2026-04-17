# Building IrpAI as a Wheel

## Quick Build

```powershell
# Install build tools
pip install build

# Build wheel (creates dist/irpai-0.1.0-py3-none-any.whl)
python -m build --wheel

# Install the wheel
pip install dist/irpai-0.1.0-py3-none-any.whl
```

## Development Installation (Editable)

```powershell
# Install core + all optional extras in editable mode
pip install -e ".[all]"

# Or install only what you need:
pip install -e .                      # core only (no OCR, no ML detector)
pip install -e ".[ocr-easyocr]"       # + EasyOCR
pip install -e ".[detection-ml]"      # + RT-DETR
pip install -e ".[training]"          # + training scripts
pip install -e ".[dev]"               # + pytest, ruff, mypy
```

## Distribution

```powershell
# Build both wheel and source distribution
python -m build

# This creates:
# - dist/irpai-0.1.0-py3-none-any.whl  (wheel)
# - dist/irpai-0.1.0.tar.gz            (source)
```

## Multi-Variant Automated Builds

The GitHub Actions workflow (`.github/workflows/`) automatically builds all distribution variants when a version tag is pushed. Refer to [RELEASE.md](RELEASE.md) for the full release workflow.

For local multi-variant builds:

```powershell
python build_all.py
```

## Advantages of Wheel Installation

- **Faster installation**: Pre-built, no compilation needed
- **Consistent**: Same build across all platforms
- **Standards-compliant**: Modern Python packaging (PEP 517, PEP 660)
- **Cleaner**: No setup.py needed
- **Cache-friendly**: pip caches wheels for faster re-installs

## Verify Installation

```powershell
pip show irpai
irpai --help
```
