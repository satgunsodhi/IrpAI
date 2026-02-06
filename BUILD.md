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
# Install in editable mode (uses PEP 660 wheel-based editable installs)
pip install -e .
```

## Distribution

```powershell
# Build both wheel and source distribution
python -m build

# This creates:
# - dist/irpai-0.1.0-py3-none-any.whl  (wheel)
# - dist/irpai-0.1.0.tar.gz            (source)
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
irpai info
```
