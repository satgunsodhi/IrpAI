# Release Guide

## Automated GitHub Releases

The project is now configured to automatically build and publish all 3 variants to GitHub Releases when you push a version tag.

## Creating a New Release

### 1. Update Version Number

First, update the version in `pyproject.toml`:

```toml
[project]
name = "irpai"
version = "0.2.0"  # Update this
```

### 2. Commit Changes

```bash
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push
```

### 3. Create and Push Version Tag

```bash
git tag v0.2.0
git push origin v0.2.0
```

**That's it!** GitHub Actions will automatically:
- ✅ Build all 3 variants (core, full, dev)
- ✅ Create a GitHub Release with version notes
- ✅ Upload all 6 distribution files as release assets
- ✅ Generate a professional release page

## What Gets Published

Each release includes 6 files organized in the release assets:

```
📦 Core Variant
  - irpai-X.Y.Z-py3-none-any.whl (from dist/core/)
  - irpai-X.Y.Z.tar.gz (from dist/core/)

📦 Full Variant
  - irpai-X.Y.Z-py3-none-any.whl (from dist/full/)
  - irpai-X.Y.Z.tar.gz (from dist/full/)

📦 Dev Variant
  - irpai-X.Y.Z-py3-none-any.whl (from dist/dev/)
  - irpai-X.Y.Z.tar.gz (from dist/dev/)
```

## Release Notes

The workflow automatically generates comprehensive release notes including:
- Installation instructions for each variant
- Variant comparison table
- Optional extras documentation
- Asset descriptions

## Manual Release (Optional)

If you need to create a release manually:

```bash
# 1. Build all variants locally
python build_all.py

# 2. Go to GitHub > Releases > Draft a new release
# 3. Choose a tag (or create new tag like v0.2.0)
# 4. Upload files from:
#    - dist/core/
#    - dist/full/
#    - dist/dev/
# 5. Add release notes and publish
```

## Troubleshooting

### Workflow doesn't trigger
- Ensure tag starts with 'v' (e.g., v0.1.0, not 0.1.0)
- Check that you pushed the tag: `git push origin v0.1.0`

### Build fails
- Verify `python build_all.py` works locally
- Check GitHub Actions logs for details

### Release creation fails
- Ensure repository has GitHub Actions enabled
- Verify GITHUB_TOKEN permissions (should be automatic)

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes, backwards compatible

Examples:
- `v0.1.0` - Initial release
- `v0.1.1` - Bug fix
- `v0.2.0` - New feature
- `v1.0.0` - First stable release
