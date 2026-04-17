# Release Guide

This guide describes how to publish an IrpAI release using the existing GitHub workflow.

Workflow file:
- `.github/workflows/release.yml`

Trigger:
- push a tag matching `v*` (example: `v0.1.1`)

## 1. Before Tagging

1. Update version in `pyproject.toml`.
2. Confirm docs and changelog notes are current.
3. Validate builds locally:

```bash
python -m build
```

4. Run tests:

```bash
pytest
```

## 2. Create a Release Tag

```bash
git add pyproject.toml
git commit -m "Bump version to 0.1.1"
git push

git tag v0.1.1
git push origin v0.1.1
```

After tag push, GitHub Actions will:
- build package artifacts into `dist/`
- create a GitHub release
- upload `dist/*` as release assets

## 3. Published Artifacts

Current workflow publishes one package build (not multiple variant directories):
- wheel file(s) from `dist/*.whl`
- source distribution from `dist/*.tar.gz`

## 4. Manual Release (Fallback)

```bash
python -m build
```

Then create a GitHub release manually and upload files from `dist/`.

## 5. Keep Metadata Aligned

Keep these synchronized to avoid stale release notes:
- `pyproject.toml` optional dependency names
- README install examples
- `.github/workflows/release.yml` generated release notes content

## 6. Versioning

Use semantic versioning (`MAJOR.MINOR.PATCH`):
- `v0.1.1` patch fixes
- `v0.2.0` new backward-compatible features
- `v1.0.0` first stable major
