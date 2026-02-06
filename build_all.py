#!/usr/bin/env python3
"""
Build script for creating all IrpAI distribution variants.

Generates three wheel packages:
1. Core - Minimal dependencies (OpenAI LLM only, PaddleOCR only)
2. Full - All optional runtime dependencies (all LLM providers, all OCR engines)
3. Dev  - Full + development tools (pytest, ruff, mypy, etc.)
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> None:
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(f"✅ Success: {description}")


def clean_build_artifacts():
    """Remove build artifacts from previous builds."""
    print("\n🧹 Cleaning build artifacts...")
    for path in ["build", "dist", "src/irpai.egg-info"]:
        if Path(path).exists():
            shutil.rmtree(path)
            print(f"   Removed: {path}")


def backup_pyproject():
    """Backup original pyproject.toml."""
    print("\n💾 Backing up pyproject.toml...")
    shutil.copy("pyproject.toml", "pyproject.toml.backup")
    print("   Created: pyproject.toml.backup")


def restore_pyproject():
    """Restore original pyproject.toml."""
    print("\n🔄 Restoring pyproject.toml...")
    shutil.copy("pyproject.toml.backup", "pyproject.toml")
    Path("pyproject.toml.backup").unlink()
    print("   Restored: pyproject.toml")


def modify_pyproject_for_variant(variant: str) -> None:
    """Modify pyproject.toml for specific build variant."""
    with open("pyproject.toml", "r", encoding="utf-8") as f:
        content = f.read()
    
    if variant == "full":
        # Add all optional runtime dependencies to core
        content = content.replace(
            '    "python-multipart>=0.0.6",\n]',
            '''    "python-multipart>=0.0.6",
    # Optional LLM providers (included in full bundle)
    "anthropic>=0.34.0",
    "langchain-community>=0.2.0",
    # Optional OCR engines (included in full bundle)
    "easyocr>=1.7.0",
    "pytesseract>=0.3.10",
]'''
        )
        content = content.replace(
            'description = "Image Resolution Pipeline AI - Machine vision preprocessing for enhanced LLM-screenshot interactions"',
            'description = "Image Resolution Pipeline AI - Full Bundle with All Optional Runtime Dependencies"'
        )
    
    elif variant == "dev":
        # Add all optional deps including dev tools
        content = content.replace(
            '    "python-multipart>=0.0.6",\n]',
            '''    "python-multipart>=0.0.6",
    # Optional LLM providers (included in dev bundle)
    "anthropic>=0.34.0",
    "langchain-community>=0.2.0",
    # Optional OCR engines (included in dev bundle)
    "easyocr>=1.7.0",
    "pytesseract>=0.3.10",
    # Development tools (included in dev bundle)
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
]'''
        )
        content = content.replace(
            'description = "Image Resolution Pipeline AI - Machine vision preprocessing for enhanced LLM-screenshot interactions"',
            'description = "Image Resolution Pipeline AI - Development Bundle with All Dependencies + Dev Tools"'
        )
    
    with open("pyproject.toml", "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"   Modified pyproject.toml for {variant} variant")


def build_wheel(variant: str) -> None:
    """Build a wheel for the specified variant."""
    # Clean previous build artifacts
    for path in ["build", "src/irpai.egg-info"]:
        if Path(path).exists():
            shutil.rmtree(path)
    
    # Build the wheel
    run_command(
        [sys.executable, "-m", "build"],
        f"Building {variant} wheel"
    )
    
    # Move files to variant-specific subdirectory
    dist_path = Path("dist")
    variant_path = dist_path / variant
    variant_path.mkdir(exist_ok=True)
    
    for file in dist_path.glob("irpai-*-py3-none-any.whl"):
        dest = variant_path / file.name
        file.rename(dest)
        print(f"   Moved: {file.name} → {variant}/{file.name}")
    
    for file in dist_path.glob("irpai-*.tar.gz"):
        dest = variant_path / file.name
        file.rename(dest)
        print(f"   Moved: {file.name} → {variant}/{file.name}")


def main():
    """Main build orchestration."""
    print("""
╔════════════════════════════════════════════════════════════╗
║         IrpAI Multi-Variant Wheel Build Script            ║
║                                                            ║
║  Building 3 variants:                                      ║
║    • Core - Minimal dependencies                           ║
║    • Full - All runtime dependencies                       ║
║    • Dev  - Full + development tools                       ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Ensure we're in the project root
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Run this script from project root.")
        sys.exit(1)
    
    # Clean and prepare
    clean_build_artifacts()
    Path("dist").mkdir(exist_ok=True)
    backup_pyproject()
    
    try:
        # Build 1: Core variant (default configuration)
        print("\n" + "="*60)
        print("Building Variant 1/3: CORE")
        print("="*60)
        build_wheel("core")
        
        # Build 2: Full variant
        print("\n" + "="*60)
        print("Building Variant 2/3: FULL")
        print("="*60)
        modify_pyproject_for_variant("full")
        build_wheel("full")
        restore_pyproject()
        backup_pyproject()  # Re-backup for next variant
        
        # Build 3: Dev variant
        print("\n" + "="*60)
        print("Building Variant 3/3: DEV")
        print("="*60)
        modify_pyproject_for_variant("dev")
        build_wheel("dev")
        
    finally:
        # Always restore original configuration
        restore_pyproject()
    
    # Generate summary
    print("\n" + "="*60)
    print("📊 BUILD SUMMARY")
    print("="*60)
    
    dist_path = Path("dist")
    variants = ["core", "full", "dev"]
    all_files = []
    
    print("\n✅ Successfully built the following distributions:\n")
    
    for variant in variants:
        variant_path = dist_path / variant
        if variant_path.exists():
            files = sorted(variant_path.glob("*"))
            all_files.extend(files)
            
            if files:
                variant_label = {
                    "core": "CORE (Minimal Dependencies)",
                    "full": "FULL (All Runtime Dependencies)",
                    "dev": "DEV (Full + Dev Tools)"
                }[variant]
                
                print(f"📦 {variant_label}:")
                print(f"   Location: dist/{variant}/")
                for f in files:
                    size = f.stat().st_size / 1024
                    print(f"   • {f.name:<40} ({size:.1f} KB)")
                print()
    
    if all_files:
        print(f"📁 Total files: {len(all_files)}")
        total_size = sum(f.stat().st_size for f in all_files) / 1024
        print(f"💾 Total size: {total_size:.1f} KB")
        
        print("\n" + "="*60)
        print("🎉 All builds completed successfully!")
        print("="*60)
        print("\nInstallation examples:")
        print("  Core:  pip install dist/core/irpai-0.1.0-py3-none-any.whl")
        print("  Full:  pip install dist/full/irpai-0.1.0-py3-none-any.whl")
        print("  Dev:   pip install dist/dev/irpai-0.1.0-py3-none-any.whl")
        print("\nOr install from source with extras:")
        print("  pip install dist/core/irpai-0.1.0-py3-none-any.whl[llm,ocr]")
        print("  pip install dist/core/irpai-0.1.0-py3-none-any.whl[dev]")
    else:
        print("❌ No distribution files found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
