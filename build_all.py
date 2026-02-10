#!/usr/bin/env python3
"""
Build script for IrpAI package with optional extras.

Generates a single wheel that supports multiple install modes:
- pip install irpai           # Minimal
- pip install irpai[core]     # + ML features
- pip install irpai[full]     # All runtime deps
- pip install irpai[dev]      # Development environment
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
    if result.stdout:
        print(result.stdout)


def clean_build_artifacts():
    """Remove build artifacts from previous builds."""
    print("\n🧹 Cleaning build artifacts...")
    for path in ["build", "dist", "src/irpai.egg-info"]:
        if Path(path).exists():
            shutil.rmtree(path)
            print(f"   Removed: {path}")


def ensure_build_tools():
    """Ensure build package is installed."""
    print("\n🔧 Checking build tools...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "build"],
        capture_output=True
    )
    if result.returncode != 0:
        print("   Installing build package...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "build"]
        )
    print("   ✅ Build tools ready")


def build_wheel() -> None:
    """Build the wheel package."""
    run_command(
        [sys.executable, "-m", "build", "--wheel"],
        "Building wheel"
    )


def display_summary():
    """Display build summary with installation examples."""
    print("\n" + "="*60)
    print("📊 BUILD SUMMARY")
    print("="*60)
    
    dist_path = Path("dist")
    if not dist_path.exists():
        print("❌ No dist directory found!")
        sys.exit(1)
    
    wheels = list(dist_path.glob("*.whl"))
    tarballs = list(dist_path.glob("*.tar.gz"))
    
    if not wheels:
        print("❌ No wheel files found!")
        sys.exit(1)
    
    print("\n✅ Successfully built:\n")
    
    for wheel in wheels:
        size = wheel.stat().st_size / 1024
        print(f"📦 {wheel.name}")
        print(f"   Size: {size:.1f} KB")
        print(f"   Path: {wheel}")
    
    for tarball in tarballs:
        size = tarball.stat().st_size / 1024
        print(f"\n📄 {tarball.name}")
        print(f"   Size: {size:.1f} KB")
        print(f"   Path: {tarball}")
    
    total_size = sum(f.stat().st_size for f in wheels + tarballs) / 1024
    print(f"\n💾 Total size: {total_size:.1f} KB")
    
    print("\n" + "="*60)
    print("🎉 Build completed successfully!")
    print("="*60)
    
    wheel_name = wheels[0].name
    print(f"\n📥 Installation examples:\n")
    print(f"Minimal (opencv, numpy, pillow only):")
    print(f"  pip install dist/{wheel_name}\n")
    
    print(f"Core (+ torch, transformers):")
    print(f"  pip install dist/{wheel_name}[core]\n")
    
    print(f"Full (all runtime dependencies):")
    print(f"  pip install dist/{wheel_name}[full]\n")
    
    print(f"Development (full + dev tools):")
    print(f"  pip install dist/{wheel_name}[dev]\n")
    
    print(f"Custom combinations:")
    print(f"  pip install dist/{wheel_name}[ocr,llm]")
    print(f"  pip install dist/{wheel_name}[full,gpu]")
    
    print("\n" + "="*60)


def main():
    """Main build orchestration."""
    print("""
╔════════════════════════════════════════════════════════════╗
║            IrpAI Package Build Script                      ║
║                                                            ║
║  Builds a single wheel with optional extras:               ║
║    • Minimal - Core dependencies only                      ║
║    • [core]  - + ML features                               ║
║    • [full]  - All runtime dependencies                    ║
║    • [dev]   - Development environment                     ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Ensure we're in the project root
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Run this script from project root.")
        sys.exit(1)
    
    try:
        # Prepare environment
        ensure_build_tools()
        clean_build_artifacts()
        
        # Build the wheel
        build_wheel()
        
        # Show results
        display_summary()
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
