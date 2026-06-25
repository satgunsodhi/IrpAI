#!/usr/bin/env python3
"""
Automated Version Bumping Script for IrpAI
==========================================

Usage:
    python scripts/bump_version.py --patch
    python scripts/bump_version.py --minor
    python scripts/bump_version.py --major
    python scripts/bump_version.py --current  (to view current version)

This script reads the version from pyproject.toml and src/irpai/__init__.py,
updates them, and optionally commits and tags the change in Git.
"""

import argparse
import re
import sys
import subprocess
from pathlib import Path

# Files to update
FILES_TO_UPDATE = [
    Path("pyproject.toml"),
    Path("src/irpai/__init__.py"),
]

# Regex patterns for finding the version string
VERSION_PATTERNS = {
    "pyproject.toml": re.compile(r'^(version\s*=\s*")(\d+\.\d+\.\d+)(")$', re.MULTILINE),
    "__init__.py": re.compile(r'^(__version__\s*=\s*")(\d+\.\d+\.\d+)(")$', re.MULTILINE),
}

def run_cmd(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_current_version() -> str:
    """Read the current version from pyproject.toml."""
    content = FILES_TO_UPDATE[0].read_text(encoding="utf-8")
    match = VERSION_PATTERNS["pyproject.toml"].search(content)
    if not match:
        print("❌ Could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return match.group(2)

def update_version_in_file(path: Path, new_version: str) -> None:
    if not path.exists():
        print(f"⚠️ Warning: File {path} not found, skipping.")
        return
        
    content = path.read_text(encoding="utf-8")
    
    # Determine which pattern to use based on filename
    pattern = None
    for key, p in VERSION_PATTERNS.items():
        if path.name.endswith(key):
            pattern = p
            break
            
    if not pattern:
        print(f"⚠️ Warning: No regex pattern for {path.name}")
        return

    # Check if pattern matches
    if not pattern.search(content):
        print(f"❌ Version string not found in {path}")
        sys.exit(1)
        
    new_content = pattern.sub(rf'\g<1>{new_version}\g<3>', content)
    path.write_text(new_content, encoding="utf-8")
    print(f"✅ Updated {path} to {new_version}")

def bump_version(version: str, part: str) -> str:
    """Bump the major, minor, or patch part of the version string."""
    major, minor, patch = map(int, version.split('.'))
    
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown part: {part}")
        
    return f"{major}.{minor}.{patch}"

def main():
    parser = argparse.ArgumentParser(description="Bump semantic version numbers.")
    parser.add_argument("--major", action="store_true", help="Bump major version (X.0.0)")
    parser.add_argument("--minor", action="store_true", help="Bump minor version (0.X.0)")
    parser.add_argument("--patch", action="store_true", help="Bump patch version (0.0.X)")
    parser.add_argument("--current", action="store_true", help="Show current version and exit")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files or run git commands")
    parser.add_argument("--no-git", action="store_true", help="Skip git commit and tag")
    
    args = parser.parse_args()
    
    # Ensure we're at project root
    if not Path("pyproject.toml").exists():
        print("❌ Error: Must run from project root directory")
        sys.exit(1)
        
    current_version = get_current_version()
    
    if args.current:
        print(current_version)
        return
        
    if args.major:
        part = "major"
    elif args.minor:
        part = "minor"
    elif args.patch:
        part = "patch"
    else:
        parser.print_help()
        sys.exit(1)
        
    new_version = bump_version(current_version, part)
    
    print(f"Bumping version: {current_version} -> {new_version}")
    
    if args.dry_run:
        print("Dry run complete. No files changed.")
        return
        
    # Update files
    for filepath in FILES_TO_UPDATE:
        update_version_in_file(filepath, new_version)
        
    # Git commit and tag
    if not args.no_git:
        try:
            print("\nCreating Git commit and tag...")
            run_cmd(["git", "add"] + [str(f) for f in FILES_TO_UPDATE])
            run_cmd(["git", "commit", "-m", f"Bump version to {new_version}"])
            run_cmd(["git", "tag", f"v{new_version}"])
            print(f"\n🎉 Successfully bumped version to {new_version}!")
            print(f"Run `git push --follow-tags` to push the commit and tag to the remote.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operations failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
