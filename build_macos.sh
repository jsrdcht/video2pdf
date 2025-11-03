#!/usr/bin/env bash
set -euo pipefail

# Build macOS executable for video_to_pdf_phash.py using PyInstaller
# Requires running on macOS (cross-compiling from Windows/Linux is not supported)

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

if [ ! -d .venv ]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi
python -m pip install pyinstaller

rm -rf build dist

# Build from spec (onefile/onedir controlled by spec)
pyinstaller --clean "$REPO_DIR/video_to_pdf_phash.spec"

echo
echo "Build finished. Output in $REPO_DIR/dist/"

