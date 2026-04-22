#!/bin/bash
# Build all DROID-W CUDA extensions from within dynamic_3dgs project.
# Usage: cd /path/to/dynamic_3dgs/mrhash/src/droidw && bash build_extensions.sh
#
# Prerequisites:
#   - conda activate dynamic_3dgs
#   - CUDA toolkit installed
#   - PyTorch with CUDA support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Building DROID-W CUDA Extensions"
echo "============================================"
echo ""

# 1. Build droid_backends
echo "[1/4] Building droid_backends..."
pip install -e . --no-build-isolation 2>&1 | tail -5
echo "      droid_backends done."
echo ""

# 2. Build lietorch
echo "[2/4] Building lietorch..."
cd thirdparty/lietorch
pip install -e . --no-build-isolation 2>&1 | tail -5
cd "$SCRIPT_DIR"
echo "      lietorch done."
echo ""

# 3. Build diff_gaussian_rasterization
echo "[3/4] Building diff_gaussian_rasterization..."
cd thirdparty/diff-gaussian-rasterization-w-pose
pip install -e . --no-build-isolation 2>&1 | tail -5
cd "$SCRIPT_DIR"
echo "      diff_gaussian_rasterization done."
echo ""

# 4. Build simple_knn
echo "[4/4] Building simple_knn..."
cd thirdparty/simple-knn
pip install -e . --no-build-isolation 2>&1 | tail -5
cd "$SCRIPT_DIR"
echo "      simple_knn done."
echo ""

echo "============================================"
echo "  All DROID-W extensions built successfully"
echo "============================================"
