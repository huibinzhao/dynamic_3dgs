#!/bin/bash
# ============================================================
# Integrated DROID-W + Dynamic 3DGS Pipeline Launcher
# ============================================================
#
# Usage:
#   ./run_integrated.sh [config_file]
#   ./run_integrated.sh mrhash/configurations/tum_integrated.cfg
#
# This script activates the dynamic_3dgs conda environment and
# runs the integrated pipeline.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Default config
CONFIG="${1:-mrhash/configurations/tum_integrated.cfg}"

# Conda environment name
CONDA_ENV="dynamic_3dgs"

echo "============================================================"
echo "  DROID-W + Dynamic 3DGS Integrated Pipeline"
echo "============================================================"
echo "  Project dir : $PROJECT_DIR"
echo "  Config      : $CONFIG"
echo "  Conda env   : $CONDA_ENV"
echo "============================================================"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Ensure conda lib is first in LD_LIBRARY_PATH (avoid ROS conflicts)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$PROJECT_DIR"

# Run the integrated pipeline
python mrhash/apps/integrated_tum_runner.py "$CONFIG"
