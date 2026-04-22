"""
DROID-W integration module for dynamic_3dgs.

This module wraps DROID-W's visual odometry pipeline to provide
estimated camera poses for dynamic_3dgs mapping.

DROID-W source code is embedded under mrhash/src/droidw/.
CUDA extensions (droid_backends, lietorch, etc.) must be built via:
    cd mrhash/src/droidw && bash build_extensions.sh
"""

import sys
import os

# Ensure the conda env lib is prioritized to avoid ROS libtiff conflicts
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix:
    _conda_lib = os.path.join(_conda_prefix, "lib")
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _conda_lib not in _ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = _conda_lib + (":" + _ld if _ld else "")

# DROIDW_ROOT points to THIS directory (mrhash/src/droidw/)
DROIDW_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add DROIDW_ROOT to sys.path so that internal DROID-W imports
# like "from src.xxx import ..." and "from thirdparty.xxx import ..." work
if DROIDW_ROOT not in sys.path:
    sys.path.insert(0, DROIDW_ROOT)
