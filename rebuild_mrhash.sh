#!/usr/bin/env bash
# Rebuild mrhash core targets and install pygeowrapper into the conda env.
#
# Default usage:
#   ./rebuild_mrhash.sh
#
# Useful options:
#   ./rebuild_mrhash.sh -j4
#   ./rebuild_mrhash.sh --no-install
#   ./rebuild_mrhash.sh --tests --apps

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-dynamic_3dgs}"
BUILD_DIR="${BUILD_DIR:-build/cp311-cp311-linux_x86_64}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
BUILD_TESTS="OFF"
BUILD_APPS="OFF"
INSTALL_SO="1"

usage() {
  cat <<'EOF'
Rebuild mrhash core targets and install pygeowrapper into the conda env.

Default usage:
  ./rebuild_mrhash.sh

Useful options:
  ./rebuild_mrhash.sh -j4
  ./rebuild_mrhash.sh --no-install
  ./rebuild_mrhash.sh --tests --apps

Options:
  -j, --jobs N        Number of build jobs. Default: 2
  --build-dir DIR     CMake build directory. Default: build/cp311-cp311-linux_x86_64
  --env NAME          Conda environment name. Default: dynamic_3dgs
  --debug             Use Debug build type
  --relwithdebinfo    Use RelWithDebInfo build type
  --tests             Also build tests
  --apps              Also build example apps
  --no-install        Compile only; do not copy pygeowrapper into site-packages
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)
      JOBS="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --relwithdebinfo)
      BUILD_TYPE="RelWithDebInfo"
      shift
      ;;
    --tests)
      BUILD_TESTS="ON"
      shift
      ;;
    --apps)
      BUILD_APPS="ON"
      shift
      ;;
    --no-install)
      INSTALL_SO="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH." >&2
  exit 1
fi

cd "$PROJECT_DIR"

echo "============================================================"
echo "  Rebuild mrhash"
echo "============================================================"
echo "  Project dir : $PROJECT_DIR"
echo "  Conda env   : $CONDA_ENV"
echo "  Build dir   : $BUILD_DIR"
echo "  Build type  : $BUILD_TYPE"
echo "  Jobs        : $JOBS"
echo "  Tests       : $BUILD_TESTS"
echo "  Apps        : $BUILD_APPS"
echo "  Install .so : $INSTALL_SO"
echo "============================================================"

conda run -n "$CONDA_ENV" cmake \
  -S mrhash \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DMRHASH_BUILD_TESTS="$BUILD_TESTS" \
  -DMRHASH_BUILD_APPS="$BUILD_APPS"

conda run -n "$CONDA_ENV" cmake --build "$BUILD_DIR" -j"$JOBS"

EXT_SUFFIX="$(conda run -n "$CONDA_ENV" python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' | tail -n 1)"
SITE_PACKAGES="$(conda run -n "$CONDA_ENV" python -c 'import site; print(site.getsitepackages()[0])' | tail -n 1)"

BUILT_SO="$BUILD_DIR/src/sdf/pybind/pygeowrapper${EXT_SUFFIX}"
TARGET_DIR="$SITE_PACKAGES/mrhash/src"
TARGET_SO="$TARGET_DIR/pygeowrapper${EXT_SUFFIX}"

if [[ ! -f "$BUILT_SO" ]]; then
  echo "Built extension was not found: $BUILT_SO" >&2
  exit 1
fi

if [[ "$INSTALL_SO" == "1" ]]; then
  mkdir -p "$TARGET_DIR"
  install -m 755 "$BUILT_SO" "$TARGET_SO"

  echo "------------------------------------------------------------"
  echo "Installed:"
  echo "  $TARGET_SO"
  echo "SHA256:"
  sha256sum "$BUILT_SO" "$TARGET_SO"

  echo "Python import target:"
  conda run -n "$CONDA_ENV" python -c \
    "import importlib.util; spec = importlib.util.find_spec('mrhash.src.pygeowrapper'); print(spec.origin if spec else 'NOT_FOUND')"
else
  echo "Build finished without installing:"
  echo "  $BUILT_SO"
fi

echo "Done."
