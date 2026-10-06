#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_ROOT=${TAGC_BUILD_ROOT:-"$REPO_ROOT/build"}
STORAGE_ROOT=${TAGC_STORAGE_ROOT:-/share/bigyang}
mkdir -p "$BUILD_ROOT" "$STORAGE_ROOT/tmp" "$STORAGE_ROOT/cache" "$STORAGE_ROOT/pip-cache"
BUILD_ROOT=$(cd "$BUILD_ROOT" && pwd)
STORAGE_ROOT=$(cd "$STORAGE_ROOT" && pwd)
export APPTAINER_TMPDIR="$STORAGE_ROOT/tmp"
export APPTAINER_CACHEDIR="$STORAGE_ROOT/cache"
export TMPDIR="$APPTAINER_TMPDIR"

cd "$REPO_ROOT"
git submodule update --init

if [ "${1:-}" != "--install-only" ]; then
    apptainer build --force "$BUILD_ROOT/tagc.sif" "$REPO_ROOT/tagc.def"
fi

# Installation runs on the internet-connected node and does not need a GPU.
apptainer exec --userns --cleanenv --nv \
    --bind "$REPO_ROOT:/workspace/TAGC" \
    --bind "$STORAGE_ROOT:$STORAGE_ROOT" \
    --env "PIP_CACHE_DIR=$STORAGE_ROOT/pip-cache,TMPDIR=$STORAGE_ROOT/tmp" \
    --env "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}" \
    --pwd /workspace/TAGC "$BUILD_ROOT/tagc.sif" bash -e -c '
        python3.12 -m venv --clear .venv
        source .venv/bin/activate
        unset PYTHONPATH
        python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
        python -m pip install -r requirements.txt

        TORCH_SITE=$(python -c "import sysconfig; print(sysconfig.get_path(\"purelib\"))")
        if ! git apply --reverse --check --unsafe-paths --directory "$TORCH_SITE" patches/torch.diff 2>/dev/null; then
            git apply --unsafe-paths --directory "$TORCH_SITE" patches/torch.diff
        fi

        python -m pip install --no-build-isolation ./lossless_homomorphic_compression/lossless_homomorphic_compression_api
        python -m pip install .
    '

echo "Ready: $BUILD_ROOT/tagc.sif and $REPO_ROOT/.venv"
