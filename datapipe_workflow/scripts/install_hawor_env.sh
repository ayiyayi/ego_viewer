#!/usr/bin/env bash
set -euo pipefail

# Install HaWoR runtime dependencies into an existing conda environment.
#
# Usage:
#   bash scripts/install_hawor_env.sh
#   HAWOR_PYTHON=/path/to/env/bin/python bash scripts/install_hawor_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAWOR_DIR="$ROOT_DIR/pipelines/hawor"
PYTHON_BIN="${HAWOR_PYTHON:-/cpfs/user/liuyu2/miniconda3/envs/hawor/bin/python}"
START_STEP="${START_STEP:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$HAWOR_DIR"

run_step() {
  local step="$1"
  [[ "$START_STEP" -le "$step" ]]
}

echo "[env] python: $PYTHON_BIN"
"$PYTHON_BIN" -V
"$PYTHON_BIN" -m pip --version

if run_step 1; then
  echo "[1/8] Upgrade packaging tools"
  "$PYTHON_BIN" -m pip install -U pip wheel packaging ninja
  "$PYTHON_BIN" -m pip install 'setuptools<70'
fi

if run_step 2; then
  echo "[2/8] Install PyTorch CUDA 11.7 build"
  "$PYTHON_BIN" -m pip install \
    torch==1.13.0+cu117 \
    torchvision==0.14.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
fi

if run_step 3; then
  echo "[3/8] Pin NumPy/OpenCV for torch/torchvision ABI compatibility"
  # opencv-python 4.13 requires NumPy >= 2, but torch 1.13/torchvision 0.14 expect NumPy 1.x.
  # Use an older headless OpenCV wheel that works with NumPy 1.26.
  "$PYTHON_BIN" -m pip uninstall -y opencv-python opencv-python-headless || true
  "$PYTHON_BIN" -m pip install numpy==1.26.4 opencv-python-headless==4.10.0.84
fi

if run_step 4; then
  echo "[4/8] Install HaWoR Python requirements except OpenCV/mmcv/torch-scatter/PyTorch3D/chumpy"
  REQ_TMP="$(mktemp)"
  grep -v -E '^(opencv-python|mmcv|torch-scatter|git\+https://github\.com/facebookresearch/pytorch3d\.git|chumpy@|chumpy @)' requirements.txt > "$REQ_TMP"
  "$PYTHON_BIN" -m pip install -r "$REQ_TMP"
  rm -f "$REQ_TMP"
fi

if run_step 5; then
  echo "[5/8] Install mmcv, torch-scatter, and chumpy"
  # mmcv 1.3.9 setup imports pkg_resources. Keep setuptools old enough even when resuming from START_STEP=5.
  "$PYTHON_BIN" -m pip install 'setuptools<70'
  "$PYTHON_BIN" - <<'PY'
import pkg_resources
print('pkg_resources: ok')
PY
  "$PYTHON_BIN" -m pip install --no-build-isolation mmcv==1.3.9
  # Use the PyG wheel matching torch 1.13.0 + CUDA 11.7 to avoid local torch-scatter builds.
  "$PYTHON_BIN" -m pip install --no-build-isolation torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
  # chumpy is an old package; build isolation can fail because its setup imports pip.
  "$PYTHON_BIN" -m pip install --no-build-isolation 'chumpy@git+https://github.com/mattloper/chumpy'
  # PyTorch3D needs CUDA_HOME/nvcc to build GPU rasterization. It is installed in step 9 after CUDA checks.
fi

if run_step 6; then
  echo "[6/8] Install Lightning compatibility packages"
  "$PYTHON_BIN" -m pip install pytorch-lightning==2.2.4 --no-deps
  "$PYTHON_BIN" -m pip install lightning-utilities torchmetrics==1.4.0
fi

if run_step 7; then
  echo "[7/8] Install HaWoR service dependencies"
  "$PYTHON_BIN" -m pip install -r service/requirements.txt
fi

if run_step 8; then
  echo "[8/9] Build/install DROID-SLAM extension"
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
      export CUDA_HOME
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
      CUDA_HOME="/usr/local/cuda"
      export CUDA_HOME
    fi
  fi

  if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    cat >&2 <<'EOF'
CUDA_HOME is not set and nvcc was not found.
DROID-SLAM requires the CUDA Toolkit compiler, not only the PyTorch CUDA runtime.

Install CUDA Toolkit 11.7 into the conda environment, then rerun step 8:

  /cpfs/user/liuyu2/miniconda3/bin/conda install -n hawor -c nvidia cuda-toolkit=11.7
  CUDA_HOME=/cpfs/user/liuyu2/miniconda3/envs/hawor START_STEP=8 bash scripts/install_hawor_env.sh

If your cluster already has CUDA installed elsewhere, use that path instead, for example:

  CUDA_HOME=/usr/local/cuda-11.7 START_STEP=8 bash scripts/install_hawor_env.sh
EOF
    exit 1
  fi

  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  if [[ -z "${CXX:-}" ]]; then
    if [[ -x "$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_config_var('BINDIR') or '')
PY
)/x86_64-conda-linux-gnu-g++" ]]; then
      CONDA_BINDIR="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_config_var('BINDIR') or '')
PY
)"
      export CC="$CONDA_BINDIR/x86_64-conda-linux-gnu-gcc"
      export CXX="$CONDA_BINDIR/x86_64-conda-linux-gnu-g++"
    elif [[ -x /usr/bin/g++-11 && -x /usr/bin/gcc-11 ]]; then
      export CC=/usr/bin/gcc-11
      export CXX=/usr/bin/g++-11
    fi
  fi

  if [[ -z "${CXX:-}" || ! -x "$CXX" ]]; then
    cat >&2 <<'EOF'
No CUDA 11.7-compatible g++ was found. CUDA 11.7 requires g++ >= 6 and <= 11.5.

Install GCC/G++ 11 into the hawor conda environment, then rerun step 8:

  /cpfs/user/liuyu2/miniconda3/bin/conda install -n hawor -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
  CUDA_HOME=/cpfs/user/liuyu2/miniconda3/envs/hawor START_STEP=8 bash scripts/install_hawor_env.sh
EOF
    exit 1
  fi

  CXX_VERSION="$($CXX -dumpfullversion -dumpversion | head -1)"
  CXX_MAJOR="${CXX_VERSION%%.*}"
  if [[ "$CXX_MAJOR" -gt 11 ]]; then
    cat >&2 <<EOF
The selected C++ compiler is too new for CUDA 11.7.
  CXX         = $CXX
  CXX version = $CXX_VERSION

CUDA 11.7 requires g++ >= 6 and <= 11.5. Install GCC/G++ 11 into the hawor env:

  /cpfs/user/liuyu2/miniconda3/bin/conda install -n hawor -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
  CUDA_HOME=/cpfs/user/liuyu2/miniconda3/envs/hawor START_STEP=8 bash scripts/install_hawor_env.sh
EOF
    exit 1
  fi

  echo "[cuda] CUDA_HOME=$CUDA_HOME"
  "$CUDA_HOME/bin/nvcc" --version
  echo "[compiler] CC=${CC:-}"
  echo "[compiler] CXX=$CXX ($CXX_VERSION)"

  TORCH_CUDA="$($PYTHON_BIN - <<'PY'
import torch
print(torch.version.cuda or '')
PY
)"
  NVCC_CUDA="$($CUDA_HOME/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
  if [[ -n "$TORCH_CUDA" && -n "$NVCC_CUDA" && "$TORCH_CUDA" != "$NVCC_CUDA" ]]; then
    cat >&2 <<EOF
CUDA version mismatch.
  torch.version.cuda = $TORCH_CUDA
  nvcc release       = $NVCC_CUDA

DROID-SLAM must be compiled with the CUDA Toolkit matching PyTorch.
For the current torch cu117 environment, install CUDA Toolkit 11.7 and rerun:

  /cpfs/user/liuyu2/miniconda3/bin/conda remove -n hawor cuda-toolkit cuda-nvcc cuda-compiler cuda-libraries-dev cuda-cudart-dev -y
  /cpfs/user/liuyu2/miniconda3/bin/conda install -n hawor -c nvidia/label/cuda-11.7.0 cuda-toolkit -y
  CUDA_HOME=/cpfs/user/liuyu2/miniconda3/envs/hawor START_STEP=8 bash scripts/install_hawor_env.sh
EOF
    exit 1
  fi

  if [[ -d "$HAWOR_DIR/thirdparty/DROID-SLAM" ]]; then
    (cd "$HAWOR_DIR/thirdparty/DROID-SLAM" && "$PYTHON_BIN" setup.py install)
  else
    echo "DROID-SLAM directory not found; did you initialize submodules?" >&2
    exit 1
  fi
fi

echo "[check] Import critical packages"
"$PYTHON_BIN" - <<'PY'
import importlib
mods = [
    'torch', 'torchvision', 'cv2', 'numpy', 'smplx', 'yacs', 'ultralytics',
    'mmcv', 'torch_scatter', 'pytorch3d', 'pytorch_lightning', 'torchmetrics', 'fastapi', 'pydantic'
]
for name in mods:
    mod = importlib.import_module(name)
    print(f'{name}: {getattr(mod, "__version__", "ok")}')
import torch
print('cuda_available:', torch.cuda.is_available())
print('torch_cuda:', torch.version.cuda)
PY

if run_step 9; then
  echo "[9/9] Rebuild PyTorch3D with CUDA support"
  if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="/cpfs/user/liuyu2/miniconda3/envs/hawor"
    export CUDA_HOME
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  CONDA_BINDIR="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_config_var('BINDIR') or '')
PY
)"
  if [[ -x "$CONDA_BINDIR/x86_64-conda-linux-gnu-g++" ]]; then
    export CC="$CONDA_BINDIR/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_BINDIR/x86_64-conda-linux-gnu-g++"
  fi
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "CUDA nvcc not found at $CUDA_HOME/bin/nvcc" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m pip uninstall -y pytorch3d || true
  FORCE_CUDA=1 "$PYTHON_BIN" -m pip install --no-build-isolation --no-cache-dir \
    'git+https://github.com/facebookresearch/pytorch3d.git@stable'
fi

echo "[done] HaWoR environment dependencies installed."
echo "[note] Model weights and MANO files are not installed by this script."
echo "       Check pipelines/hawor/README.md for required weights under weights/, thirdparty/Metric3D/weights/, and _DATA/."
