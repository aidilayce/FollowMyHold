#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="foho"
PY_VER="3.10"
CONDA_SH="${CONDA_SH:-}"
FOHO_ROOT="FollowMyHold"

conda create -y -n "${ENV_NAME}" "python=${PY_VER}"

if [[ -z "${CONDA_SH}" ]]; then
  echo "[FOHO] Set CONDA_SH to your conda.sh path before running this script."
  echo "[FOHO] Example: export CONDA_SH=/path/to/miniforge3/etc/profile.d/conda.sh"
  exit 1
fi
source "${CONDA_SH}"
conda activate "${ENV_NAME}"

# # OPTIONAL (RECOMMENDATION): figure the following flags to put checkpoints and cache files in a common place
# conda env config vars set HF_HOME=/path/to/cache
# conda env config vars set HF_DATASETS_CACHE=/path/to/cache
# conda env config vars set HF_HUB_CACHE=/path/to/cache
# conda env config vars set U2NET_HOME=/path/to/cache
# conda env config vars set TRANSFORMERS_CACHE=/path/to/cache
# conda env config vars set TORCH_HOME=/path/to/cache
# conda env config vars set HUGGINGFACE_HUB_CACHE=/path/to/cache
# conda env config vars set PIP_CACHE_DIR=/path/to/cache
# conda env config vars set CONDA_PKGS_DIRS=/path/to/cache
# conda deactivate
# conda activate "${ENV_NAME}"

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.0+cu124 torchvision==0.20.0+cu124

pip install \
  nvidia-cublas-cu12==12.4.5.8 \
  nvidia-cuda-nvrtc-cu12==12.4.127 \
  nvidia-cuda-runtime-cu12==12.4.127 \
  nvidia-cudnn-cu12==9.1.0.70 \
  nvidia-cusparse-cu12==12.3.1.170 \
  nvidia-nvjitlink-cu12==12.4.127

ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=\
$ENV_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:\
$ENV_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:\
$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

pip install \
  accelerate safetensors \
  google-generativeai \
  trimesh pyvista kiui opencv-python pandas tqdm pillow scipy scikit-image pymeshlab \
  ultralytics \
  sam2 --no-deps \
  segment-anything --no-deps \
  hqq \
  git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900 \
  detectron2==0.6 \
  smplx==0.1.28 \
  mmcv==1.3.9 \
  pytorch-lightning==2.6.0 \
  pyrender timm webdataset pycocotools 

pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html

# building ddshan's hand_object_detector, for details, check https://github.com/ddshan/hand_object_detector
cd "$FOHO_ROOT/third_party/estimator/hand_object_detector/lib"
python setup.py build_ext --inplace

# building pytorch3d
pip install --upgrade setuptools wheel
cd "$FOHO_ROOT/third_party"
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e . --no-build-isolation
cd ../..

pip install git+https://github.com/mattloper/chumpy --no-build-isolation
pip install mmpose mmengine

cd third_party/estimator/hamer
pip install -v -e third-party/ViTPose
cd ../../..

pip uninstall onnxruntime-gpu
pip install "rembg[gpu]"

pip uninstall numpy
pip install numpy==1.24.0

pip install diffusers==0.35.0
pip install transformers==4.54.0
pip install huggingface_hub==0.34.3