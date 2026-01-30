""" Script to download checkpoint and data for FollowMyHold """

set -euo pipefail

cd FollowMyHold

# HaMeR ckpts from https://github.com/geopavlakos/hamer/blob/main/fetch_demo_data.sh
gdown https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT
# OR use wget --> wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
mv hamer_demo_data third_party/estimator/hamer/_DATA

# download WiLoR hand detector model from https://github.com/rolpotamias/WiLoR
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./third_party/estimator/wilor_ckpt/
export WILOR_CKPT="./third_party/estimator/wilor_ckpt/detector.pt"
