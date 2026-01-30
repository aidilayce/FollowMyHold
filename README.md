# Follow My Hold: Hand-Object Interaction Reconstruction through Geometric Guidance

Official implementation of FollowMyHold (3DV 2026).
<a href='https://aidilayce.github.io/FollowMyHold-page/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/pdf/2508.18213'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>

![teaser](assets/teaser.png)

### 🚀 Updates:
- **[January 30 2026]** Full code is released!
- **[December 4 2025]** Added core model logic files (see `src/`) to make it easier to understand how the method works.
- **[November 14 2025]** <a href="https://github.com/aidilayce/FollowMyHold/tree/main/test_splits">Test splits</a> are now available.
- **[November 9 2025]** Got accepted to 3DV 2026!

## Structure
- `configs/` pipeline configs (`pipeline.env`, `pipeline.env.example`)
- `third_party/` vendored dependencies (`estimator`, `MoGe`, `LSAM`)
- `third_party/Hunyuan3D-2/` cloned externally (not bundled here)
- `third_party_patches/` patches to apply to external dependencies
- `src/foho/` pipeline modules
  - `preprocess/` input generation + inpainting
  - `geometry/` MoGe + Hunyuan
  - `hand/` HaMeR
  - `alignment/` mesh alignment
  - `guidance/` optimization-in-the-loop guidance step
  - `main.py` entrypoint to the full method

## Installation
1) Clone the repo:

```bash
git clone https://github.com/aidilayce/FollowMyHold.git
cd FollowMyHold
```

2) Clone Hunyuan3D-2 into `third_party/`:

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git third_party/Hunyuan3D-2
cd third_party/Hunyuan3D-2
git checkout e664e7471642c09921d23baaeba8ebe79bd6c48b
cd ../..
```

3) Replace Hunyuan3D-2’s `pipelines.py` and HF scheduler with the FOHO patches:

```bash
cp third_party_patches/hy3dgen/shapegen/pipelines.py \
  third_party/Hunyuan3D-2/hy3dgen/shapegen/pipelines.py
cp third_party_patches/hy3dgen/shapegen/schedulers.py \
  third_party/Hunyuan3D-2/hy3dgen/shapegen/schedulers.py
```

4) Create the environment (set your conda.sh path first):

```bash
export CONDA_SH="/path/to/miniforge3/etc/profile.d/conda.sh"
bash scripts/create_env_foho.sh
```

If `CONDA_PREFIX` is not set in your shell, you can also export:
```bash
export ENV_PREFIX="/path/to/miniforge3/envs/foho"
export CUDA_HOME="/usr/local/cuda"
```

More detailed conda environment flags are at [_env_foho](src/foho/main.py).

5) Fetch bundled weights/data:

```bash
bash scripts/fetch_data.sh
```

Download hand-object detector weights from [hand_object_detector](https://github.com/ddshan/hand_object_detector) and place them here:

```text
third_party/estimator/hand_object_detector/data/pretrained_model/resnet101_caffe.pth
third_party/estimator/hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_89999.pth
```

For MANO assets, visit the [MANO website](https://mano.is.tue.mpg.de) and register to download.
Only the right hand model is required. Put `MANO_RIGHT.pkl` under:
```
third_party/estimator/hamer/_DATA/data/mano
```

6) Set required environment variables:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
export GEMINI_API_KEY="YOUR_KEY_HERE"
export HY3DGEN_MODELS="/path/to/your/ckpt/cache"
```

## Run the method
1) Edit the config for your image path: `configs/pipeline.env`. Full editable config example is at `configs/pipeline.env.example`.

2) Run the full pipeline:

```bash
PYTHONPATH=src python3 -m foho.main --config configs/pipeline.env
```

### Single-step usage
Each module exposes a `run(...)` function, but required flags differ per module. Use `-h/--help` for the full list. Example:

```bash
conda activate foho
PYTHONPATH=src python3 -m foho.preprocess.get_hunyuan_input \
  --split_path /path/to/split.csv \
  --occ_img_dir /path/to/masked_obj_imgs \
  --cropped_img_dir /path/to/cropped_hoi_imgs \
  --cropped_img_wo_bckg_dir /path/to/cropped_hoi_imgs_wo_bckg \
  --mask_dir /path/to/cropped_hand_masks \
  --original_img_dir /path/to/original_imgs
```

## Notes
- `RUN_INPAINT=0` in the config skips the inpainting step.
- All output paths are derived from `BASE_DIR` unless overridden.
- With different versions of *diffusers* and *transformers* libraries, the inpainting model might perform worse. So, it's strongly recommended to use the library versions given in `scripts/create_env_foho.sh`.
- Warnings are suppressed by default via `FOHO_SUPPRESS_WARNINGS=1`. To re-enable warnings, run `export FOHO_SUPPRESS_WARNINGS=0` in your shell before running.
- It is OK to see HaMeR warnings like `unexpected key in source state_dict: backbone.blocks.0.mlp.experts.0.weight`. They are expected and the pipeline should still work.
- You can set `FOHO_DEBUG_DIR` to enable extra debug outputs from Hy3DGen/Hunyuan3D-2 (e.g., `export FOHO_DEBUG_DIR=/tmp/foho_debug` or add it to your `configs/pipeline.env`).


## Questions
For technical questions, please open a GitHub issue in this repo. For other things, please contact [me](https://aidilayce.github.io).

## Acknowledgements
This project builds on the following projects, many thanks to the authors for open-sourcing their codes: 
[LSAM](https://github.com/luca-medeiros/lang-segment-anything),
[HaMeR](https://github.com/geopavlakos/hamer), 
[WiLoR](https://github.com/rolpotamias/WiLoR), 
[MoGe](https://github.com/microsoft/MoGe), 
[Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2).

## Citation
If you use this code in your research, please consider citing the paper:

```bibtex
@article{aytekin2025follow,
  title={Follow My Hold: Hand-object Interaction Reconstruction through Geometric Guidance},
  author={Aytekin, Ayce Idil and Rhodin, Helge and Dabral, Rishabh and Theobalt, Christian},
  year={2026},
  journal={International Conference on 3D Vision},
}
```

## License
FollowMyHold is licensed under MIT license. 
However, please note that this repo builds on [acknowledged projects](#acknowledgements), which fall under their own licenses.
