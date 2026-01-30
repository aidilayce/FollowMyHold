"""Config loader for FOHO pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from foho.configs.paths import third_party_root


@dataclass(frozen=True)
class PipelineConfig:
    project_root: str
    hamer_demo_dir: str
    conda_sh: str
    env_name: str
    env_prefix: str | None
    cuda_home: str | None
    gemini_api_key: str | None
    hf_token: str | None
    hy3dgen_models: str | None
    split_path: str | None
    image_path: str | None
    base_dir: str
    original_img_dir: str
    masked_obj_path: str
    cropped_hoi_path: str
    cropped_hoi_wo_bckg_path: str
    cropped_inpainted_obj: str
    mask_dir_path: str
    moge_out_path: str
    hunyuan_hoi_mesh_path: str
    hamer_out_path: str
    h2m_rt_path: str
    aligned_mano_path: str
    guidance_out_path: str
    gemini_responses: str | None
    run_inpaint: bool
    suppress_warnings: bool


_DEF_KEYS = {
    "CONDA_SH": os.environ.get("CONDA_SH", ""),
    "ENV_NAME": "foho",
    "ENV_PREFIX": os.environ.get("CONDA_PREFIX", ""),
    "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
}


def _parse_env_file(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"").strip("'")
            data[key] = val
    return data


def load_config(path: str) -> PipelineConfig:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing config: {path}")

    env = _parse_env_file(path)
    env = {**_DEF_KEYS, **env}

    project_root = env.get("PROJECT_ROOT")
    split_path = env.get("SPLIT_PATH")
    base_dir = env.get("BASE_DIR")
    if not project_root or not base_dir:
        raise ValueError("PROJECT_ROOT and BASE_DIR are required in config")
    tp_root = third_party_root()

    def _p(key: str, default: str) -> str:
        return env.get(key, default)

    original_img_dir = _p("ORIGINAL_IMG_DIR", f"{base_dir}/original_imgs")
    masked_obj_path = _p("MASKED_OBJ_PATH", f"{base_dir}/masked_obj_imgs")
    cropped_hoi_path = _p("CROPPED_HOI_PATH", f"{base_dir}/cropped_hoi_imgs")
    cropped_hoi_wo_bckg_path = _p(
        "CROPPED_HOI_WO_BCKG_PATH", f"{base_dir}/cropped_hoi_imgs_wo_bckg"
    )
    cropped_inpainted_obj = _p("CROPPED_INPAINTED_OBJ", f"{base_dir}/ours_inpaint")
    mask_dir_path = _p("MASK_DIR_PATH", f"{base_dir}/cropped_hand_masks")
    moge_out_path = _p("MOGE_OUT_PATH", f"{base_dir}/moge_out")
    hunyuan_hoi_mesh_path = _p("HUNYUAN_HOI_MESH_PATH", f"{base_dir}/hunyuan_hoi_out")
    hamer_out_path = _p("HAMER_OUT_PATH", f"{base_dir}/hamer_out")
    h2m_rt_path = _p("H2M_RT_PATH", f"{base_dir}/h2m_transformations")
    aligned_mano_path = _p("ALIGNED_MANO_PATH", f"{base_dir}/aligned_mano")
    guidance_out_path = _p("GUIDANCE_OUT_PATH", f"{base_dir}/guidance_out")

    gemini_responses = env.get("GEMINI_RESPONSES") or None
    gemini_api_key = env.get("GEMINI_API_KEY") or None
    hf_token = env.get("HF_TOKEN") or None
    hy3dgen_models = env.get("HY3DGEN_MODELS") or None
    image_path = env.get("IMAGE_PATH") or None
    if not split_path and not image_path:
        raise ValueError("Set either SPLIT_PATH or IMAGE_PATH in config")

    run_inpaint = env.get("RUN_INPAINT", "1") == "1"
    suppress_warnings = env.get("FOHO_SUPPRESS_WARNINGS", "1") == "1"

    # Back-compat: fall back to older env keys if ENV_NAME is not set.
    env_name = env.get("ENV_NAME") or env.get("ENV_DSINE") or "foho"
    env_prefix = env.get("ENV_PREFIX") or None
    cuda_home = env.get("CUDA_HOME") or None
    conda_sh = env.get("CONDA_SH") or None
    if not conda_sh:
        raise ValueError("CONDA_SH is required in config or environment")

    return PipelineConfig(
        project_root=project_root,
        hamer_demo_dir=env.get("HAMER_DEMO_DIR", os.path.join(tp_root, "estimator", "hamer")),
        conda_sh=conda_sh,
        env_name=env_name,
        env_prefix=env_prefix,
        cuda_home=cuda_home,
        gemini_api_key=gemini_api_key,
        hf_token=hf_token,
        hy3dgen_models=hy3dgen_models,
        split_path=split_path,
        image_path=image_path,
        base_dir=base_dir,
        original_img_dir=original_img_dir,
        masked_obj_path=masked_obj_path,
        cropped_hoi_path=cropped_hoi_path,
        cropped_hoi_wo_bckg_path=cropped_hoi_wo_bckg_path,
        cropped_inpainted_obj=cropped_inpainted_obj,
        mask_dir_path=mask_dir_path,
        moge_out_path=moge_out_path,
        hunyuan_hoi_mesh_path=hunyuan_hoi_mesh_path,
        hamer_out_path=hamer_out_path,
        h2m_rt_path=h2m_rt_path,
        aligned_mano_path=aligned_mano_path,
        guidance_out_path=guidance_out_path,
        gemini_responses=gemini_responses,
        run_inpaint=run_inpaint,
        suppress_warnings=suppress_warnings,
    )
