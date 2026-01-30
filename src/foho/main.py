"""FOHO pipeline entrypoint."""

from __future__ import annotations

import argparse
import os
import shlex
import warnings
from typing import Dict

from foho.configs import PipelineConfig, load_config
from foho.utils.runner import run_in_conda


def _foho_src() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _env_with_paths(cfg: PipelineConfig) -> Dict[str, str]:
    env = {
        "PYTHONPATH": f"{_foho_src()}:{os.environ.get('PYTHONPATH', '')}",
        "FOHO_PROJECT_ROOT": cfg.project_root,
    }
    if cfg.suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
        env["FOHO_SUPPRESS_WARNINGS"] = "1"
    else:
        env["FOHO_SUPPRESS_WARNINGS"] = "0"
    if cfg.gemini_api_key:
        env["GEMINI_API_KEY"] = cfg.gemini_api_key
    if cfg.hf_token:
        env["HF_TOKEN"] = cfg.hf_token
    if cfg.hy3dgen_models:
        env["HY3DGEN_MODELS"] = cfg.hy3dgen_models
    return env


def _env_foho(cfg: PipelineConfig) -> Dict[str, str]:
    env = _env_with_paths(cfg)
    cuda_home = cfg.cuda_home or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    env_prefix = cfg.env_prefix or os.environ.get("CONDA_PREFIX", "")
    nvidia_root = os.path.join(env_prefix, "lib", "python3.10", "site-packages", "nvidia") if env_prefix else ""
    nvjitlink_lib = os.path.join(nvidia_root, "nvjitlink", "lib") if nvidia_root else ""
    cusparse_lib = os.path.join(nvidia_root, "cusparse", "lib") if nvidia_root else ""
    ld_parts = [
        p
        for p in [
            nvjitlink_lib,
            cusparse_lib,
            f"{cuda_home}/lib64",
            os.environ.get("LD_LIBRARY_PATH", ""),
        ]
        if p
    ]
    env.update(
        {
            "CUDA_HOME": cuda_home,
            "PATH": f"{cuda_home}/bin:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": ":".join(ld_parts),
            "CC": os.environ.get("CC", "/usr/bin/gcc"),
            "CXX": os.environ.get("CXX", "/usr/bin/g++"),
            "ORT_DISABLE_CPU_AFFINITY": "1", # Suppress onnxruntime CPU affinity errors on multi-socket servers.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ORT_LOG_SEVERITY_LEVEL": "3", # Reduce onnxruntime log verbosity.
        }
    )
    return env


def _mkdirs(paths) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _cmd(module: str, args: Dict[str, str]) -> str:
    parts = ["python3", "-m", module]
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
            continue
        parts.append(f"--{k}")
        parts.append(str(v))
    return " ".join(shlex.quote(p) for p in parts)


def run_pipeline(cfg: PipelineConfig) -> None:
    paths = {
        "original_img_dir": cfg.original_img_dir,
        "masked_obj_path": cfg.masked_obj_path,
        "cropped_hoi_path": cfg.cropped_hoi_path,
        "cropped_hoi_wo_bckg_path": cfg.cropped_hoi_wo_bckg_path,
        "cropped_inpainted_obj": cfg.cropped_inpainted_obj,
        "mask_dir_path": cfg.mask_dir_path,
        "hunyuan_hoi_mesh_path": cfg.hunyuan_hoi_mesh_path,
        "hamer_out_path": cfg.hamer_out_path,
        "h2m_rt_path": cfg.h2m_rt_path,
        "aligned_mano_path": cfg.aligned_mano_path,
        "gemini_responses": os.path.join(cfg.base_dir, "gemini_responses.csv"),
    }

    _mkdirs(
        [
            paths["original_img_dir"],
            paths["masked_obj_path"],
            paths["cropped_hoi_path"],
            paths["cropped_hoi_wo_bckg_path"],
            paths["cropped_inpainted_obj"],
            paths["mask_dir_path"],
            cfg.moge_out_path,
            paths["hunyuan_hoi_mesh_path"],
            paths["hamer_out_path"],
            paths["h2m_rt_path"],
            paths["aligned_mano_path"],
            cfg.guidance_out_path,
        ]
    )
    env_foho = _env_foho(cfg)
    env_name = cfg.env_name

    gemini_csv = cfg.gemini_responses or paths["gemini_responses"]
    if not cfg.gemini_responses:
        run_in_conda(
            cfg.conda_sh,
            env_name,
            _cmd(
                "foho.preprocess.gemini_objname",
                {
                    "image_path": cfg.image_path,
                    "split_path": cfg.split_path,
                    "out_csv": gemini_csv,
                },
            ),
            cwd=cfg.project_root,
            extra_env=env_foho,
        )

    run_in_conda(
        cfg.conda_sh,
        env_name,
        _cmd(
            "foho.preprocess.get_hunyuan_input",
            {
                "split_path": cfg.split_path,
                "image_path": cfg.image_path,
                "occ_img_dir": paths["masked_obj_path"],
                "cropped_img_dir": paths["cropped_hoi_path"],
                "cropped_img_wo_bckg_dir": paths["cropped_hoi_wo_bckg_path"],
                "mask_dir": paths["mask_dir_path"],
                "original_img_dir": paths["original_img_dir"],
                "gemini_responses": gemini_csv,
                "project_root": cfg.project_root,
            },
        ),
        cwd=cfg.project_root,
        extra_env=env_foho,
    )

    if cfg.run_inpaint:
        run_in_conda(
            cfg.conda_sh,
            env_name,
            _cmd(
                "foho.preprocess.inpaint",
                {
                    "save_dir": paths["cropped_inpainted_obj"],
                    "cropped_img_dir": paths["cropped_hoi_path"],
                    "gemini_responses": gemini_csv,
                },
            ),
            cwd=cfg.project_root,
            extra_env=env_foho,
        )

    run_in_conda(
        cfg.conda_sh,
        env_name,
        _cmd(
            "foho.geometry.moge",
            {
                "project_root": cfg.project_root,
                "input": paths["cropped_hoi_wo_bckg_path"],
                "output": cfg.moge_out_path,
            },
        ),
        cwd=cfg.project_root,
        extra_env=env_foho,
    )

    run_in_conda(
        cfg.conda_sh,
        env_name,
        _cmd(
            "foho.geometry.hunyuan",
            {
                "project_root": cfg.project_root,
                "image_dir": paths["cropped_hoi_wo_bckg_path"],
                "save_dir": paths["hunyuan_hoi_mesh_path"],
            },
        ),
        cwd=cfg.project_root,
        extra_env=env_foho,
    )

    run_in_conda(
        cfg.conda_sh,
        env_name,
        _cmd(
            "foho.hand.hamer",
            {
                "hamer_demo_dir": cfg.hamer_demo_dir,
                "img_folder": paths["cropped_hoi_path"],
                "out_folder": paths["hamer_out_path"],
                "full_img_dir": paths["original_img_dir"],
                "save_mesh": True,
            },
        ),
        cwd=cfg.hamer_demo_dir,
        extra_env=env_foho,
    )

    run_in_conda(
        cfg.conda_sh,
        env_name,
        _cmd(
            "foho.alignment.h2m",
            {
                "hunyuan_mesh_dir": paths["hunyuan_hoi_mesh_path"],
                "moge_out_dir": cfg.moge_out_path,
                "h2m_rt_dir": paths["h2m_rt_path"],
            },
        ),
        cwd=cfg.project_root,
        extra_env=env_foho,
    )

    run_in_conda(
        cfg.conda_sh,
        env_name,
        _cmd(
            "foho.alignment.mano",
            {
                "hamer_out_dir": paths["hamer_out_path"],
                "hunyuan_mesh_dir": paths["hunyuan_hoi_mesh_path"],
                "aligned_mano_dir": paths["aligned_mano_path"],
            },
        ),
        cwd=cfg.project_root,
        extra_env=env_foho,
    )

    run_in_conda(
        cfg.conda_sh,
        env_name,
        _cmd(
            "foho.guidance.run",
            {
                "project_root": cfg.project_root,
                "cropped_obj_img_dir": paths["cropped_inpainted_obj"],
                "mask_dir": paths["mask_dir_path"],
                "moge_out_dir": cfg.moge_out_path,
                "hunyuan_hoi_mesh_dir": paths["hunyuan_hoi_mesh_path"],
                "hamer_out_dir": paths["hamer_out_path"],
                "h2m_rt_dir": paths["h2m_rt_path"],
                "aligned_mano_dir": paths["aligned_mano_path"],
                "guidance_out_dir": cfg.guidance_out_path,
            },
        ),
        cwd=cfg.project_root,
        extra_env=env_foho,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
