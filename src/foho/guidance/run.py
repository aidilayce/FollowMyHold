"""Run FOHO guidance step."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import trimesh
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    FoVPerspectiveCameras,
    BlendParams,
)
from pytorch3d.io import IO

from foho.configs import OptimizationConfig
from foho.configs import third_party_root

_TP = third_party_root()
_HY3DGEN_ROOT = os.path.join(_TP, "Hunyuan3D-2")
sys.path.append(_TP)
sys.path.append(_HY3DGEN_ROOT)
sys.path.append(os.path.join(_HY3DGEN_ROOT, "hy3dgen"))
sys.path.append(os.path.join(_TP, "estimator"))

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen.pipelines import (
    Hunyuan3DDiTFlowMatchingPipeline_main,
    PhongNormalShader,
)
from hy3dgen.shapegen.postprocessors import (
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
)

warnings.filterwarnings(
    "ignore",
    message="Bin size was too small in the coarse rasterization phase.*",
)


def _setup_sys_path(project_root: str) -> None:
    tp = third_party_root()
    hy3dgen_root = os.path.join(tp, "Hunyuan3D-2")
    sys.path.append(tp)
    sys.path.append(os.path.join(tp, "estimator"))
    sys.path.append(hy3dgen_root)
    sys.path.append(os.path.join(hy3dgen_root, "hy3dgen"))
    sys.path.append(project_root)


def run_hunyuan_w_guid(
    cropped_obj_img_path,
    fovx,
    hamer_for_guid_path,
    aligned_mano_mesh_path,
    cropped_obj_mask_path,
    cropped_hand_mask_path,
    moge_mesh_path,
    T_h2m_path,
    hunyuan_hoi_mesh_path,
    save_path_obj,
    save_path_hand,
    config,
    device="cuda",
):
    original_img_hand_mask = cv2.imread(cropped_hand_mask_path, cv2.IMREAD_GRAYSCALE)
    img_size = original_img_hand_mask.shape
    H, W = img_size[:2]

    rotation_y_180 = torch.tensor(
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=device, dtype=torch.float32
    )
    fov = fovx
    R = rotation_y_180.unsqueeze(0)
    T = torch.zeros(1, 3, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=100.0, fov=fov)
    blend_params = BlendParams(
        sigma=torch.tensor(1e-8, dtype=torch.float32, device=device),
        gamma=torch.tensor(1e-8, dtype=torch.float32, device=device),
    )
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=1,
        bin_size=-1,
        max_faces_per_bin=None,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=PhongNormalShader(cameras=cameras, blend_params=blend_params),
    )
    silhoutte_raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,
        bin_size=None,
        max_faces_per_bin=None,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=silhoutte_raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    model_path = "tencent/Hunyuan3D-2"
    rembg = BackgroundRemover()
    seed = 2 #2025

    if "inpaint" in cropped_obj_img_path:
        images = []
        image = Image.open(cropped_obj_img_path).convert("RGB")
        if image.mode == "RGB":
            image = rembg(image)
        images.append(image)
    else:
        image = Image.open(cropped_obj_img_path).convert("RGBA")
        datas = image.getdata()
        new_data = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        image.putdata(new_data)
        images = [image]

    pipeline = Hunyuan3DDiTFlowMatchingPipeline_main.from_pretrained(model_path)
    obj_mesh, hand_mesh = pipeline(
        image=images,
        mc_algo="mc",
        generator=torch.manual_seed(seed),
        config=config,
        renderer=renderer,
        sil_renderer=silhouette_renderer,
        cropped_obj_img_path=cropped_obj_img_path,
        hamer_for_guid_path=hamer_for_guid_path,
        aligned_mano_mesh_path=aligned_mano_mesh_path,
        obj_mask_path=cropped_obj_mask_path,
        hand_mask_path=cropped_hand_mask_path,
        moge_mesh_path=moge_mesh_path,
        h2m_rt_path=T_h2m_path,
        hunyuan_hoi_mesh_path=hunyuan_hoi_mesh_path,
    )

    try:
        obj_mesh_verts, obj_mesh_faces = obj_mesh.verts_packed(), obj_mesh.faces_packed()
        obj_mesh = trimesh.Trimesh(vertices=obj_mesh_verts.cpu().numpy(), faces=obj_mesh_faces.cpu().numpy())
        obj_mesh = FloaterRemover()(obj_mesh)
        obj_mesh = DegenerateFaceRemover()(obj_mesh)
        obj_mesh = FaceReducer()(obj_mesh)
        obj_mesh.export(save_path_obj)

        IO().save_mesh(hand_mesh, save_path_hand)
    except Exception:
        print(f"Error in saving mesh for {cropped_obj_img_path}")
        return None, None

    if len(obj_mesh.vertices) == 0:
        print(f"Empty mesh for {cropped_obj_img_path}")
        return None, None

    return obj_mesh, hand_mesh


def _load_task_list(task_list_file: Optional[str], cropped_obj_img_dir: str) -> List[str]:
    if task_list_file and os.path.exists(task_list_file):
        with open(task_list_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        return chunks[array_task_id]

    return sorted(os.listdir(cropped_obj_img_dir))


def run(
    project_root: str,
    cropped_obj_img_dir: str,
    mask_dir: str,
    moge_out_dir: str,
    hunyuan_hoi_mesh_dir: str,
    hamer_out_dir: str,
    h2m_rt_dir: str,
    aligned_mano_dir: str,
    guidance_out_dir: str,
    task_list_file: Optional[str] = None,
) -> None:
    _setup_sys_path(project_root)

    config = OptimizationConfig()

    os.makedirs(guidance_out_dir, exist_ok=True)

    assigned_imgs = _load_task_list(task_list_file, cropped_obj_img_dir)

    for cropped_obj_img in tqdm(assigned_imgs):
        try:
            cropped_obj_img_path = os.path.join(cropped_obj_img_dir, cropped_obj_img)
            index = cropped_obj_img.split("_")[0]
            is_right = cropped_obj_img.split("_")[-1].split(".")[0]
            cropped_hand_mask_path = os.path.join(mask_dir, f"{index}_cropped_hand_mask.png")
            cropped_obj_mask_path = os.path.join(mask_dir, f"{index}_cropped_obj_mask.png")
            moge_mesh_path = os.path.join(moge_out_dir, f"{index}_cropped_hoi/mesh.glb")
            moge_fov_path = os.path.join(moge_out_dir, f"{index}_cropped_hoi/fov.json")
            T_h2m_path = os.path.join(h2m_rt_dir, f"{index}_hoi_mesh.npy")
            aligned_mano_mesh_path = os.path.join(aligned_mano_dir, f"{index}_hamer_aligned_mano.ply")
            hunyuan_hoi_mesh_path = os.path.join(hunyuan_hoi_mesh_dir, f"{index}_hoi_mesh.ply")
            hamer_for_guid_path = os.path.join(hamer_out_dir, f"{index}_kps_for_guidance.npy")
            save_path_obj = os.path.join(guidance_out_dir, f"{index}_obj.ply")
            save_path_hand = os.path.join(guidance_out_dir, f"{index}_hand.ply")

            if os.path.exists(save_path_obj) and os.path.exists(save_path_hand):
                print(f"{index} already exists, skipping")
                continue

            with open(moge_fov_path, "r", encoding="utf-8") as f:
                moge_dict = json.load(f)
            fovx = float(moge_dict["fov_x"])

            cropped_hand_mask = cv2.imread(cropped_hand_mask_path, cv2.IMREAD_UNCHANGED)
            cropped_obj_mask = cv2.imread(cropped_obj_mask_path, cv2.IMREAD_UNCHANGED)
            if cropped_hand_mask.max() == 0 or cropped_obj_mask.max() == 0:
                print(f"Skipping {index} due to empty mask")
                continue

            print(f"Processing {index}")
            obj_mesh, hand_mesh = run_hunyuan_w_guid(
                cropped_obj_img_path=cropped_obj_img_path,
                fovx=fovx,
                hamer_for_guid_path=hamer_for_guid_path,
                aligned_mano_mesh_path=aligned_mano_mesh_path,
                cropped_obj_mask_path=cropped_obj_mask_path,
                cropped_hand_mask_path=cropped_hand_mask_path,
                moge_mesh_path=moge_mesh_path,
                T_h2m_path=T_h2m_path,
                hunyuan_hoi_mesh_path=hunyuan_hoi_mesh_path,
                save_path_obj=save_path_obj,
                save_path_hand=save_path_hand,
                config=config,
            )
            if obj_mesh is None or hand_mesh is None:
                print(f"Error in reconstruction for {index}")
                continue
            print(f"Reconstructed object {index}")
        except Exception as e:
            print(f"Error in processing {cropped_obj_img} : {e}")
            continue

    print("Finished processing all images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hunyuan3D-2 guidance")
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--cropped_obj_img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--moge_out_dir", required=True)
    parser.add_argument("--hunyuan_hoi_mesh_dir", required=True)
    parser.add_argument("--hamer_out_dir", required=True)
    parser.add_argument("--h2m_rt_dir", required=True)
    parser.add_argument("--aligned_mano_dir", required=True)
    parser.add_argument("--guidance_out_dir", required=True)
    parser.add_argument("--task_list_file", default=None)
    args = parser.parse_args()

    run(
        project_root=args.project_root,
        cropped_obj_img_dir=args.cropped_obj_img_dir,
        mask_dir=args.mask_dir,
        moge_out_dir=args.moge_out_dir,
        hunyuan_hoi_mesh_dir=args.hunyuan_hoi_mesh_dir,
        hamer_out_dir=args.hamer_out_dir,
        h2m_rt_dir=args.h2m_rt_dir,
        aligned_mano_dir=args.aligned_mano_dir,
        guidance_out_dir=args.guidance_out_dir,
        task_list_file=args.task_list_file,
    )


if __name__ == "__main__":
    main()
