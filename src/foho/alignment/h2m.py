"""Align Hunyuan HOI mesh to MoGe mesh and write transforms."""

from __future__ import annotations

import argparse
import glob
import os

from foho.alignment.mesh_align import align_meshes_impl


def run(hunyuan_mesh_dir: str, moge_out_dir: str, h2m_rt_dir: str) -> None:
    meshes = glob.glob(os.path.join(hunyuan_mesh_dir, "*.ply"))
    if not meshes:
        print(f"No Hunyuan HOI meshes found in {hunyuan_mesh_dir}")
        return

    for mesh_path in meshes:
        base_name = os.path.basename(mesh_path)
        i = base_name.split("_")[0]
        j = os.path.splitext(base_name)[0]
        moge_dir = os.path.join(moge_out_dir, f"{i}_cropped_hoi")
        target_mesh = os.path.join(moge_dir, "mesh.ply")
        if not os.path.isfile(target_mesh):
            # MoGe typically writes pointcloud.ply and/or mesh.glb.
            pointcloud_mesh = os.path.join(moge_dir, "pointcloud.ply")
            glb_mesh = os.path.join(moge_dir, "mesh.glb")
            if os.path.isfile(pointcloud_mesh):
                target_mesh = pointcloud_mesh
            elif os.path.isfile(glb_mesh):
                target_mesh = glb_mesh
            else:
                print(f"No MoGe mesh found for {i} in {moge_dir}. Skipping.")
                continue
        align_meshes_impl(
            source_mesh_path=mesh_path,
            target_mesh_path=target_mesh,
            transform_path=os.path.join(h2m_rt_dir, j),
            transformed_mesh_path=None,
            fixed_scale=False,
            outliers=0.2,
            test_rotations=False,
            test_reflections=False,
            on_surface=False,
            iterations_coarse=50,
            count_source_coarse=1000,
            count_target_coarse=5000,
            iterations_fine=100,
            count_source_fine=5000,
            count_target_fine=10000,
            min_scale=0.7,
            max_scale=3.0,
            plot=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hunyuan_mesh_dir", required=True)
    parser.add_argument("--moge_out_dir", required=True)
    parser.add_argument("--h2m_rt_dir", required=True)
    args = parser.parse_args()

    run(
        hunyuan_mesh_dir=args.hunyuan_mesh_dir,
        moge_out_dir=args.moge_out_dir,
        h2m_rt_dir=args.h2m_rt_dir,
    )


if __name__ == "__main__":
    main()
