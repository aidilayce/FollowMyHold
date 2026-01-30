"""Align MANO hand mesh to Hunyuan HOI mesh."""

from __future__ import annotations

import argparse
import glob
import os

from foho.alignment.mesh_align import align_meshes_impl


def run(hamer_out_dir: str, hunyuan_mesh_dir: str, aligned_mano_dir: str) -> None:
    meshes = glob.glob(os.path.join(hamer_out_dir, "*.obj"))
    if not meshes:
        print(f"No HaMeR meshes found in {hamer_out_dir}")
        return

    for mesh_path in meshes:
        base_name = os.path.basename(mesh_path)
        i = base_name.split("_")[0]
        j = os.path.splitext(base_name)[0]
        target_mesh = os.path.join(hunyuan_mesh_dir, f"{i}_hoi_mesh.ply")
        out_path = os.path.join(aligned_mano_dir, f"{j}_aligned_mano.ply")
        align_meshes_impl(
            source_mesh_path=mesh_path,
            target_mesh_path=target_mesh,
            transform_path=None,
            transformed_mesh_path=out_path,
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
    parser.add_argument("--hamer_out_dir", required=True)
    parser.add_argument("--hunyuan_mesh_dir", required=True)
    parser.add_argument("--aligned_mano_dir", required=True)
    args = parser.parse_args()

    run(
        hamer_out_dir=args.hamer_out_dir,
        hunyuan_mesh_dir=args.hunyuan_mesh_dir,
        aligned_mano_dir=args.aligned_mano_dir,
    )


if __name__ == "__main__":
    main()
