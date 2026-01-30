"""Run Hunyuan HOI mesh generation."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from foho.configs import third_party_root

def _setup_sys_path(project_root: str) -> None:
    tp = third_party_root()
    hy3dgen_root = os.path.join(tp, "Hunyuan3D-2")
    sys.path.append(tp)
    sys.path.append(os.path.join(tp, "estimator"))
    sys.path.append(hy3dgen_root)
    sys.path.append(os.path.join(hy3dgen_root, "hy3dgen"))
    sys.path.append(project_root)


def _collect_images(image_dir: str) -> List[str]:
    return [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def _hunyuan_minimal_demo(image_list: List[str], save_path: str) -> None:
    from PIL import Image
    import torch

    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FaceReducer,
        FloaterRemover,
        DegenerateFaceRemover,
    )

    model_path = "tencent/Hunyuan3D-2"

    images = []
    img_ids = []
    for image_path in image_list:
        img_id = os.path.basename(image_path).split("_")[0]

        if os.path.exists(f"{save_path}/{img_id}_hoi_mesh.ply"):
            print(f"{img_id} already exists, skipping")
            continue

        image = Image.open(image_path).convert("RGBA")
        datas = image.getdata()
        new_data = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        image.putdata(new_data)
        images.append(image)
        img_ids.append(img_id)

    if not images:
        print("No images to process.")
        return

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    batch_size = 5
    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]
    batch_ids = [img_ids[i : i + batch_size] for i in range(0, len(img_ids), batch_size)]

    for batch, batch_id in zip(batches, batch_ids):
        print(f"Processing batch {batch_id}")

        real_batch = []
        real_ids = []
        for bid, img in zip(batch_id, batch):
            if os.path.isfile(f"{save_path}/{bid}_hoi_mesh.ply"):
                print(f"{bid} already exists, skipping")
                continue
            real_batch.append(img)
            real_ids.append(bid)

        if not real_batch:
            continue

        meshes = pipeline(
            image=real_batch,
            num_inference_steps=30,
            mc_algo="mc",
            generator=torch.manual_seed(2025),
            multi_view=True,
        )
        for i, mesh in zip(real_ids, meshes):
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh)
            mesh.export(f"{save_path}/{i}_hoi_mesh.ply")
            print(f"Mesh saved with name {save_path}/{i}_hoi_mesh.ply")


def run(project_root: str, image_dir: str, save_dir: str) -> None:
    _setup_sys_path(project_root)
    image_list = _collect_images(image_dir)
    _hunyuan_minimal_demo(image_list, save_dir)
    print("Saved Hunyuan HOI meshes to", save_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    run(project_root=args.project_root, image_dir=args.image_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
