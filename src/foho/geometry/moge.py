"""Run MoGe inference without external script dependency."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Optional

import warnings

from foho.configs import third_party_root

def run(
    project_root: str,
    input_path: str,
    output_path: str,
    fov_x: Optional[float] = None,
    pretrained_model_name_or_path: Optional[str] = None,
    model_version: str = "v2",
    device_name: str = "cuda",
    use_fp16: bool = False,
    resize_to: Optional[int] = None,
    resolution_level: int = 9,
    num_tokens: Optional[int] = None,
    threshold: float = 0.04,
    save_maps: bool = True,
    save_glb: bool = False,
    save_ply: bool = True,
    show: bool = False,
) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    tp = third_party_root()
    sys.path.append(tp)
    sys.path.append(os.path.join(tp, "MoGe"))
    sys.path.append(project_root)

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from moge.model import import_model_class_by_version
    from moge.utils.io import save_glb, save_ply
    from moge.utils.vis import colorize_depth, colorize_normal
    import utils3d

    device = torch.device(device_name)

    include_suffices = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
    if Path(input_path).is_dir():
        image_paths = sorted(
            itertools.chain(*(Path(input_path).rglob(f"*.{suffix}") for suffix in include_suffices))
        )
    else:
        image_paths = [Path(input_path)]

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image files found in {input_path}")

    if pretrained_model_name_or_path is None:
        default_models = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = default_models[model_version]

    model = (
        import_model_class_by_version(model_version)
        .from_pretrained(pretrained_model_name_or_path)
        .to(device)
        .eval()
    )
    if use_fp16:
        model.half()

    if not any([save_maps, save_glb, save_ply]):
        warnings.warn(
            'No output format specified. Defaults to saving all. Use "--maps", "--glb", or "--ply".'
        )
        save_maps = save_glb = save_ply = True

    for image_path in tqdm(image_paths, desc="Inference", disable=len(image_paths) <= 1):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if resize_to is not None:
            height, width = (
                min(resize_to, int(resize_to * height / width)),
                min(resize_to, int(resize_to * width / height)),
            )
            image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

        output = model.infer(
            image_tensor,
            fov_x=fov_x,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )
        points = output["points"].cpu().numpy()
        depth = output["depth"].cpu().numpy()
        mask = output["mask"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()
        normal = output["normal"].cpu().numpy() if "normal" in output else None

        save_path = Path(output_path, image_path.relative_to(input_path).parent, image_path.stem)
        save_path = Path(str(save_path).split("hoi")[0] + "hoi")
        save_path.mkdir(exist_ok=True, parents=True)

        if save_maps:
            cv2.imwrite(str(save_path / "image.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / "depth_vis.png"), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / "depth.exr"), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path / "mask.png"), (mask * 255).astype(np.uint8))
            cv2.imwrite(
                str(save_path / "points.exr"),
                cv2.cvtColor(points, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
            )
            if normal is not None:
                cv2.imwrite(str(save_path / "normal.png"), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
            fov_x_out, fov_y_out = utils3d.numpy.intrinsics_to_fov(intrinsics)
            with open(save_path / "fov.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fov_x": round(float(np.rad2deg(fov_x_out)), 2),
                        "fov_y": round(float(np.rad2deg(fov_y_out)), 2),
                    },
                    f,
                )

        if save_glb or save_ply or show:
            mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold)
            if normal is None:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=mask_cleaned,
                    tri=True,
                )
                vertex_normals = None
            else:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                    points,
                    image.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=width, height=height),
                    normal,
                    mask=mask_cleaned,
                    tri=True,
                )
            vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
            if normal is not None:
                vertex_normals = vertex_normals * [1, -1, -1]

        if save_glb:
            save_glb(save_path / "mesh.glb", vertices, faces, vertex_uvs, image, vertex_normals)

        if save_ply:
            save_ply(save_path / "pointcloud.ply", vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

        if show:
            import trimesh

            trimesh.Trimesh(
                vertices=vertices,
                vertex_colors=vertex_colors,
                vertex_normals=vertex_normals,
                faces=faces,
                process=False,
            ).show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fov_x", type=float, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--num_tokens", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--glb", action="store_true")
    parser.add_argument("--ply", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run(
        project_root=args.project_root,
        input_path=args.input,
        output_path=args.output,
        fov_x=args.fov_x,
        pretrained_model_name_or_path=args.pretrained,
        model_version=args.version,
        device_name=args.device,
        use_fp16=args.fp16,
        resize_to=args.resize,
        resolution_level=args.resolution_level,
        num_tokens=args.num_tokens,
        threshold=args.threshold,
        save_maps=args.maps or (not args.glb and not args.ply and not args.show),
        save_glb=args.glb,
        save_ply=args.ply or (not args.glb and not args.maps and not args.show),
        show=args.show,
    )


if __name__ == "__main__":
    main()
