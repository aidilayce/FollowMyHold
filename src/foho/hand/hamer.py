"""Run HaMeR on cropped HOI images without external script dependency."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torchvision.ops as ops
import pandas as pd


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def non_max_suppression(bboxes, scores, iou_threshold=0.3):
    if len(bboxes) == 0:
        return []

    boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep_indices.numpy()


def run(
    hamer_demo_dir: str,
    img_folder: str,
    out_folder: str,
    full_img_dir: str,
    checkpoint: str,
    side_view: bool = False,
    full_frame: bool = True,
    save_mesh: bool = False,
    batch_size: int = 1,
    rescale_factor: float = 2.0,
    body_detector: str = "vitdet",
    file_type: List[str] | None = None,
) -> None:
    sys.path.append(hamer_demo_dir)
    if os.environ.get("FOHO_SUPPRESS_WARNINGS", "1") == "1":
        logging.getLogger("mmcv").setLevel(logging.ERROR)
        logging.getLogger("mmengine").setLevel(logging.ERROR)
        logging.getLogger("detectron2").setLevel(logging.ERROR)

    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    from vitpose_model import ViTPoseModel

    if checkpoint == "":
        checkpoint = DEFAULT_CHECKPOINT
    
    file_type = file_type or ["*.jpg", "*.png"]

    model, model_cfg = load_hamer(checkpoint)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if body_detector == "vitdet":
        from detectron2.config import LazyConfig
        import hamer

        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif body_detector == "regnety":
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        raise ValueError(f"Unknown body_detector: {body_detector}")

    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    # save J_regressor for using in optimization-in-the-loop part 
    if not os.path.exists('./J_regressor_hamer.pt'): 
        torch.save(model.mano.J_regressor, './J_regressor_hamer.pt')

    os.makedirs(out_folder, exist_ok=True)

    img_paths = [img for end in file_type for img in Path(img_folder).glob(end)]

    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            continue

        yolo_hand_id = int(str(img_path).split("/")[-1].split("_")[-1].split(".")[0])

        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        scores = []

        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.3
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                scores.append(keyp[valid, 2].mean())

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.3
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                scores.append(keyp[valid, 2].mean())

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        scores = np.array(scores)
        is_right = np.array(is_right)

        bboxes = boxes
        left_hand_idxs = np.where(is_right == 0)[0]
        right_hand_idxs = np.where(is_right == 1)[0]

        if len(left_hand_idxs) > 0:
            keep_left = non_max_suppression(bboxes[left_hand_idxs], scores[left_hand_idxs])
            left_hand_idxs = left_hand_idxs[keep_left]

        if len(right_hand_idxs) > 0:
            keep_right = non_max_suppression(bboxes[right_hand_idxs], scores[right_hand_idxs])
            right_hand_idxs = right_hand_idxs[keep_right]

        keep_indices = np.concatenate([left_hand_idxs, right_hand_idxs])

        boxes = bboxes[keep_indices]
        right = is_right[keep_indices]

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            to_save = out.copy()
            to_save["img"] = batch["img"]
            to_save["right"] = batch["right"]
            to_save["box_center"] = batch["box_center"]
            to_save["box_size"] = batch["box_size"]

            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2 * batch["right"] - 1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            to_save["pred_cam_t_full"] = pred_cam_t_full
            np.save(os.path.join(out_folder, f"{img_path.stem}.npy"), to_save)

            batch_size_local = batch["img"].shape[0]
            for n in range(batch_size_local):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                img_fn = img_fn.split("_")[0]
                right_flag = int(batch["right"][n])

                if right_flag != yolo_hand_id:
                    continue

                white_img = (torch.ones_like(batch["img"][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                    DEFAULT_STD[:, None, None] / 255
                )
                input_patch = batch["img"][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                    DEFAULT_MEAN[:, None, None] / 255
                )
                input_patch = input_patch.permute(1, 2, 0).numpy()

                regression_img = renderer(
                    out["pred_vertices"][n].detach().cpu().numpy(),
                    out["pred_cam_t"][n].detach().cpu().numpy(),
                    batch["img"][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )

                if side_view:
                    side_img = renderer(
                        out["pred_vertices"][n].detach().cpu().numpy(),
                        out["pred_cam_t"][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                    )
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(out_folder, f"{img_fn}.png"), 255 * final_img[:, :, ::-1])

                verts = out["pred_vertices"][n].detach().cpu().numpy()
                is_right_flag = batch["right"][n].cpu().numpy()
                verts[:, 0] = (2 * is_right_flag - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_flag)

                from hamer.utils.geometry import perspective_projection

                H, W, _ = img_cv2.shape
                pred_kps_3d = out["pred_keypoints_3d"][n]
                pred_kps_3d_proj = pred_kps_3d.unsqueeze(0)
                pred_kps_3d_proj[:, :, 0] = (2 * is_right_flag - 1) * pred_kps_3d_proj[:, :, 0]
                cam_t_arr = cam_t[np.newaxis, ...]
                kps_2d_orig_img = perspective_projection(
                    points=pred_kps_3d_proj,
                    translation=torch.tensor(cam_t_arr).to(device),
                    focal_length=torch.tensor([[scaled_focal_length, scaled_focal_length]], device=pred_kps_3d.device),
                    camera_center=torch.tensor([[W / 2, H / 2]], device=pred_kps_3d.device),
                )
                kps_2d_orig_img = kps_2d_orig_img.detach().squeeze(0).cpu().numpy()

                save4guidance = {}
                save4guidance["mano_3d_kps"] = pred_kps_3d_proj
                save4guidance["mano_2d_kps"] = kps_2d_orig_img
                save4guidance["cam_t"] = cam_t_arr
                np.save(os.path.join(out_folder, f"{img_fn}_kps_for_guidance.npy"), save4guidance)

                if save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right_flag)
                    tmesh.export(os.path.join(out_folder, f"{img_fn.split('_')[0]}_hamer.obj"))

        if full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args
            )

            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

            cv2.imwrite(os.path.join(out_folder, f"{img_fn}_all.jpg"), 255 * input_img_overlay[:, :, ::-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument("--hamer_demo_dir", required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--side_view", action="store_true", default=False)
    parser.add_argument("--full_frame", action="store_true", default=True)
    parser.add_argument("--save_mesh", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--body_detector", type=str, default="vitdet", choices=["vitdet", "regnety"])
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png"])
    parser.add_argument("--full_img_dir", type=str, required=True)
    args = parser.parse_args()

    run(
        hamer_demo_dir=args.hamer_demo_dir,
        img_folder=args.img_folder,
        out_folder=args.out_folder,
        full_img_dir=args.full_img_dir,
        checkpoint=args.checkpoint,
        side_view=args.side_view,
        full_frame=args.full_frame,
        save_mesh=args.save_mesh,
        batch_size=args.batch_size,
        rescale_factor=args.rescale_factor,
        body_detector=args.body_detector,
        file_type=args.file_type,
    )


if __name__ == "__main__":
    main()
