"""Generate Hunyuan input assets (masks, cropped HOI images)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from foho.configs import third_party_root

def _setup_sys_path(project_root: str) -> None:
    tp = third_party_root()
    sys.path.insert(0, os.path.dirname(project_root))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(tp, "MoGe"))
    sys.path.insert(0, os.path.join(tp, "models"))
    sys.path.insert(0, os.path.join(tp, "estimator"))
    sys.path.insert(0, tp)


def run(
    split_path: str | None,
    image_path: str | None,
    occ_img_dir: str,
    cropped_img_dir: str,
    cropped_img_wo_bckg_dir: str,
    mask_dir: str,
    original_img_dir: str,
    gemini_responses: Optional[str] = None,
    project_root: Optional[str] = None,
) -> None:
    project_root = project_root or os.environ.get(
        "FOHO_PROJECT_ROOT", third_party_root()
    )
    _setup_sys_path(project_root)

    import torch
    from PIL import Image
    import pandas as pd
    from tqdm import tqdm
    from ultralytics import YOLO

    from estimator.hand_object_detector.hoi_detector import hand_object_detector
    from LSAM.lang_sam import LangSAM
    from foho.preprocess import segment_hoi_sam2 as hoiSAM2

    if image_path:
        img_id = os.path.basename(image_path).split("_")[0].split(".")[0]
        df = pd.DataFrame([{"img_id": str(img_id), "img_path": image_path}])
    else:
        if not split_path:
            raise ValueError("Either split_path or image_path must be provided.")
        df = pd.read_csv(split_path)
        df["img_id"] = df["img_id"].astype(str)

    obj_name_df = None
    if gemini_responses:
        obj_name_df = pd.read_csv(gemini_responses)

    wilor_ckpt = os.environ.get(
        "WILOR_CKPT",
        os.path.join(third_party_root(), "estimator", "wilor_ckpt", "detector.pt"),
    )
    hand_detector = YOLO(wilor_ckpt)
    sam_model = LangSAM(sam_type="sam2.1_hiera_large")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row["img_id"]
        source_image = row["img_path"]
        print(f"Processing {source_image} with id {img_id}.")

        if f"{img_id}_cropped_obj_mask.png" in os.listdir(mask_dir):
            print(f"Already processed {img_id}. Skipping.")
            continue

        try:
            if obj_name_df is not None:
                object_name = obj_name_df.loc[
                    obj_name_df["image_path"] == source_image, "response"
                ].values
                object_name = object_name[0] if len(object_name) else None
                print(f"Object name for {img_id}: {object_name}")
                out = hoiSAM2.get_hoi_mask(
                    source_image, hand_detector, sam_model, hand_object_detector, object_name
                )
            else:
                out = hoiSAM2.get_hoi_mask(
                    source_image, hand_detector, sam_model, hand_object_detector
                )

            if out is None:
                print(f"No masks for {img_id}. Skipping.")
                continue
            occ_img, cropped_obj_mask, cropped_hand_mask, cropped_img_wo_bckg, crop_img_hoi, is_right = out

            occ_img_path = os.path.join(occ_img_dir, f"{img_id}_occ_obj.png")
            crop_img_hoi_path = os.path.join(cropped_img_dir, f"{img_id}_cropped_hoi_{is_right}.png")
            cropped_img_wo_bckg_path = os.path.join(
                cropped_img_wo_bckg_dir, f"{img_id}_cropped_hoi_wo_bckg_{is_right}.png"
            )
            cropped_obj_mask_path = os.path.join(mask_dir, f"{img_id}_cropped_obj_mask.png")
            cropped_hand_mask_path = os.path.join(mask_dir, f"{img_id}_cropped_hand_mask.png")

            occ_img = Image.fromarray(occ_img, mode="RGB")
            crop_img_hoi = Image.fromarray(crop_img_hoi, mode="RGB")
            cropped_img_wo_bckg = Image.fromarray(cropped_img_wo_bckg, mode="RGB")
            cropped_obj_mask = Image.fromarray(cropped_obj_mask * 255, mode="L")
            cropped_hand_mask = Image.fromarray(cropped_hand_mask * 255, mode="L")

            occ_img.save(occ_img_path)
            crop_img_hoi.save(crop_img_hoi_path)
            cropped_img_wo_bckg.save(cropped_img_wo_bckg_path)
            cropped_obj_mask.save(cropped_obj_mask_path)
            cropped_hand_mask.save(cropped_hand_mask_path)

            ori_img = Image.open(source_image)
            ori_img.save(os.path.join(original_img_dir, f"{img_id}_full_image_{is_right}.png"))

            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM on image {img_id}. Clearing cache and stopping.")
                return
            print(f"Runtime error on image {img_id}: {e}. Skipping.")
        except Exception as e:
            print(f"Error processing image {img_id} ({source_image}): {e}. Skipping.")
            continue

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--occ_img_dir", required=True)
    parser.add_argument("--cropped_img_dir", required=True)
    parser.add_argument("--cropped_img_wo_bckg_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--original_img_dir", required=True)
    parser.add_argument("--gemini_responses", default=None)
    parser.add_argument("--project_root", default=None)
    args = parser.parse_args()

    run(
        split_path=args.split_path,
        image_path=args.image_path,
        occ_img_dir=args.occ_img_dir,
        cropped_img_dir=args.cropped_img_dir,
        cropped_img_wo_bckg_dir=args.cropped_img_wo_bckg_dir,
        mask_dir=args.mask_dir,
        original_img_dir=args.original_img_dir,
        gemini_responses=args.gemini_responses,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
