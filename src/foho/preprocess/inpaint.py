"""
Script to get inpainted object-only images using FLUX.1 Kontext model guided by Gemini responses.
"""

from __future__ import annotations

import argparse
import os

# os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image, logging as diffusers_logging
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

# diffusers_logging.set_verbosity_error()
# transformers_logging.set_verbosity_error()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def run(save_dir: str, cropped_img_dir: str, gemini_responses: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    obj_name_df = pd.read_csv(gemini_responses)
    if "image_id" not in obj_name_df.columns or "response" not in obj_name_df.columns:
        raise ValueError("gemini_responses must include image_id and response columns")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pipe = FluxKontextPipeline.from_pretrained(
        "HighCWu/FLUX.1-Kontext-dev-bnb-hqq-4bit", torch_dtype=dtype
    )
    pipe.to("cuda")

    sorted_imgs = sorted(os.listdir(cropped_img_dir), key=lambda x: int(x.split("_")[0]))
    for cropped_img_name in tqdm(sorted_imgs):
        index = cropped_img_name.split("_")[0]
        response = obj_name_df.loc[
            obj_name_df["image_id"].astype(str) == str(index), "response"
        ].values
        if len(response) == 0:
            print(f"No Gemini response for {index}. Available ids: {obj_name_df['image_id'].head().tolist()}...")
            response_text = "manipulated object"
        else:
            response_text = response[0]
        
        input_image_path = os.path.join(cropped_img_dir, cropped_img_name)
        save_path = os.path.join(save_dir, f"{index}_inpainted_object.png")

        if os.path.basename(save_path) in os.listdir(save_dir):
            print("Already processed. Skipping.")
            continue

        input_image = load_image(input_image_path)
        prompt =  f"Remove hands but keep the {response_text}."
        negative_prompt = "hands, fingers, arms, person, human, skin, limb"
        image = pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=2.5,
            num_inference_steps=28,
            generator=torch.Generator("cuda").manual_seed(2),
            height=input_image.size[1],
            width=input_image.size[0],
        ).images[0]

        image.save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--cropped_img_dir", required=True)
    parser.add_argument("--gemini_responses", required=True)
    args = parser.parse_args()

    run(
        save_dir=args.save_dir,
        cropped_img_dir=args.cropped_img_dir,
        gemini_responses=args.gemini_responses,
    )


if __name__ == "__main__":
    main()
