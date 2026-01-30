"""Generate object-name prompts with Gemini for each image."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import csv
import os
from typing import List, Optional

import google.generativeai as genai
from PIL import Image

import os

QUESTION = "What is the person holding in their hand? Describe only the object shortly without mentioning the person."

def _image_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return stem.split("_")[0]


def _collect_images(image_dir: Optional[str], image_path: Optional[str]) -> List[str]:
    if image_path:
        return [image_path]
    if not image_dir:
        return []
    return [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def run(
    image_dir: Optional[str],
    image_path: Optional[str],
    split_path: Optional[str],
    out_csv: str,
) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel("gemini-2.5-flash-lite")

    if split_path:
        import pandas as pd

        df = pd.read_csv(split_path)
        images = df["img_path"].tolist()
    else:
        images = _collect_images(image_dir, image_path)
    if not images:
        raise ValueError("No images found for Gemini prompt generation")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv) or os.stat(out_csv).st_size == 0

    with open(out_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_id", "image_path", "response"])

        for img_path in images:
            image_id = _image_id_from_path(img_path)
            prompt_image = Image.open(img_path).convert("RGB")
            response = gemini.generate_content([QUESTION, prompt_image])
            if response.candidates is None or len(response.candidates) == 0:
                print(f"No Gemini response for {image_id}, skipping.")
                continue
            writer.writerow([image_id, img_path, response.text.strip()])
            print(f"Gemini response for {image_id}: {response.text.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    run(
        image_dir=args.image_dir,
        image_path=args.image_path,
        split_path=args.split_path,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
