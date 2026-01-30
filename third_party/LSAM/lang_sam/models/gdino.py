import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import inspect

from LSAM.lang_sam.models.utils import DEVICE

class GDINO:
    def build_model(self, ckpt_path: str | None = None, device=DEVICE):
        model_id = "IDEA-Research/grounding-dino-base" if ckpt_path is None else ckpt_path
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            device
        )

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        for i, prompt in enumerate(texts_prompt):
            if prompt[-1] != ".":
                texts_prompt[i] += "."
        inputs = self.processor(images=images_pil, text=texts_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [k.size[::-1] for k in images_pil]
        # Handle transformers API changes by only passing supported kwargs.
        kwargs = {}
        sig = inspect.signature(self.processor.post_process_grounded_object_detection)
        params = sig.parameters
        if "box_threshold" in params:
            kwargs["box_threshold"] = box_threshold
        elif "threshold" in params:
            kwargs["threshold"] = box_threshold
        if "text_threshold" in params:
            kwargs["text_threshold"] = text_threshold
        if "target_sizes" in params:
            kwargs["target_sizes"] = target_sizes
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            **kwargs,
        )
        return results


if __name__ == "__main__":
    gdino = GDINO()
    gdino.build_model()
    out = gdino.predict(
        [Image.open("./assets/car.jpeg"), Image.open("./assets/car.jpeg")],
        ["wheel", "wheel"],
        0.3,
        0.25,
    )
    print(out)
