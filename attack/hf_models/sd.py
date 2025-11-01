import torch
from diffusers import StableDiffusion3Pipeline


class StableDiffusion:
    def __init__(self, model_path):
        self.model_path = model_path
        self.sd_pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

    def generate(self, text, size_px, num_inference_steps, guidance_scale):
        return self.sd_pipe(
            text,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,  # Higher guidance scale, sticks more closely to the text prompt
            height=size_px,
            width=size_px,
        ).images[0]
