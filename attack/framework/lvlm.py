from .agent import Agent
from PIL import Image
from io import BytesIO
import json
from typing import List


class LVLM(Agent):
    ENDPOINT = "lvlm_gen"

    def __init__(self, model_name: str, max_new_tokens: int, temperature: float, top_p: float):
        super().__init__(model_name, max_new_tokens, temperature, top_p)

    @staticmethod
    def pil_to_file(image: Image.Image, filename: str, format="JPEG"):
        buf = BytesIO()
        image.save(buf, format=format)
        buf.seek(0)  # Go back to start
        return (filename, buf, f"image/{format.lower()}")

    def get_generation(self, conv: List, images: List):
        payload = dict(
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        payload["conv"] = conv
        files = [("files", self.pil_to_file(image, f"image{i+1}.jpg")) for i, image in enumerate(images)]

        data = {"metadata": json.dumps(payload)}

        response = self.request_to_server(self.ENDPOINT, use_GET=False, files=files, data=data)

        return response
