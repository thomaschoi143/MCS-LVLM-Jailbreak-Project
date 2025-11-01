from .lvlm import LVLM


class LlamaGuard(LVLM):
    def __init__(self, temperature: float = 0, top_p: float = 0):
        super().__init__("llama_guard", 20, temperature, top_p)

    # Generation
    def is_safe(self, text, image):
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]

        images = []
        if image:
            conv[-1]["content"].append(
                {
                    "type": "image",
                },
            )
            images.append(image)

        response = self.get_generation(conv, images=images).strip(' "')

        return not "unsafe" in response
