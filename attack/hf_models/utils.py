from dotenv import load_dotenv
import os
from enum import Enum
import torch


load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR")

current_dir = os.path.dirname(os.path.abspath(__file__))


def load_model(model_name):
    model_path, model_type = model_name_to_path_type(model_name)
    print(f"Loading model: {model_name}")
    if model_type == ModelType.LLM:
        from .llm import HuggingFaceLLM

        if "strongreject_judge" in model_name:
            from .strongreject import StrongRejectModel

            model = StrongRejectModel(model_name, model_path)
        else:
            model = HuggingFaceLLM(model_name, model_path)
    elif model_type == ModelType.LVLM:
        from .lvlm import HuggingFaceLVLM

        model = HuggingFaceLVLM(model_name, model_path)
    elif model_type == ModelType.CLIP:
        from .clip import HuggingFaceCLIP

        model = HuggingFaceCLIP(model_path)
    else:
        from .sd import StableDiffusion

        model = StableDiffusion(model_path)

    print(f"Loaded model: {model_name}")
    return model


class ModelType(Enum):
    LLM = "LLM"
    LVLM = "LVLM"
    CLIP = "CLIP"
    SD = "SD"


def model_name_to_path_type(name) -> tuple[str, ModelType]:
    look_up = {
        "mistral_7b": {"folder": "mistral_7b_instruct_v0.3", "type": ModelType.LLM},
        "mistral": {"folder": "mistral_small_24b_instruct_2501", "type": ModelType.LLM},
        "llava": {"folder": "llava_v1.6_mistral_7b_hf", "type": ModelType.LVLM},
        "qwen": {"folder": "qwen2.5_vl_7b_instruct", "type": ModelType.LVLM},
        "llama": {"folder": "llama_3.2_11b_vision_instruct", "type": ModelType.LVLM},
        "clip": {"folder": "clip_vit_base_patch32", "type": ModelType.CLIP},
        "stable_diffusion": {"folder": "stable_diffusion_3.5_medium", "type": ModelType.SD},
        "strongreject_judge": {"folder": "strongreject_15k_v1", "type": ModelType.LLM},
        "harmbench_judge": {"folder": "harmbench_llama_2_13b_cls", "type": ModelType.LLM},
        "llama_guard": {"folder": "llama_guard_3_11b_vision", "type": ModelType.LVLM},
    }
    path = os.path.join(current_dir, "..", MODEL_DIR, look_up[name]["folder"])
    type = look_up[name]["type"]

    return path, type
