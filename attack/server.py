import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from hf_models.utils import load_model
from hf_models import HuggingFaceLLM, HuggingFaceLVLM
from typing import List
import json
from datetime import datetime
import logging
import io
import argparse
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO)


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--llm_name", type=str, default="mistral")
    config.add_argument("--llm_7b_name", type=str, default="mistral_7b")
    config.add_argument("--lvlm_name", type=str, default="llava", choices=["", "llava", "llama", "qwen"])
    config.add_argument("--load_clip", action="store_true")
    config.add_argument("--load_sd", action="store_true")
    config.add_argument("--load_sem_relevance", action="store_true")
    config.add_argument("--load_toxigen", action="store_true")
    config.add_argument("--load_llama_guard", action="store_true")
    config.add_argument("--dataset", type=str, default="")
    config.add_argument("--port", type=int, default=8000)

    return config


def main(args):
    app = FastAPI()

    # harmbench_judge_model = None
    # strong_reject_model = None
    # if "strongreject" in args.dataset:
    strong_reject_model: HuggingFaceLLM = load_model("strongreject_judge")
    # elif "harmbench" in args.dataset:
    harmbench_judge_model: HuggingFaceLLM = load_model("harmbench_judge")

    llm_7b_hf_model: HuggingFaceLLM = load_model(args.llm_7b_name)
    llm_hf_model: HuggingFaceLLM = load_model(args.llm_name)

    @app.middleware("http")
    async def log_request_time(request: Request, call_next):
        now = datetime.now().isoformat()
        logging.info(f"{request.method} {request.url} at {now}")
        response = await call_next(request)
        return response

    @app.get("/llm_gen")
    async def llm_generation(request: Request):
        data = await request.json()
        full_prompt = data.get("full_prompt")
        model_name = data.get("model_name")
        max_new_tokens = data.get("max_new_tokens")
        temperature = data.get("temperature")
        top_p = data.get("top_p")

        if model_name == args.llm_name:
            model = llm_hf_model
        elif model_name == args.llm_7b_name:
            model = llm_7b_hf_model
        elif model_name == "strongreject_judge":
            # print("RECEIVED PROMPT:", full_prompt)
            model = strong_reject_model
        elif model_name == "harmbench_judge":
            # print("RECEIVED PROMPT:", full_prompt)
            model = harmbench_judge_model
        else:
            error_msg = f"Model not found: {model_name}."
            print(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        response = model.generate(full_prompt, max_new_tokens, temperature, top_p)

        return response

    if args.lvlm_name:
        lvlm_hf_model: HuggingFaceLVLM = load_model(args.lvlm_name)
        if args.load_llama_guard:
            llama_guard: HuggingFaceLVLM = load_model("llama_guard")

        @app.post("/lvlm_gen")
        async def lvlm_generation(files: List[UploadFile] = File(default=None), metadata: str = Form(...)):
            metadata_dict = json.loads(metadata)
            images = []
            if files:
                for file in files:
                    # Read the file content (bytes)
                    contents = await file.read()

                    # Convert to PIL Image
                    try:
                        image = Image.open(BytesIO(contents))
                        images.append(image)
                    except Exception as e:
                        return {"error": f"Could not process {file.filename}: {e}"}

            model_name = metadata_dict.get("model_name", "")
            if "llama_guard" in model_name:
                model = llama_guard
            else:
                model = lvlm_hf_model

            response = model.generate(
                metadata_dict["conv"],
                images,
                metadata_dict["max_new_tokens"],
                metadata_dict["temperature"],
                metadata_dict["top_p"],
            )

            return response

    if args.load_clip:
        from hf_models import HuggingFaceCLIP

        clip_hf_model: HuggingFaceCLIP = load_model("clip")

        @app.get("/clip_gen")
        async def clip_generation(request: Request):
            data = await request.json()
            text = data.get("text")

            response = clip_hf_model.encode(text)

            return response

    if args.load_sd:
        from hf_models import StableDiffusion

        sd_model: StableDiffusion = load_model("stable_diffusion")

        @app.get("/sd_gen")
        async def sd_generation(request: Request):
            data = await request.json()
            text = data.get("text")
            size_px = data.get("size_px")
            num_inference_steps = data.get("num_inference_steps")
            guidance_scale = data.get("guidance_scale")

            img = sd_model.generate(text, size_px, num_inference_steps, guidance_scale)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            return StreamingResponse(buffer, media_type="image/png")

    if args.load_sem_relevance:
        from hf_models import SemRelvance

        sem_relevance_model: SemRelvance = SemRelvance()

        @app.post("/sem_relevance")
        async def sem_relevance_generate(request: Request):
            # Get request data
            data = await request.json()
            text1 = data.get("text1", "")
            text2 = data.get("text2", "")
            similarity = sem_relevance_model.compute_similarity(text1, text2)

            return {"similarity": similarity}

    if args.load_toxigen:
        from hf_models import Toxigen

        toxigen_model: Toxigen = Toxigen()

        @app.post("/toxigen")
        async def toxigen_generate(request: Request):
            # Get request data
            try:
                data = await request.json()
                texts = data.get("texts", [])
                texts_rebuild = []
                for text in texts:
                    if len(text) > 256:
                        text = text[:128]
                    texts_rebuild.append(text)
                result = toxigen_model.get_toxicity(texts_rebuild)
            except Exception as e:
                # make sure the length of texts less than 512
                texts_rebuild = []
                for text in texts:
                    if len(text) > 256:
                        text = text[:256]
                    texts_rebuild.append(text)
                result = toxigen_model.get_toxicity(texts_rebuild)
            return {"result": result}

        from hf_models import HuggingFaceLVLM

        llama_guard: HuggingFaceLVLM = load_model("llama_guard")

        @app.post("/llama_guard")
        async def llama_guard_generate(files: List[UploadFile] = File(default=None), metadata: str = Form(...)):
            # Get request data
            metadata_dict = json.loads(metadata)
            images = []
            if files:
                for file in files:
                    # Read the file content (bytes)
                    contents = await file.read()

                    # Convert to PIL Image
                    try:
                        image = Image.open(BytesIO(contents))
                        images.append(image)
                    except Exception as e:
                        return {"error": f"Could not process {file.filename}: {e}"}

            response = llama_guard.generate(
                metadata_dict["conv"],
                images,
                metadata_dict["max_new_tokens"],
                metadata_dict["temperature"],
                metadata_dict["top_p"],
            )

            return response

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    args = config().parse_args()

    main(args)
