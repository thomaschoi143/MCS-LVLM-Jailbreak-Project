# vllm
# pillow
# diffusers
# pandas
# import torch
from PIL import Image
import os

# from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
# from transformers import AutoModel, CLIPImageProcessor
# from transformers import pipeline
# import transformers

import argparse

# from tqdm import tqdm
# import numpy as np
import pandas as pd
import logging
import requests
import json
from io import BytesIO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "attack")))
from framework import HarmbenchJudge, StrongRejectJudge

PORT = None


def pil_to_file(image: Image.Image, filename: str, format="png"):
    buf = BytesIO()
    image.save(buf, format=format)
    buf.seek(0)  # Go back to start
    return (filename, buf, f"image/{format.lower()}")


def get_victim_generation(query: str, image):
    conv = []
    conv.append({"role": "user", "content": [{"type": "text", "text": query}, {"type": "image"}]})

    payload = dict(model_name="lvlm", max_new_tokens=300, temperature=0, top_p=0)
    try:
        endpoint = "lvlm_gen"
        payload["conv"] = conv
        files = [("files", pil_to_file(image, f"image.png")) for i, image in enumerate([image])]

        data = {"metadata": json.dumps(payload)}

        response = requests.post(f"http://localhost:{PORT}/{endpoint}", files=files, data=data)
        response_text = response.text.strip('"')

        return response_text
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
    except requests.exceptions.Timeout:
        print("Error: Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError:
        print("Error: Response is not valid JSON.")
    except Exception as e:
        print(e)
    return None


def attack(text_prompt, image_path, task, judge):
    img = Image.open(image_path)
    response = get_victim_generation(text_prompt, img)
    judge_res = judge.eval(task, response)
    return judge_res, response


def main(args, result_dir):
    # Open image folder
    data_path = args.data_path
    dataset = args.dataset
    data_file = args.data_file

    image_root = args.image_root

    query_df = pd.read_csv(f"{data_path}/{dataset}/{data_file}.csv")
    batch_query_text = query_df["jailbreak_query"]
    batch_image_path = [f"{image_root}/{image_file}" for image_file in query_df["jailbreak_query_image"]]
    tasks = query_df["redteam_query"]
    results = []
    if "harmbench" in args.dataset:
        judge = HarmbenchJudge(args.port)
    else:
        judge = StrongRejectJudge(args.port)

    for task_i, (text_prompt, image_path, task) in enumerate(zip(batch_query_text, batch_image_path, tasks)):
        result_logger.info(f"Task {task_i}. {task}")
        for attempt_i in range(args.num_attempts):
            judge_res, last_response = attack(text_prompt, image_path, task, judge)
            is_success = judge_res if isinstance(judge_res, bool) else False
            judge_score = judge_res if isinstance(judge_res, float) else 0
            results.append(
                {
                    "task_i": task_i,
                    "task": task,
                    "attempt_i": attempt_i,
                    "is_success": is_success,
                    "judge_score": judge_score,
                    "last_response": last_response,
                }
            )
            result_logger.info(
                f"\tAttempt {attempt_i+1}: is_success: {is_success}, Score: {judge_score} - Last response: {last_response}"
            )

    results_df = pd.DataFrame(results)
    binary_asr = results_df.groupby("task_i")["is_success"].any().mean()
    average_asr = results_df["is_success"].mean()
    average_score = results_df["judge_score"].mean()
    result_logger.info(f"Binary ASR: {binary_asr}; Average ASR: {average_asr}; Average Score: {average_score}")

    stats_path = os.path.join(result_dir, f"stats.csv")
    results_df.to_csv(stats_path, index=False)
    result_logger.info(f"My results saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="llava1.6", choices=["llava1.6", "llama", "qwen2.5"])
    parser.add_argument("--data_file", type=str, default="vrp")
    parser.add_argument("--dataset", type=str, default="harmbench")
    parser.add_argument("--data_path", type=str, default="./prompt")
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--image_root", type=str, default="./image")

    # My argument
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--num_attempts", type=int, default=1)

    args = parser.parse_args()

    PORT = args.port

    output_dir = os.path.join(os.getcwd(), args.save_path)
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, args.experiment_name)
    os.makedirs(result_dir, exist_ok=True)

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    result_log_file = os.path.join(result_dir, f"results.log")
    result_logger = logging.getLogger("ResultLogger")
    result_logger.setLevel(logging.INFO)

    result_file_handler = logging.FileHandler(result_log_file, mode="w")
    result_file_handler.setLevel(logging.INFO)
    result_file_handler.setFormatter(file_formatter)
    result_logger.addHandler(result_file_handler)

    main(args, result_dir)
