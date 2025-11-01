from PIL import Image
import os
import argparse
from const import *
import logging
import json
import requests
import pandas as pd
from fastchat.model import get_conversation_template
from io import BytesIO
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "attack")))
from framework import HarmbenchJudge, StrongRejectJudge

RESULT_DIR = "./results"
DATA_DIR = "../data"
PORT = None

dataformat2prompt = {
    "images_wr": wr_game_prompt,
    "images_mirror": mirror_game_prompt,
    "images_base64": base64_game_prompt,
    "images_rotate": rotate_game_prompt,
}


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--images_dir", type=str, default="./mirror_images")
    config.add_argument("--dataset_name", type=str, default="advbench_tiny")
    config.add_argument("--experiment_name", type=str, default="default_experiment")
    config.add_argument("--num_attempts", type=int, default=1)
    config.add_argument("--port", type=int, default=8300)

    return config


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


def random_shuffle_sentence(sentence):
    ssp = sentence.split()
    random.shuffle(ssp)
    return ssp


def attack(image_path, task, task_i, instruction, judge):
    image = Image.open(os.path.join(image_path, f"{task_i}.png"))
    prompt = dataformat2prompt["images_mirror"].format(random_shuffle_sentence(instruction))
    response_text = get_victim_generation(prompt, image)
    logger.info(f"{'='*20} Task {task_i}: {task} {'='*20}")
    logger.info(f"Response: {response_text}")
    judge_res = judge.eval(task, response_text)

    return judge_res, response_text


if __name__ == "__main__":
    args = config().parse_args()
    PORT = args.port

    output_dir = os.path.join(os.getcwd(), RESULT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, args.experiment_name)
    os.makedirs(result_dir, exist_ok=True)

    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    log_file = os.path.join(result_dir, f"running.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    result_log_file = os.path.join(result_dir, f"results.log")
    result_logger = logging.getLogger("ResultLogger")
    result_logger.setLevel(logging.INFO)

    result_file_handler = logging.FileHandler(result_log_file, mode="w")
    result_file_handler.setLevel(logging.INFO)
    result_file_handler.setFormatter(file_formatter)
    result_logger.addHandler(result_file_handler)

    images_dir = os.path.join(args.images_dir, args.dataset_name)
    data_df = pd.read_csv(os.path.join(images_dir, "data.csv"))

    results = []

    if "harmbench" in args.dataset_name:
        judge = HarmbenchJudge(args.port)
    else:
        judge = StrongRejectJudge(args.port)

    for task_i, row in data_df.iterrows():
        result_logger.info(f"Task {task_i}. {row['task']}")
        for attempt_i in range(args.num_attempts):
            judge_res, last_response = attack(images_dir, row["task"], task_i, row["instruction"], judge)

            is_success = judge_res if isinstance(judge_res, bool) else False
            judge_score = judge_res if isinstance(judge_res, float) else 0
            results.append(
                {
                    "task_i": task_i,
                    "task": row["task"],
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
    print(f"My results saved to {stats_path}")
