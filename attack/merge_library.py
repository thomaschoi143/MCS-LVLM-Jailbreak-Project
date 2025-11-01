from dotenv import load_dotenv
import os
import argparse
import pandas as pd
from framework import Library
import logging
from framework import ColorFormatter

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
RESULT_DIR = os.getenv("RESULT_DIR")
LOG_DIR = os.getenv("LOG_DIR")


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--dataset", type=str, default="harmbench_warmup")
    config.add_argument("--tasks_interval", type=int, default=12)
    config.add_argument("--experiment_name", type=str, default="default_experiment")
    config.add_argument("--base_library_name", type=str, default="strategy_library_initialized_24b_no_test_tasks")

    return config


def merge_library(dataset, tasks_interval, experiment_name, base_library_name):
    data_df = pd.read_csv(os.path.join(DATA_DIR, dataset, "data.csv"))
    tasks_num = len(data_df)
    task_i_start_from = 0

    library = Library()
    library.load(
        f"{base_library_name}_{experiment_name}_{task_i_start_from}", os.path.join(RESULT_DIR, experiment_name)
    )

    task_i_start_from += tasks_interval
    while task_i_start_from < tasks_num:
        library2 = Library()
        loaded_success = library2.load(
            f"{base_library_name}_{experiment_name}_{task_i_start_from}", os.path.join(RESULT_DIR, experiment_name)
        )
        if loaded_success:
            library.merge(library2)

        task_i_start_from += tasks_interval

    target = "qwen"
    if "llava" in experiment_name:
        target = "llava"
    elif "llama" in experiment_name:
        target = "llama"

    library.save(f"{base_library_name}_warmup_{target}")


if __name__ == "__main__":
    args = config().parse_args()

    # Logging setup
    log_dir = os.path.join(os.getcwd(), LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "merge_library.log")
    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColorFormatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Initialize the strategy library with human multi-turn attack dataset
    logger.info("START")

    merge_library(args.dataset, args.tasks_interval, args.experiment_name, args.base_library_name)
