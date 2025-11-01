import pandas as pd
import argparse
import os
import logging
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = "results"
DATA_DIR = "../data"


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--experiment_name", type=str, default="default_experiment")
    config.add_argument("--tasks_interval", type=int, default=12)
    config.add_argument("--dataset", type=str, default="harmbench")

    return config


def cal_successful_rounds_num(stats_df):
    successful_rounds_nums = []
    for idx, row in stats_df.iterrows():
        if row["is_success"]:
            task_i = row["task_i"]
            json_file_path = os.path.join(
                script_dir, RESULTS_DIR, args.experiment_name, str(task_i), "attempt", "1.json"
            )
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)
                    successful_rounds_nums.append(len(json_data["messages"]) / 2)
            else:
                logger.error("File not found: {json_file_path}")
    return successful_rounds_nums and sum(successful_rounds_nums) / len(successful_rounds_nums) or 0


if __name__ == "__main__":
    args = config().parse_args()
    dataset_df = pd.read_csv(os.path.join(script_dir, DATA_DIR, args.dataset, "data.csv"))
    num_tasks = len(dataset_df)

    result_dir = os.path.join(script_dir, RESULTS_DIR, args.experiment_name)

    # Logging setup
    result_log_file = os.path.join(result_dir, f"recal_successful_rounds.log")
    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.INFO)

    result_file_handler = logging.FileHandler(result_log_file, mode="w")
    result_file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    result_file_handler.setFormatter(file_formatter)
    logger.addHandler(result_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    combined_stats_df = None
    results = []
    for task_i_start_from in range(0, num_tasks, args.tasks_interval):
        stats_path = os.path.join(result_dir, f"stats_{task_i_start_from}.csv")
        if not os.path.exists(stats_path):
            logger.info(f"Stats {stats_path} does not exist.")
            continue

        stats_df = pd.read_csv(stats_path)
        avg_successful_rounds_num = cal_successful_rounds_num(stats_df)
        batch = f"{task_i_start_from}-{task_i_start_from+args.tasks_interval-1}"
        results.append(
            {
                "batch": batch,
                "average_successful_rounds_num": avg_successful_rounds_num,
            }
        )
        logger.info(f"Batch {batch[:5]:<5} - average_successful_rounds_num: {avg_successful_rounds_num:.4f}")

    logger.info(
        f"------Summary for {args.experiment_name} on {args.dataset} with tasks interval {args.tasks_interval}------"
    )
    results_df = pd.DataFrame(results)
    avg_successful_rounds_num = results_df["average_successful_rounds_num"].mean()
    logger.info(f"Overall average_successful_rounds_num: {avg_successful_rounds_num:.4f}")
