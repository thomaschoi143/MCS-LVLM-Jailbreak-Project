import pandas as pd
import argparse
import os
import logging
from collections import defaultdict
from pprint import pformat

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "./data"


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--attack_method_dir", type=str, default="attack")
    config.add_argument("--experiment_name", type=str, default="default_experiment")
    config.add_argument("--tasks_interval", type=int, default=12)
    config.add_argument("--dataset", type=str, default="harmbench")
    config.add_argument("--max_rounds_num", type=int, default=5)

    return config


def cal_stats(results_df, max_rounds_num):
    groupby_col = "task_i"
    if groupby_col not in results_df.columns:
        groupby_col = "task"
    binary_asr = results_df.groupby(groupby_col)["is_success"].any().mean()
    average_asr = results_df["is_success"].mean()
    average_rounds_num_in_success = results_df[results_df["is_success"]]["rounds_num"].mean()
    average_score = results_df.groupby(groupby_col).tail(1)["judge_score"].mean()

    average_steps_num_in_success = 0
    if "steps_num" in results_df.columns:
        average_steps_num_in_success = results_df[results_df["is_success"]]["steps_num"].mean()

    average_success_jailbreak_score = 0
    average_jailbreak_score = 0
    if "jailbreak_score" in results_df.columns:
        average_success_jailbreak_score = results_df[results_df["is_success"]]["jailbreak_score"].mean()
        average_jailbreak_score = results_df["jailbreak_score"].mean()

    average_success_sem = 0
    average_failure_sem = 0
    average_sem = 0
    if "similarity" in results_df.columns:
        average_success_sem = results_df[results_df["is_success"]]["similarity"].mean()
        average_failure_sem = results_df[results_df["is_success"] == False]["similarity"].mean()
        average_sem = results_df["similarity"].mean()

    average_total_target_query_in_success = 0
    average_total_target_query_to_get_success = 0
    average_total_target_query = 0
    if "total_target_query" in results_df.columns:
        average_total_target_query_in_success = results_df[results_df["is_success"]]["total_target_query"].mean()

        successful_tasks = results_df[results_df["is_success"]][groupby_col]
        total_target_query_sum_by_task = results_df.groupby(groupby_col)["total_target_query"].sum()
        total_target_query_sum_by_successful_task = total_target_query_sum_by_task[
            total_target_query_sum_by_task.index.isin(successful_tasks)
        ]
        average_total_target_query = total_target_query_sum_by_task.mean()
        average_total_target_query_to_get_success = total_target_query_sum_by_successful_task.mean()

    # Attack version statistics
    attack_version_stat = defaultdict(lambda: [0] * max_rounds_num)
    for round_i in range(max_rounds_num):
        col = f"attack_version_{round_i}"
        if col in results_df.columns:
            round_attack_version_dict = results_df[results_df["is_success"]][col].value_counts().to_dict()
            for attack_version, count in round_attack_version_dict.items():
                attack_version_stat[attack_version][round_i] = count

    return (
        binary_asr,
        average_asr,
        average_score,
        average_rounds_num_in_success,
        average_steps_num_in_success,
        average_success_jailbreak_score,
        average_jailbreak_score,
        average_success_sem,
        average_failure_sem,
        average_sem,
        average_total_target_query_in_success,
        average_total_target_query_to_get_success,
        average_total_target_query,
        attack_version_stat,
    )


if __name__ == "__main__":
    args = config().parse_args()
    dataset_df = pd.read_csv(os.path.join(script_dir, DATA_DIR, args.dataset, "data.csv"))
    num_tasks = len(dataset_df)

    result_dir = os.path.join(script_dir, args.attack_method_dir, "results", args.experiment_name)

    # Logging setup
    result_log_file = os.path.join(result_dir, f"summarizer.log")
    result_logger = logging.getLogger("SummarizerLogger")
    result_logger.setLevel(logging.INFO)

    result_file_handler = logging.FileHandler(result_log_file, mode="w")
    result_file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    result_file_handler.setFormatter(file_formatter)
    result_logger.addHandler(result_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    result_logger.addHandler(console_handler)

    combined_stats_df = None
    results = []
    for task_i_start_from in range(0, num_tasks, args.tasks_interval):
        stats_path = os.path.join(result_dir, f"stats_{task_i_start_from}.csv")
        if not os.path.exists(stats_path):
            result_logger.info(f"Stats {stats_path} does not exist.")
            continue

        stats_df = pd.read_csv(stats_path)
        (
            binary_asr,
            average_asr,
            average_score,
            average_rounds_num_in_success,
            average_steps_num_in_success,
            average_success_jailbreak_score,
            average_jailbreak_score,
            average_success_sem,
            average_failure_sem,
            average_sem,
            average_total_target_query_in_success,
            average_total_target_query_to_get_success,
            average_total_target_query,
            attack_version_stat,
        ) = cal_stats(stats_df, args.max_rounds_num)
        batch = f"{task_i_start_from}-{task_i_start_from+args.tasks_interval-1}"
        results.append(
            {
                "batch": batch,
                "binary_asr": binary_asr,
                "average_asr": average_asr,
                "average_score": average_score,
                "average_rounds_num_in_success": average_rounds_num_in_success,
                "average_steps_num_in_success": average_steps_num_in_success,
                "average_success_jailbreak_score": average_success_jailbreak_score,
                "average_jailbreak_score": average_jailbreak_score,
                "average_success_sem": average_success_sem,
                "average_failure_sem": average_failure_sem,
                "average_sem": average_sem,
                "average_total_target_query_in_success": average_total_target_query_in_success,
                "average_total_target_query_to_get_success": average_total_target_query_to_get_success,
                "average_total_target_query": average_total_target_query,
            }
        )
        result_logger.info(
            f"Batch {batch[:5]:<5} - Binary ASR: {binary_asr:.2f}, Average ASR: {average_asr:.4f}, Average max score: {average_score:.4f}, Average successful rounds num: {average_rounds_num_in_success:.4f}, Average successful steps num: {average_steps_num_in_success:.4f}, Average successful jailbreak score: {average_success_jailbreak_score:.4f}, Average jailbreak score: {average_jailbreak_score:.4f}, Average successful similarity: {average_success_sem:.4f}, Average failed similarity: {average_failure_sem:.4f}, Average similarity: {average_sem:.4f}, Average total target query in success: {average_total_target_query_in_success:.4f}, Average total target query to get success: {average_total_target_query_to_get_success:.4f}, Average total target query: {average_total_target_query:.4f}"
        )
        result_logger.info(f"Attack version stats: \n{pformat(attack_version_stat)}")

        if combined_stats_df is None:
            combined_stats_df = stats_df
        else:
            combined_stats_df = pd.concat([combined_stats_df, stats_df], ignore_index=True)

    (
        binary_asr,
        average_asr,
        average_score,
        average_rounds_num_in_success,
        average_steps_num_in_success,
        average_success_jailbreak_score,
        average_jailbreak_score,
        average_success_sem,
        average_failure_sem,
        average_sem,
        average_total_target_query_in_success,
        average_total_target_query_to_get_success,
        average_total_target_query,
        attack_version_stat,
    ) = cal_stats(combined_stats_df, args.max_rounds_num)
    result_logger.info(
        f"------Summary for {args.experiment_name} on {args.dataset} with tasks interval {args.tasks_interval}------"
    )
    result_logger.info(f"Total number of tasks summarized: {combined_stats_df['task'].nunique()}")
    result_logger.info(f"Total number of samples summarized: {combined_stats_df.shape[0]}")
    result_logger.info(f"Binary ASR: {binary_asr}")
    result_logger.info(f"Average ASR: {average_asr}")
    result_logger.info(f"Average Score: {average_score}")
    result_logger.info(f"Average successful rounds_num: {average_rounds_num_in_success}")
    result_logger.info(f"Average successful steps_num: {average_steps_num_in_success}")
    result_logger.info(f"Average successful jailbreak score: {average_success_jailbreak_score}")
    result_logger.info(f"Average jailbreak score: {average_jailbreak_score}")
    result_logger.info(f"Average successful similarity: {average_success_sem}")
    result_logger.info(f"Average failed similarity: {average_failure_sem}")
    result_logger.info(f"Average similarity: {average_sem}")
    result_logger.info(f"Average total target query in success: {average_total_target_query_in_success}")
    result_logger.info(f"Average total target query to get success: {average_total_target_query_to_get_success}")
    result_logger.info(f"Average total target query: {average_total_target_query}")
    result_logger.info(f"Attack version stats: \n{pformat(attack_version_stat)}")

    results_df = pd.DataFrame(results)
    results_path = os.path.join(result_dir, "summary.csv")
    results_df.to_csv(results_path, index=False)
    result_logger.info(f"Summary saved to {results_path}")
