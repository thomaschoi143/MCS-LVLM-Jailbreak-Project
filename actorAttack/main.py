from preattack import PreAttack
from inattack import InAttack
from config import PreAttackConfig, InAttackConfig
import config
import argparse
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "attack")))
from framework import HarmbenchJudge, StrongRejectJudge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ActorAttack")
    # actor
    parser.add_argument("--questions", type=int, default=-1, help="Number of questions.")
    parser.add_argument("--actors", type=int, default=3, help="Number of actors for each question.")
    # parser.add_argument(
    #     "--behavior", default="../data/advbench_tiny/data.csv", help="Path of harmful behaviors CSV file."
    # )
    # attack
    parser.add_argument("--attack_model_name", type=str, default="gpt-4o", help="Attack Model name.")
    parser.add_argument("--target_model_name", type=str, default="gpt-4o", help="Target Model name.")
    parser.add_argument("--early_stop", type=bool, default=True, help="early stop if the judge LLM yields success.")
    parser.add_argument("--dynamic_modify", type=bool, default=True, help="apply dynamic modification.")

    parser.add_argument("--task_i_start_from", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dataset", type=str, default="advbench_tiny")

    args = parser.parse_args()
    config.port = args.port

    output_dir = os.path.join(os.getcwd(), "./results")
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, args.experiment_name)
    os.makedirs(result_dir, exist_ok=True)

    result_log_file = os.path.join(result_dir, f"results_{args.task_i_start_from}.log")
    result_logger = logging.getLogger("ResultLogger")
    result_logger.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    result_file_handler = logging.FileHandler(result_log_file, mode="w")
    result_file_handler.setLevel(logging.INFO)
    result_file_handler.setFormatter(file_formatter)
    result_logger.addHandler(result_file_handler)

    strongreject_judge = StrongRejectJudge(args.port)
    judge = HarmbenchJudge(args.port)

    behavior_csv = os.path.join("../data", args.dataset, "data.csv")

    pre_attack_config = PreAttackConfig(
        model_name=args.attack_model_name, actor_num=args.actors, behavior_csv=behavior_csv
    )
    pre_attacker = PreAttack(pre_attack_config)

    pre_attack_data_path = pre_attacker.infer(
        args.questions, task_i_start_from=args.task_i_start_from, experiment_name=args.experiment_name
    )
    print(f"pre-attack data path: {pre_attack_data_path}")

    in_attack_config = InAttackConfig(
        attack_model_name=args.attack_model_name,
        target_model_name=args.target_model_name,
        pre_attack_data_path=pre_attack_data_path,
        early_stop=args.early_stop,
        dynamic_modify=args.dynamic_modify,
    )

    in_attacker = InAttack(in_attack_config, judge, strongreject_judge)
    final_result_path = in_attacker.infer(args.task_i_start_from, args.experiment_name)
    print(f"final attack result path: {final_result_path}")
