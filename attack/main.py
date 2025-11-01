from framework import (
    Attacker,
    Judge,
    StrategyExtractor,
    Target,
    Library,
    Connector,
    ColorFormatter,
    Agent,
    SemRelevance,
    StableDiffusion,
    StrongRejectJudge,
    HarmbenchJudge,
    AttackVersion,
)
import logging
import os
import pandas as pd
import argparse
from typing import Optional
from dotenv import load_dotenv
import random
from enum import Enum
import commons.utils
from typing import List
import json

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")
LOG_DIR = os.getenv("LOG_DIR")
RESULT_DIR = os.getenv("RESULT_DIR")

BACK_WALK_2_STEPS_RATE = 0.7
REGEN_WALK_RATE = 0.4
REGEN_LIMIT = 1


class Action(Enum):
    NEXT = "NEXT"
    REGEN = "REGEN"
    BACK = "BACK"


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--attacker_model_name", type=str, default="mistral")
    config.add_argument("--attacker_max_new_tokens", type=int, default=2000)
    config.add_argument("--target_model_name", type=str, default="llava1.6")
    config.add_argument("--target_max_new_tokens", type=int, default=300)
    config.add_argument("--num_attempts", type=int, default=3)
    config.add_argument("--dataset", type=str, default="advbench_tiny")
    config.add_argument("--task_i_start_from", type=int, default=0)
    config.add_argument("--num_tasks", type=int, default=0)
    config.add_argument("--rounds", type=int, default=5)
    config.add_argument("--max_rounds", type=int, default=9)  # Used in old decision mechanism
    config.add_argument("--library_name", type=str, default="strategy_library_initialized_24b_no_test_tasks")
    config.add_argument("--experiment_name", type=str, default="default_experiment")
    config.add_argument("--disable_vision", action="store_true")
    config.add_argument("--disable_reflection", action="store_true")
    config.add_argument("--disable_text_connection", action="store_true")
    config.add_argument("--disable_retrieval", action="store_true")
    config.add_argument("--disable_policy_selection", action="store_true")
    config.add_argument("--full_only", action="store_true")
    config.add_argument("--is_adashield", action="store_true")
    config.add_argument("--use_llama_guard", type=str, default="")
    config.add_argument("--max_num_steps", type=int, default=10)  # Used in CoA decision mechanism
    config.add_argument("--save_img", action="store_true")
    config.add_argument("--load_attack_chains_from", type=str, default="")  # an experiment name
    # config.add_argument("--use_my_judge", action="store_true")
    config.add_argument("--port", type=int, default=8000)

    return config


# def attack_pipeline(
#     task_i: int,
#     task: str,
#     attacker: Attacker,
#     judge: HarmbenchJudge | StrongRejectJudge,
#     target: Target,
#     library: Library,
#     connector: Optional[Connector],
#     stable_diffusion: Optional[StableDiffusion],
#     failure_history: str,
#     attempt_i: int,
#     is_text_prompt_connected: bool,
#     enable_retrieval: bool,
#     attack_chain_dict=None,
# ):
#     logger.info(f"{'='*20} Task {task_i}: {task} {'='*20}")
#     retrieved_strategies = None
#     if enable_retrieval:
#         # Retrieve strategy
#         retrieved_strategies = library.retrieve(task, strategy_selection_i=attempt_i if failure_history else 0)

#     # Generate attack chain
#     if not attack_chain_dict:
#         if failure_history:
#             attack_chain, strategy = attacker.get_attack_chain(task, retrieved_strategies)
#         else:
#             attack_chain, strategy, reflection = attacker.get_attack_chain_reflection(
#                 task, retrieved_strategies, failure_history
#             )
#             logger.info(f"Reflection: {reflection}")
#     else:
#         attacker.set_attack_chain(attack_chain_dict)
#         attack_chain, strategy = attacker.get_attack_chain_rounds(), attacker.get_strategy()
#     logger.info(f"Strategy: {strategy}")
#     logger.info(f"Attack chain:\n{attacker.get_attack_chain_text()}")

#     # Start the attack pipeline
#     for round_i in range(len(attack_chain)):
#         logger.info(f"Round {round_i + 1}")
#         attack_text_prompt = attack_chain[round_i]["prompt"]

#         # Vision connector
#         if connector:
#             sd_prompt, connected_text_prompt = connector.get_linked_prompt(task, strategy, attack_text_prompt)
#             img = stable_diffusion.gen_image(sd_prompt, task_i)
#             if is_text_prompt_connected:
#                 attack_version = AttackVersion.FULL
#                 attack_text_prompt = connected_text_prompt
#                 logger.info(f"Connected prompt: {attack_text_prompt}")
#             else:
#                 attack_version = AttackVersion.NO_TEXT_CONN
#         else:
#             attack_version = AttackVersion.NO_VISION
#             img = None

#         # Generate the target's response
#         response = target.get_response(attack_text_prompt, img)
#         logger.info(f"Response: {response}")

#         is_success = judge.eval(task, response)
#         logger.info(f"Judge Response: {is_success}")

#         # Determine if the response is jailbroken
#         if is_success:
#             logger.info("Jailbreak successful.")
#             return is_success, response, round_i + 1

#         target.add_prompt_to_conv(attack_text_prompt, img, attack_version)
#         target.add_response_to_conv(response)

#     logger.info("Jailbreak failed.")
#     return is_success, response, round_i + 1


def next_walk(target: Target, attacker: Attacker, task: str, round_i: int):
    # Next walk
    # Record latest response
    logger.info("NEXT walk")
    round_i += 1

    original_prompt = attacker.attack_chain[round_i]["prompt"]
    attack_text_prompt, refinement_justification = attacker.get_next_attack(task, round_i, target.get_conv_txt())
    logger.info(f"Original prompt: {original_prompt}")
    logger.debug(f"Refined prompt justification: {refinement_justification}")
    logger.info(f"Refined prompt: {attack_text_prompt}")

    return round_i, attack_text_prompt


def back_walk(
    target: Target,
    attacker: Attacker,
    task: str,
    round_i: int,
    retrieved_strategies,
    failure_history,
    backtrace_steps=1,
):
    # Back walk
    logger.info("BACK walk")

    round_i = max(0, round_i - backtrace_steps)
    if round_i == 0:
        logger.info(f"Regenerate attack chain")
        target.clear_history()
        if failure_history:
            attack_chain, strategy = attacker.get_attack_chain(task, retrieved_strategies)
        else:
            attack_chain, strategy, reflection = attacker.get_attack_chain_reflection(
                task, retrieved_strategies, failure_history
            )
            logger.info(f"Reflection: {reflection}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Attack chain:\n{attacker.get_attack_chain_text()}")
        attack_text_prompt = attack_chain[round_i]["prompt"]
        return round_i, attack_text_prompt

    target.backtrack(backtrace_steps)
    target.remove_previous_similarity(backtrace_steps)

    original_prompt = attacker.attack_chain[round_i]["prompt"]
    attack_text_prompt, refinement_justification = attacker.get_next_attack(task, round_i, target.get_conv_txt())
    logger.info(f"Original prompt: {original_prompt}")
    logger.debug(f"Refined prompt justification: {refinement_justification}")
    logger.info(f"Refined prompt: {attack_text_prompt}")

    return round_i, attack_text_prompt


def regen_walk(target: Target, attacker: Attacker, task: str, round_i: int):
    # REGEN walk
    logger.info("REGEN walk")
    original_prompt = attacker.attack_chain[round_i]["prompt"]
    attack_text_prompt, regen_justification = attacker.get_regen_attack(task, round_i, target.get_conv_txt())
    logger.info(f"Original prompt: {original_prompt}")
    logger.debug(f"Regenerated prompt justification: {regen_justification}")
    logger.info(f"Regenerated prompt: {attack_text_prompt}")

    return attack_text_prompt


def CoA_attack_pipeline(
    task_i: int,
    task: str,
    attacker: Attacker,
    judge: HarmbenchJudge,
    stronreject_judge: StrongRejectJudge,
    my_judge: Judge,
    target: Target,
    library: Library,
    connector: Optional[Connector],
    stable_diffusion: Optional[StableDiffusion],
    sem_relevance: SemRelevance,
    max_num_steps: int,
    failure_history: str,
    attempt_i: int,
    rounds: int,
    attack_versions: List[AttackVersion],
    enable_retrieval: bool,
    disable_policy_selection: bool,
    attack_chain_dict=None,
):

    logger.info(f"{'='*20} Task {task_i}: {task} {'='*20}")
    retrieved_strategies = None
    if enable_retrieval:
        # Retrieve strategy
        retrieved_strategies = library.retrieve(task, strategy_selection_i=attempt_i if failure_history else 0)

    # Generate attack chain
    if not attack_chain_dict:
        if not failure_history:
            attack_chain, strategy = attacker.get_attack_chain(task, retrieved_strategies)
        else:
            attack_chain, strategy, reflection = attacker.get_attack_chain_reflection(
                task, retrieved_strategies, failure_history
            )
            logger.info(f"Reflection: {reflection}")
    else:
        attacker.set_attack_chain(attack_chain_dict)
        attack_chain, strategy = attacker.get_attack_chain_rounds(), attacker.get_strategy()
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Attack chain:\n{attacker.get_attack_chain_text()}")

    # Start the attack pipeline
    round_i = 0
    proposed_text_prompt = attack_chain[round_i]["prompt"]
    action = Action.NEXT
    regen_count = 0
    regen_records = []  # List of tuples: [(sem, prompt, image, response, sem_no_history, attack_version)]
    for step in range(max_num_steps):
        logger.info(f"---------- Step: {step}, Round {round_i + 1} ----------")

        # Generate the target's response
        # logger.debug(f"Target conv: rounds: {len(target.conv)/2}\n{target.get_conv_txt()}")
        # logger.debug(f"Target sems length: {len(target.sems)}; images length: {len(target.images)}")
        recording_tuples = []
        sd_prompt, connected_text_prompt, img = None, None, None
        for attack_version in attack_versions:
            logger.info(f"Attack version: {attack_version.name}")
            if attack_version == AttackVersion.FULL or attack_version == AttackVersion.NO_TEXT_CONN:
                # Vision
                if connector:
                    if not sd_prompt and not connected_text_prompt and not img:
                        sd_prompt, connected_text_prompt = connector.get_linked_prompt(
                            task, strategy, proposed_text_prompt
                        )
                        img = stable_diffusion.gen_image(sd_prompt, task_i, round=round_i + 1)

                    if attack_version == AttackVersion.FULL:
                        attack_text_prompt = connected_text_prompt
                        logger.info(f"Connected prompt: {attack_text_prompt}")
                    else:
                        attack_text_prompt = proposed_text_prompt
                else:
                    continue
            else:
                # No vision
                img = None
                attack_text_prompt = proposed_text_prompt

            response = target.get_response(attack_text_prompt, img)
            # sem = stronreject_judge.eval(task, response)
            sem = sem_relevance.compute_similarity(task, response)
            logger.info(f"Response: {response}")
            logger.info(f"Similarity: {sem:.4f}")

            is_success = judge.eval(task, response)
            logger.info(f"Judge Response: {is_success}")
            # Determine if the response is jailbroken
            if is_success:
                logger.info("Jailbreak successful.")
                target.add_prompt_to_conv(attack_text_prompt, img, attack_version)
                return (
                    is_success,
                    response,
                    round_i + 1,
                    step + 1,
                )

            recording_tuples.append((sem, attack_text_prompt, img, response, attack_version))

        # Compare the responses
        top_sem, top_text_prompt, top_img, top_response, top_attack_version = max(recording_tuples, key=lambda x: x[0])
        logger.info(f"Choose top response from: {top_attack_version.name}")

        jailbreak_score = my_judge.eval(task, top_response)
        logger.info(f"Jailbreak score: {jailbreak_score}")

        # Make action decision

        # The first round
        if round_i == 0 or disable_policy_selection:
            # Always NEXT walk for the first round
            if round_i == rounds - 1:
                # set best_response to be top_response for return
                best_response = top_response
                break
            action = Action.NEXT
            logger.info(f"Similarity: sem: {top_sem:.4f}")
            target.add_response_similarity(top_sem)
            target.add_prompt_to_conv(top_text_prompt, top_img, top_attack_version)
            target.add_response_to_conv(top_response)
            round_i, proposed_text_prompt = next_walk(target, attacker, task, round_i)
            continue

        # The second round or above
        response_no_history = target.get_response_no_history(top_text_prompt, top_img)
        # sem_no_history = stronreject_judge.eval(task, response_no_history)
        sem_no_history = sem_relevance.compute_similarity(task, response_no_history)
        logger.debug(f"Response from no history: {response_no_history}")

        # Record this step
        regen_records.append((top_sem, top_text_prompt, top_img, top_response, sem_no_history, top_attack_version))
        # Select the best text prompt so far for this round
        best_sem, best_text_prompt, best_img, best_response, best_sem_no_history, best_attack_version = max(
            regen_records, key=lambda x: x[0]
        )

        sem_prev = target.get_similarity(round_i - 1)
        logger.info(
            f"Similarity: sem: {best_sem:.4f}; Sem no history: {best_sem_no_history:.4f}; Sem prev: {sem_prev:.4f}"
        )
        # NEXT walk check
        if best_sem >= sem_prev and best_sem >= best_sem_no_history:
            if round_i == rounds - 1:
                # The last round has reached but the jailbreak has not completed
                if random.random() < REGEN_WALK_RATE:
                    if regen_count >= REGEN_LIMIT:
                        # Tried to regenerate before, end the attack
                        break
                    action = action.REGEN
                    regen_count += 1
                    proposed_text_prompt = regen_walk(target, attacker, task, round_i)
                else:
                    backtrace_steps = 2 if random.random() < BACK_WALK_2_STEPS_RATE else 1
                    regen_count = 0
                    regen_records = []
                    action = action.BACK
                    round_i, proposed_text_prompt = back_walk(
                        target,
                        attacker,
                        task,
                        round_i,
                        retrieved_strategies,
                        failure_history,
                        backtrace_steps=backtrace_steps,
                    )
            else:
                # NEXT walk to the next round
                regen_count = 0
                regen_records = []
                action = action.NEXT
                target.add_response_similarity(best_sem)
                target.add_prompt_to_conv(best_text_prompt, best_img, best_attack_version)
                target.add_response_to_conv(best_response)
                round_i, proposed_text_prompt = next_walk(target, attacker, task, round_i)
            continue

        # BACK walk check
        if best_sem < sem_prev and best_sem_no_history > sem_prev:
            # else:
            regen_count = 0
            regen_records = []
            action = action.BACK
            round_i, proposed_text_prompt = back_walk(
                target, attacker, task, round_i, retrieved_strategies, failure_history
            )
            continue

        # REGEN walk
        if regen_count >= REGEN_LIMIT:
            if round_i == rounds - 1:
                # After retry, the last round still failed, end the attack
                break
            # Already regenerated 2 times, use the best try and then next walk
            action = action.NEXT
            regen_records = []
            regen_count = 0

            target.add_response_similarity(best_sem)
            target.add_prompt_to_conv(best_text_prompt, best_img, best_attack_version)
            target.add_response_to_conv(best_response)
            round_i, proposed_text_prompt = next_walk(target, attacker, task, round_i)
        else:
            # REGEN walk
            regen_count += 1
            action = action.REGEN
            proposed_text_prompt = regen_walk(target, attacker, task, round_i)

    logger.info("Jailbreak failed.")
    return is_success, best_response, round_i + 1, step + 1


def attacks(args, result_dir):
    # Strategy library
    library = Library()
    library.load(args.library_name)

    # Red team
    attacker = Attacker(args.rounds, args.attacker_model_name, args.attacker_max_new_tokens)
    my_judge = Judge(attacker.model_name, attacker.max_new_tokens)
    judge = HarmbenchJudge()
    strongreject_judge = StrongRejectJudge()
    sem_relevance = SemRelevance()
    strategy_extractor = StrategyExtractor(attacker.model_name, attacker.max_new_tokens)

    if args.disable_vision:
        connector = None
        stable_diffusion = None
        attack_versions = [AttackVersion.NO_VISION]
    else:
        connector = Connector(attacker.model_name, attacker.max_new_tokens)
        stable_diffusion = StableDiffusion(args.experiment_name, args.save_img)
        if args.disable_text_connection:
            attack_versions = [AttackVersion.NO_TEXT_CONN]
        elif args.full_only:
            attack_versions = [AttackVersion.FULL]
        else:
            attack_versions = [AttackVersion.FULL, AttackVersion.NO_TEXT_CONN, AttackVersion.NO_VISION]

    given_attack_chains = {}
    if args.load_attack_chains_from:
        try:
            if "json" in args.load_attack_chains_from:
                attack_chains_path = os.path.join(RESULT_DIR, args.load_attack_chains_from)
            else:
                attack_chains_path = os.path.join(
                    RESULT_DIR, args.load_attack_chains_from, f"attack_chains_{args.task_i_start_from}.json"
                )
            with open(attack_chains_path, "r") as f:
                given_attack_chains_str_key = json.load(f)
            for k, v in given_attack_chains_str_key.items():
                given_attack_chains[int(k)] = v
        except FileNotFoundError:
            print(f"Error: The file '{attack_chains_path}' does not exist.")

    # Blue team
    target = Target(args.target_model_name, args.target_max_new_tokens, args.is_adashield, args.use_llama_guard)

    # Load tasks
    df = pd.read_csv(os.path.join(DATA_DIR, args.dataset, "data.csv"))
    total_num_tasks = len(df)
    if args.num_tasks == 0:
        num_tasks = len(df["task"]) - args.task_i_start_from
    else:
        num_tasks = args.num_tasks

    # Attack each task for num_attempts times and record the results
    results = []
    attack_chains_dict = {}
    for task_i, task in enumerate(df["task"]):
        if task_i < args.task_i_start_from:
            continue
        num_attempts = args.num_attempts
        if given_attack_chains and task_i in given_attack_chains:
            num_attempts = 1

        result_logger.info(f"Task {task_i}. {task}")
        failure_history = ""

        for attempt_i in range(num_attempts):
            target.clear_history()

            # if not args.use_CoA:
            #     is_success, last_response, rounds_num = attack_pipeline(
            #         task_i,
            #         task,
            #         attacker,
            #         judge,
            #         target,
            #         library,
            #         connector,
            #         stable_diffusion,
            #         failure_history,
            #         attempt_i,
            #         is_text_prompt_connected=not args.disable_text_connection,
            #         enable_retrieval=not args.disable_retrieval,
            #         attack_chain_dict=given_attack_chains.get(task_i, None) if given_attack_chains else None,
            #     )
            #     jailbreak_score = 0
            #     sem = 0
            #     steps_num = rounds_num
            # else:
            is_success, last_response, rounds_num, steps_num = CoA_attack_pipeline(
                task_i,
                task,
                attacker,
                judge,
                strongreject_judge,
                my_judge,
                target,
                library,
                connector,
                stable_diffusion,
                sem_relevance,
                args.max_num_steps,
                failure_history,
                attempt_i,
                args.rounds,
                attack_versions=attack_versions,
                enable_retrieval=not args.disable_retrieval,
                disable_policy_selection=args.disable_policy_selection,
                attack_chain_dict=given_attack_chains.get(task_i, None) if given_attack_chains else None,
            )
            jailbreak_score = my_judge.eval(task, last_response)
            sem = sem_relevance.compute_similarity(task, last_response)

            judge_score = strongreject_judge.eval(task, last_response)
            last_response_cleaned = last_response.replace("\n", " ")
            result = {
                "task_i": task_i,
                "task": task,
                "attempt_i": attempt_i,
                "is_success": is_success,
                "judge_score": judge_score,
                "jailbreak_score": jailbreak_score,
                "similarity": sem,
                "rounds_num": rounds_num,
                "steps_num": steps_num,
                "last_response": last_response_cleaned,
            }
            for round_i in range(args.rounds):
                result[f"attack_version_{round_i}"] = target.get_attack_version(round_i)
            result["total_target_query"] = target.get_total_target_query()

            results.append(result)
            result_logger.info(
                f"\tAttempt {attempt_i+1}: {'Success' if is_success else 'Failure'}, Judge score: {judge_score}, Jailbreak score: {jailbreak_score}, Similarity: {sem}, Rounds num: {rounds_num}, Steps num: {steps_num}, Total target query: {result['total_target_query']} - Last response: {last_response_cleaned}"
            )

            # Conclusion of the attack attempt
            if is_success:
                # Extract strategy from a successful jailbreak attempt and add it to the library
                strategy_json = strategy_extractor.extract_strategy(
                    task, target.get_attacker_msgs(), judge_score, library.get_all_in_txt()
                )
                strategy_json["task"] = task
                strategy_json["example"] = target.get_attacker_msgs()
                strategy_json["score"] = judge_score  # Have to use an actual harmfulness score

                library.add(strategy_json, True, if_notify=True)

                # Record attack chain
                attack_chains_dict[task_i] = {
                    "strategy": attacker.get_strategy(),
                    "rounds": attacker.get_attack_chain_rounds(),
                }
                break
            if not args.disable_reflection:
                # If the jailbreak attempt failed, add the attempt to the failure history
                failure_history += f"[Attempt {attempt_i + 1}]:\nFailed Strategy: {attacker.strategy}\nFailed Attack Plan:\n{attacker.get_attack_chain_text(rounds_num-1)}\nOpponent Last Response: {last_response_cleaned}\n"

        if task_i == args.task_i_start_from + num_tasks - 1 or task_i == total_num_tasks - 1:
            # Reached the last task, stop attacking more tasks
            break

    library.save(args.library_name + "_" + args.experiment_name + "_" + str(args.task_i_start_from), result_dir)

    return results, attack_chains_dict


if __name__ == "__main__":
    # Argument parsing
    args = config().parse_args()
    commons.utils.PORT = args.port

    output_dir = os.path.join(os.getcwd(), RESULT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, args.experiment_name)
    os.makedirs(result_dir, exist_ok=True)

    # Logging setup
    log_file = os.path.join(result_dir, f"running_{args.task_i_start_from}.log")

    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColorFormatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    result_log_file = os.path.join(result_dir, f"results_{args.task_i_start_from}.log")
    result_logger = logging.getLogger("ResultLogger")
    result_logger.setLevel(logging.INFO)

    result_file_handler = logging.FileHandler(result_log_file, mode="w")
    result_file_handler.setLevel(logging.INFO)
    result_file_handler.setFormatter(file_formatter)
    result_logger.addHandler(result_file_handler)

    # Start the attack
    logger.info("START")
    results, attack_chains_dict = attacks(args, result_dir)

    # Save results to a DataFrame and calculate statistics
    results_df = pd.DataFrame(results)
    binary_asr = results_df.groupby("task_i")["is_success"].any().mean()
    average_asr = results_df["is_success"].mean()
    average_rounds_num_in_success = results_df[results_df["is_success"]]["rounds_num"].mean()
    average_steps_num_in_success = results_df[results_df["is_success"]]["steps_num"].mean()
    average_score = results_df.groupby("task_i")["judge_score"].max().mean()
    average_success_jailbreak_score = results_df[results_df["is_success"]]["jailbreak_score"].mean()
    average_jailbreak_score = results_df["jailbreak_score"].mean()
    average_success_sem = results_df[results_df["is_success"]]["similarity"].mean()
    average_failure_sem = results_df[results_df["is_success"] == False]["similarity"].mean()
    average_sem = results_df["similarity"].mean()
    result_logger.info(
        f"Binary ASR: {binary_asr}; Average ASR: {average_asr}; Average Score of the highest score attempts: {average_score}; Average successful rounds num: {average_rounds_num_in_success}; Average successful steps num: {average_steps_num_in_success}; Average successful jailbreak score: {average_success_jailbreak_score}; Average jailbreak score: {average_jailbreak_score}; Average successful similarity: {average_success_sem}; Average similarity: {average_sem}; Average failed similarity: {average_failure_sem}"
    )

    stats_path = os.path.join(result_dir, f"stats_{args.task_i_start_from}.csv")
    results_df.to_csv(stats_path, index=False)
    result_logger.info(f"Results saved to {stats_path}")

    if attack_chains_dict:
        with open(os.path.join(result_dir, f"attack_chains_{args.task_i_start_from}.json"), "w") as f:
            json.dump(attack_chains_dict, f, indent=4)
