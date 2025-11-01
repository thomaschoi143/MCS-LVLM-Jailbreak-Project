import logging
import json
import requests
import numpy as np
import os
import pickle
import faiss
from dotenv import load_dotenv
import commons.utils

load_dotenv()
LOG_DIR = os.getenv("LOG_DIR")

logger = logging.getLogger("CustomLogger")


class Library:

    def __init__(self):
        # Key: strategy name
        # Value: a dict: "definition": str, "source": "attack" | "human",
        #                "tasks": list of str, "examples": 2-D list of str,
        #                "tasks_embeddings": list of np array(512, ), "scores": list of str
        self.strategies_dict = {}

    def add(self, new_strategy, is_from_attack, if_notify=False):
        new_strategy_name = new_strategy["strategy"]

        if new_strategy_name not in self.strategies_dict:
            self.strategies_dict[new_strategy_name] = dict()
            self.strategies_dict[new_strategy_name]["definition"] = new_strategy["definition"]
            self.strategies_dict[new_strategy_name]["source"] = "attack" if is_from_attack else "human"
            self.strategies_dict[new_strategy_name]["tasks"] = []
            self.strategies_dict[new_strategy_name]["examples"] = []
            self.strategies_dict[new_strategy_name]["tasks_embeddings"] = []
            self.strategies_dict[new_strategy_name]["scores"] = []

        if "embedding" in new_strategy:
            task_embedding = new_strategy["embedding"]
        else:
            task_embedding = self.get_embedding(new_strategy["task"])

        # Duplicated task
        if new_strategy["task"] in self.strategies_dict[new_strategy_name]["tasks"]:
            return

        self.strategies_dict[new_strategy_name]["tasks"].append(new_strategy["task"])
        self.strategies_dict[new_strategy_name]["examples"].append(new_strategy["example"])
        self.strategies_dict[new_strategy_name]["tasks_embeddings"].append(task_embedding)
        self.strategies_dict[new_strategy_name]["scores"].append(new_strategy["score"])

        if if_notify:
            if len(self.strategies_dict[new_strategy_name]["tasks"]) == 1:
                new_dict = {"strategy": new_strategy_name, "definition": new_strategy["definition"]}
                logger.info(f"Identified a new strategy: {json.dumps(new_dict, indent=4)}")
            else:
                logger.info(f"Strategy '{new_strategy_name}' has been updated with a new task.")

    def retrieve(self, task, strategy_selection_i=0):
        task_embedding = self.get_embedding(task)

        all_embeddings = []
        all_tasks = []
        all_examples = []
        all_scores = []
        reverse_map = []  # To map from index number to strategy name
        for s_name, s_info in self.strategies_dict.items():
            for i, emb in enumerate(self.strategies_dict[s_name]["tasks_embeddings"]):
                all_embeddings.append(emb)
                all_tasks.append(self.strategies_dict[s_name]["tasks"][i])
                all_examples.append(self.strategies_dict[s_name]["examples"][i])
                all_scores.append(self.strategies_dict[s_name]["scores"][i])
                reverse_map.append(s_name)

        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        dim = all_embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(all_embeddings)

        num_to_retrieve = len(all_embeddings)
        distances, indexes = index.search(task_embedding.reshape(1, dim), k=num_to_retrieve)

        distances, indexes = distances[0], indexes[0]
        seen_strategies = []
        retrieved_strategies = {}

        for dist, idx in zip(distances, indexes):
            s_name = reverse_map[idx]
            s_score = all_scores[idx]
            s_example = all_examples[idx]
            s_task = all_tasks[idx]
            if s_name not in seen_strategies:
                seen_strategies.append(s_name)
                s_info = self.strategies_dict[s_name]
                retrieved_strategies[s_name] = {
                    "definition": s_info["definition"],
                    "task": s_task,
                    "example": s_example,
                    "score": s_score,
                }
            else:
                prev_score = retrieved_strategies[s_name]["score"]
                retrieved_strategies[s_name]["score"] = (prev_score + s_score) / 2
                if prev_score < s_score:
                    retrieved_strategies[s_name]["example"] = s_example
                    retrieved_strategies[s_name]["task"] = s_task
            # Stop if we've collected k unique strategies
            if len(list(retrieved_strategies.keys())) >= strategy_selection_i + 1:
                break

        return {seen_strategies[strategy_selection_i]: retrieved_strategies[seen_strategies[strategy_selection_i]]}

    def get_embedding(self, task):
        try:
            endpoint = "clip_gen"
            payload = {"text": task}
            response = requests.get(f"http://localhost:{commons.utils.PORT}/{endpoint}", json=payload)
            encoding = response.json()

            embedding = np.array(encoding).astype(np.float32)
            # Shape: (512, )

            return embedding

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        except requests.exceptions.ConnectionError:
            logging.error("Error: Could not connect to the server.")
        except requests.exceptions.Timeout:
            logging.error("Error: Request timed out.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
        except ValueError:
            logging.error("Error: Response is not valid JSON.")
        except Exception as e:
            logging.error(e)

    def get_all_in_txt(self):
        if not self.strategies_dict:
            return ""

        existing_strategies_str = ""
        for strategy_i, strategy_name in enumerate(self.strategies_dict):
            existing_strategies_str += f"{strategy_i + 1}. {strategy_name}\n"

        return existing_strategies_str

    def merge(self, library2):
        for s_name, s_info in library2.strategies_dict.items():
            for task_i, task in enumerate(s_info["tasks"]):
                if s_name not in self.strategies_dict or task not in self.strategies_dict[s_name]["tasks"]:
                    strategy = {
                        "strategy": s_name,
                        "definition": s_info["definition"],
                        "task": task,
                        "example": s_info["examples"][task_i],
                        "embedding": s_info["tasks_embeddings"][task_i],
                        "score": s_info["scores"][task_i],
                    }
                    self.add(strategy, is_from_attack=True, if_notify=True)

    def save(self, filename, save_dir=LOG_DIR):
        strategy_library_json = {
            s_name: {
                "definition": s_info["definition"],
                "source": s_info["source"],
                "tasks": s_info["tasks"],
                "examples": s_info["examples"],
            }
            for s_name, s_info in self.strategies_dict.items()
        }
        json_filepath = os.path.join(save_dir, f"{filename}.json")
        pkl_filepath = os.path.join(save_dir, f"{filename}.pkl")

        try:
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(strategy_library_json, f, indent=4)
            logger.info(
                f"Strategy library has been saved to {json_filepath} with {len(strategy_library_json)} strategies."
            )
        except Exception as e:
            logger.error(f"Saving strategy library to {json_filepath} failed: {e}")

        try:
            with open(pkl_filepath, "wb") as f:
                pickle.dump(self.strategies_dict, f)
            logger.info(
                f"Strategy library has been saved to {pkl_filepath} with {len(self.strategies_dict)} strategies."
            )
        except Exception as e:
            logger.error(f"Saving strategy library to {pkl_filepath} failed: {e}")

    def load(self, filename, library_dir=LOG_DIR):
        pkl_filepath = os.path.join(library_dir, f"{filename}.pkl")
        try:
            with open(pkl_filepath, "rb") as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    raise ValueError("Loaded object is not a dictionary.")
                self.strategies_dict = data
                return True
        except Exception as e:
            logger.error(f"Error when loading strategy dict: {e}")
            return False
