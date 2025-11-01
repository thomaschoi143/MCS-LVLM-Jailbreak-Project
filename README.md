# Master of Computer Science Research Project - MAPA: Multi-Turn Adaptive Prompting Attack on Large Vision-Language Models

This repository consists of the implementation of our method **MAPA** and the baselines. The code of the baselines are cloned from the official repository with some mandatory adaptation.

Here is the mapping between the attack method and folder name:
| Attack Method | Folder Name |
|----------------------|---------------------|
| **MAPA (ours)** | `attack` |
| Chain of Attack <sup>[1](#ref1)</sup> | `CoA` |
| FootInTheDoor <sup>[2](#ref2)</sup> | `footInTheDoor` |
| ActorAttack <sup>[3](#ref3)</sup> | `actorAttack` |
| Multi-Modal Linkage <sup>[4](#ref4)</sup> | `MML` |
| Visual Role-Play <sup>[5](#ref5)</sup> | `visualRolePlay` |

## How to run the **MAPA** project

1. Download the models from Hugging Face and save them under the `models` folder. The mapping between the model name and model folder name is conducted by the `model_name_to_path_type` function in `attack/hf_models/utils.py`.
2. Create a Python virtual environment and enable it.
3. Go into the `attack` folder, `cd attack`.
4. Install the dependencies according to `attack/requirements.txt`.
5. Run the model server. For example, `python server.py --lvlm_name=llava --load_clip --load_sd --load_sem_relevance --port=8000.`
6. Run the attack. For example, `python main.py --lvlm_name=llava --dataset=harmbench --task_i_start_from=0 --num_tasks=60 --experiment_name=llava_test --port=8000`.
7. We have also provided the slurms file to conduct a range of experiments on HPC, which are in the `slurms` folder under each of the attack method folder.

## How to run the baselines

Please refer to the individual `README` file under each attack method baseline folder.

## References

<a name="ref1">[1]</a> Xikang Yang et al. “Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for LLM”. In: CoRR abs/2405.05610 (2024).

<a name="ref2">[2]</a> Zixuan Weng et al. “Foot-In-The-Door: A Multi-turn Jailbreak for LLMs”. In: arXiv preprint arXiv:2502.19820 (2025).

<a name="ref3">[3]</a> Qibing Ren et al. “Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues”. In: CoRR abs/2410.10700 (2024).

<a name="ref4">[4]</a> Yu Wang et al. “Jailbreak Large Visual Language Models Through Multi-Modal Linkage”. In: arXiv preprint arXiv:2412.00473 (2024).

<a name="ref5">[5]</a> Siyuan Ma et al. “Visual-roleplay: Universal jailbreak attack on multimodal large language models via role-playing image character”. In: arXiv preprint arXiv:2405.20773 (2024).
