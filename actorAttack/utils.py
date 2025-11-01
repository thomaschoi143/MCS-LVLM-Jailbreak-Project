import os
import json
import time
from openai import OpenAI
from typing import Union, List
from dotenv import load_dotenv
from fastchat.model import get_conversation_template
import config

load_dotenv()
import requests


def get_env_variable(var_name):
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)


CALL_SLEEP = 1
clients = {}


def initialize_clients():
    """Dynamically initialize available clients based on environment variables."""
    try:
        gpt_api_key = get_env_variable("GPT_API_KEY")
        gpt_base_url = get_env_variable("BASE_URL_GPT")
        if gpt_api_key and gpt_base_url:
            clients["gpt"] = OpenAI(base_url=gpt_base_url, api_key=gpt_api_key)

        claude_api_key = get_env_variable("CLAUDE_API_KEY")
        claude_base_url = get_env_variable("BASE_URL_CLAUDE")
        if claude_api_key and claude_base_url:
            clients["claude"] = OpenAI(base_url=claude_base_url, api_key=claude_api_key)

        deepseek_api_key = get_env_variable("DEEPSEEK_API_KEY")
        deepseek_base_url = get_env_variable("BASE_URL_DEEPSEEK")
        if deepseek_api_key and deepseek_base_url:
            clients["deepseek"] = OpenAI(base_url=deepseek_base_url, api_key=deepseek_api_key)

        deepinfra_api_key = get_env_variable("DEEPINFRA_API_KEY")
        deepinfra_base_url = get_env_variable("BASE_URL_DEEPINFRA")
        if deepinfra_api_key and deepinfra_base_url:
            clients["deepinfra"] = OpenAI(base_url=deepinfra_base_url, api_key=deepinfra_api_key)

        if not clients:
            print("No valid API credentials found. Exiting.")
            exit(1)

    except Exception as e:
        print(f"Error during client initialization: {e}")
        exit(1)


# initialize_clients()


def get_client(model_name):
    return None
    """Select appropriate client based on the given model name."""
    if "gpt" in model_name or "o1-" in model_name:
        client = clients.get("gpt")
    elif "claude" in model_name:
        client = clients.get("claude")
    elif "deepseek" in model_name:
        client = clients.get("deepseek")
    elif any(keyword in model_name.lower() for keyword in ["llama", "qwen", "mistral", "microsoft"]):
        client = clients.get("deepinfra")
    else:
        raise ValueError(f"Unsupported or unknown model name: {model_name}")

    if not client:
        raise ValueError(f"{model_name} client is not available.")
    return client


def read_prompt_from_file(filename):
    with open(filename, "r") as file:
        prompt = file.read()
    return prompt


def read_data_from_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def parse_json(output):
    try:
        output = "".join(output.splitlines())

        if "{" in output and "}" in output:
            start = output.index("{")
            end = output.rindex("}")
            output = output[start : end + 1]
        data = json.loads(output)

        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary.")

        return data
    except Exception as e:
        print(f"Failed to parse json in parse_json - {e}: {output}")

        return None


def check_file(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        raise IOError(f"File not found error: {file_path}.")


def gpt_call(client, query: Union[List, str], model_name="gpt-4o", temperature=0):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "user", "content": query}]
        # for _ in range(3):
        # try:
        #     if 'o1-' in model_name:
        #         completion = client.chat.completions.create(
        #             model=model_name,
        #             messages=messages
        #         )
        #     else:
        #         completion = client.chat.completions.create(
        #             model=model_name,
        #             messages=messages,
        #             temperature=temperature
        #         )
        #     resp = completion.choices[0].message.content
        #     return resp
        # except Exception as e:
        #     print(f"GPT_CALL Error: {model_name}:{e}")
        #     time.sleep(CALL_SLEEP)
        #     continue
    payload = dict(max_new_tokens=300, temperature=0.7, top_p=0.9)

    conv = get_conversation_template("mistral-instruct")
    for msg in messages:
        if msg["role"] == "user":
            conv.append_message(conv.roles[0], msg["content"])
        else:
            conv.append_message(conv.roles[1], msg["content"])

    try:
        endpoint = "llm_gen"
        payload["full_prompt"] = conv.get_prompt()
        payload["model_name"] = "mistral"
        response = requests.get(f"http://localhost:{config.port}/{endpoint}", json=payload)

        response_text = response.text
        response_text = response_text.encode("utf-8").decode("unicode_escape")

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
    return ""


def gpt_call_append(client, model_name, dialog_hist: List, query: str):
    dialog_hist.append({"role": "user", "content": query})
    resp = gpt_call(client, dialog_hist, model_name=model_name)
    dialog_hist.append({"role": "assistant", "content": resp})
    return resp, dialog_hist


def wrapper(response_text, system_prompt):
    user_prompt = f"[INPUT]: '{response_text}'"

    conv = get_conversation_template("mistral-instruct")
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    try:
        endpoint = "llm_gen"
        payload = dict(max_new_tokens=500, temperature=0.3, top_p=0.8)
        payload["full_prompt"] = conv.get_prompt()
        payload["model_name"] = "mistral_7b"
        response = requests.get(f"http://localhost:{config.port}/{endpoint}", json=payload)

        response_text = response.text
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

    response_text = response_text.encode("utf-8").decode("unicode_escape")
    if response_text.startswith('""'):
        response_text = response_text[1:]

    wrapped_json = parse_json(response_text)

    return wrapped_json


def get_victim_generation(dialog_hist: List, conv: List, query: str, images=None):
    dialog_hist.append({"role": "user", "content": query})
    conv.append({"role": "user", "content": [{"type": "text", "text": query}]})

    payload = dict(max_new_tokens=300, temperature=0.7, top_p=0.9)
    try:
        endpoint = "lvlm_gen"
        payload["conv"] = conv
        if images is None:
            print("LVLM generation is missing images.")
            return None
        files = None

        data = {"metadata": json.dumps(payload)}

        response = requests.post(f"http://localhost:{config.port}/{endpoint}", files=files, data=data)
        response_text = response.text

        conv.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        dialog_hist.append({"role": "assistant", "content": response_text})

        return response_text, dialog_hist, conv
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
