from .agent import Agent
from fastchat.model import get_conversation_template
from commons import parse_json_str
import re


class LLM(Agent):
    ENDPOINT = "llm_gen"

    def __init__(self, model_name: str, max_new_tokens: int, temperature: float, top_p: float):
        super().__init__(model_name, max_new_tokens, temperature, top_p)
        self.template_name = self.model_name_to_template(model_name)

    @staticmethod
    def model_name_to_template(name):
        look_up = {"mistral": "mistral-instruct"}
        if name in look_up:
            return look_up[name]

        return None

    def get_generation_single(self, system_prompt, user_prompt, condition, is_small_model=False, port=None):
        conv = get_conversation_template(self.template_name)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], condition)

        payload = dict(max_new_tokens=self.max_new_tokens, temperature=self.temperature, top_p=self.top_p)
        payload["full_prompt"] = conv.get_prompt()

        if is_small_model:
            payload["model_name"] = self.model_name + "_7b"
        else:
            payload["model_name"] = self.model_name

        response = self.request_to_server(self.ENDPOINT, port=port, json=payload)

        return response

    def wrapper(self, system_prompt, response_text, condition=None):
        user_prompt = f"[INPUT]: '{response_text}'"
        wrapped_txt = self.get_generation_single(system_prompt, user_prompt, condition, is_small_model=True)
        wrapped_json = parse_json_str(wrapped_txt, prefix=condition)

        return wrapped_json

    def token_wrapping_extractor(self, text, required_tokens):
        results = {}
        for token in required_tokens:
            upper_token = token.upper()
            pattern = rf"\[{upper_token}_START\](.*?)\[{upper_token}_END\]"
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                return None
            results[token.lower()] = match.group(1).strip().replace("\n", "")
        return results
