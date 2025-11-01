import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


class HuggingFaceLLM:
    def __init__(self, model_name: str, model_path: str, **kwargs):
        self.model_name = model_name
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", **kwargs
        )

    def generate(self, full_prompt, max_new_tokens, temperature, top_p):
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.llm.device)
        inputs_len = inputs["input_ids"].shape[1]

        if temperature > 0:
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        else:
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1,
                top_p=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        outputs.to("cpu")
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        return response
