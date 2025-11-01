from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR")


class Toxigen:
    def __init__(self):
        # roberta-base, princeton-nlp/sup-simcse-roberta-large, sentence-transformers/paraphrase-MiniLM-L12-v2
        self.model_name = "toxigen_roberta"

        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        print("Loading model: {}".format(self.model_name))
        model_path = os.path.join(MODEL_DIR, self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        return self.model, self.tokenizer

    def get_toxicity(self, texts):
        try:
            # Use tokenizer to convert text to input tokens
            tokens = self.tokenizer(texts, return_tensors="pt", padding=True)
            # Get the last hidden state from the model
            max_tokens = 512
            tokens = {k: v[:, :max_tokens] for k, v in tokens.items()}

            with torch.no_grad():
                output = self.model(**tokens).logits
            # Extract the last hidden state
            prob = torch.nn.functional.softmax(output, dim=1)
            label = torch.argmax(prob, dim=1)
        except Exception as e:
            print(e)
            label = torch.zeros(len(texts))

        result = []

        for i in range(len(texts)):
            result.append({"text": texts[i], "type": str(label[i].item()), "toxicity": prob[i][1].item()})

        return result
