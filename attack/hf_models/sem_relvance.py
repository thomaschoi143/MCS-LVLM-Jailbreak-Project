from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR")


class SemRelvance:
    def __init__(self):
        # roberta-base, princeton-nlp/sup-simcse-roberta-large, sentence-transformers/paraphrase-MiniLM-L12-v2
        self.model_name = "sup_simcse_roberta_large"

        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        print("Loading model: {}".format(self.model_name))
        model_path = os.path.join(MODEL_DIR, self.model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(model_path)

        return self.model, self.tokenizer

    def encode_text(self, text):
        # Use tokenizer to convert text to input tokens
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        max_tokens = 512
        tokens = tokens[:, :max_tokens]
        # Get the last hidden state from the model
        with torch.no_grad():
            output = self.model(tokens)
        # Extract the last hidden state
        text_representation = output.last_hidden_state.mean(dim=1).squeeze()
        return text_representation

    def compute_similarity(self, text1, text2):
        # Encode the text into embeddings
        representation1 = self.encode_text(text1)
        representation2 = self.encode_text(text2)

        # Compute the cosine similarity between the two vectors
        similarity = torch.nn.functional.cosine_similarity(representation1, representation2, dim=0)
        return similarity.item()
