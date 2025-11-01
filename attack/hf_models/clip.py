from transformers import CLIPTokenizer, CLIPModel
import torch
import gc


class HuggingFaceCLIP:
    def __init__(self, model_path):
        self.model = CLIPModel.from_pretrained(model_path, device_map="auto")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            # shape: [batch_size, embedding_dim]
            # Normalize embeddings for better retrieval performance
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        embedding = features.cpu()[0].tolist()

        del inputs, features
        torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Run garbage collection on CPU

        return embedding
