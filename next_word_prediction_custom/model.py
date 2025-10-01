from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT2:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()

    def predict_next(self, text, k):
        # Encode input
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(device)

        # Forward pass (deterministic with seeds set)
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits

        # Softmax probabilities
        probs = logits[0, -1, :].softmax(dim=-1)

        # Deterministic top-k
        topk = torch.topk(probs, k)
        tokens = [self.tokenizer.decode([idx]) for idx in topk.indices.tolist()]
        values = topk.values.tolist()

        return tokens, values
