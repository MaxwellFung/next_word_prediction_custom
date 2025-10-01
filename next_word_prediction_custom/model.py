from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT2:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()

    def predict_next(self, text, k):
        # Encode
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits  # modern API

        # Get probs
        probs = logits[0, -1, :].softmax(dim=-1)
        topk = torch.topk(probs, k)

        words = [self.tokenizer.decode([idx]) for idx in topk.indices]
        probs = topk.values.tolist()

        return words, probs
