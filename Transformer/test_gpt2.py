import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

# import logging
# logging.getLogger().setLevel(logging.CRITICAL)

# import warnings
# warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

def predict_next_word(input_text, model, tokenizer, device, top_k=10):
    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    next_token_logits = logits[:, -1, :]

    probabilities = torch.softmax(next_token_logits, dim=-1).squeeze().cpu().numpy()

    token_ids = np.arange(len(probabilities))
    probability_distribution = list(zip(token_ids, probabilities))

    probability_distribution = [(tokenizer.decode([token_id]), prob) for token_id, prob in probability_distribution]
    probability_distribution = sorted(probability_distribution, key=lambda x: x[1], reverse=True)

    return probability_distribution[:top_k]

user_input = input("Prompt : ")

predictions = predict_next_word(user_input, model, tokenizer, device)

print(f"Top {len(predictions)} next words :")
for word, prob in predictions:
    print(f"{word}: {prob:.6f}")
