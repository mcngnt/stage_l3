import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

# Set the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

# Input text
input_text = "I drink"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors='pt')
inputs = {key: value.to(device) for key, value in inputs.items()}

# Get the model outputs (logits)
with torch.no_grad():
    outputs = model(**inputs)

# The logits for the next word prediction
logits = outputs.logits

# Select the logits for the last token in the input sequence
next_token_logits = logits[:, -1, :]

# Convert logits to probabilities
probabilities = torch.softmax(next_token_logits, dim=-1).squeeze().cpu().numpy()

# Get the token ids and corresponding probabilities
token_ids = np.arange(len(probabilities))
probability_distribution = list(zip(token_ids, probabilities))

# Convert token ids to words and sort by probability
probability_distribution = [(tokenizer.decode([token_id]), prob) for token_id, prob in probability_distribution]
probability_distribution = sorted(probability_distribution, key=lambda x: x[1], reverse=True)

# Display the top 10 probable next words
for word, prob in probability_distribution[:10]:
    print(f"{word}: {prob:.6f}")
