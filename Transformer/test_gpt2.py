import torch
import transformers
import numpy as np

# import logging
# logging.getLogger().setLevel(logging.CRITICAL)

# import warnings
# warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

# def predict_next_word(input_text, model, tokenizer, device, top_k=10):
#     inputs = tokenizer(input_text, return_tensors='pt')
#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits

#     next_token_logits = logits[:, -1, :]

#     probabilities = torch.softmax(next_token_logits, dim=-1).squeeze().cpu().numpy()

#     token_ids = np.arange(len(probabilities))
#     probability_distribution = list(zip(token_ids, probabilities))

#     probability_distribution = [(tokenizer.decode([token_id]), prob) for token_id, prob in probability_distribution]
#     probability_distribution = sorted(probability_distribution, key=lambda x: x[1], reverse=True)

#     return probability_distribution[:top_k]

# print("Prompt : ")
# user_input = input()

# predictions = predict_next_word(user_input, model, tokenizer, device)

# print(f"Top {len(predictions)} next words :")
# for word, prob in predictions:
#     print(f"{word}: {prob:.6f}")



def load_dataset(file_path, tokenizer, block_size=128):
    # Function to load the dataset and prepare it for training
    dataset = transformers.TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

def fine_tune_gpt2(dataset, model, tokenizer, output_dir, epochs=3, batch_size=4):
    # Create a data collator
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We are not using masked language modeling
    )

    # Define training arguments
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=200,
    )

    # Initialize the Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# File path to your training data
train_file_path = "./Transformer/dataset.txt"

# Load the dataset
dataset = load_dataset(train_file_path, tokenizer)

# Output directory to save the model
output_dir = "./Transformer/output/gpt2-finetuned"

# Fine-tune the model
fine_tune_gpt2(dataset, model, tokenizer, output_dir)