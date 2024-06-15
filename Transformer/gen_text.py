import torch
import transformers

# Set the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# Load the fine-tuned model and tokenizer
output_dir = "./stage_l3/Transformer/output/gpt2-finetuned"
tokenizer = transformers.GPT2Tokenizer.from_pretrained(output_dir)
fine_tuned_model = transformers.GPT2LMHeadModel.from_pretrained(output_dir)
fine_tuned_model = fine_tuned_model.to(device)

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
basic_model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium')
basic_model = basic_model.to(device)


def generate_text(prompt, model, tokenizer, device, max_length=50, num_return_sequences=1):
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,           # Use sampling instead of greedy decoding
            top_k=50,                 # Keep only top k tokens with highest probability
            top_p=0.95,               # Nucleus sampling: keep the top 95% cumulative probability
            temperature=0.7           # Lower temperature makes the model more deterministic
        )
    
    # Decode the generated text
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return generated_texts



print("Use fine_tuned_model ? [y/n]")
use_model_name = input()


# Prompt the user for input
print("Prompt : ")
user_prompt = input()


model = fine_tuned_model
if use_model_name == "n":
    model = basic_model

# Generate text with the fine-tuned model
generated_texts = generate_text(user_prompt, model, tokenizer, device)

# Display the generated text
print("Generated Text:")
for idx, text in enumerate(generated_texts):
    print(f"{idx + 1}: {text}\n")
