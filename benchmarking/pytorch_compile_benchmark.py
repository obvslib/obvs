from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
llama = "nickypro/tinyllama-110M"
model = AutoModelForCausalLM.from_pretrained(llama)
tokenizer = AutoTokenizer.from_pretrained(llama)

# Encode the prompt text into tokens
prompt = "The quick brown fox jumps over the lazy"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
prompt_length = input_ids.shape[-1]


model = model.to(device)
input_ids = input_ids.to(device)

# Perform a forward pass to get model's output
outputs = model(input_ids)

# Extract logits
logits = outputs.logits


@torch.compile
def generate(input_ids):
    outputs = model.generate(input_ids, max_length=prompt_length + 10)
    return outputs


optimized_generate = generate(input_ids)
