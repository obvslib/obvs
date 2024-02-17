from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
llama = "nickypro/tinyllama-110M"
model = AutoModelForCausalLM.from_pretrained(llama)
tokenizer = AutoTokenizer.from_pretrained(llama)

# Encode the prompt text into tokens
prompt = "The quick brown fox jumps over the lazy"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
prompt_length = input_ids.shape[-1]

# Perform a forward pass to get model's output
outputs = model(input_ids)

# Extract logits
logits = outputs.logits

# Now, to generate text from the model, you can use:
generated_output = model.generate(input_ids, max_length=prompt_length + 10)
generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

# Print generated text
print(generated_text)
