import transformer_lens

llama = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = transformer_lens.HookedTransformer.from_pretrained(llama, device="cpu")

prompt = "The quick brown fox jumps over the lazy"

logits, activations = model.run_with_cache(prompt)
