from __future__ import annotations

import transformer_lens

llama = "nickypro/tinyllama-110M"
model = transformer_lens.HookedTransformer.from_pretrained(llama, device="cpu")

prompt = "The quick brown fox jumps over the lazy"

logits, activations = model.run_with_cache(prompt)
