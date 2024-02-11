from nnsight import LanguageModel

llama = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = LanguageModel(llama)

prompt = "The quick brown fox jumps over the lazy"

with model.invoke(prompt) as invoker:
    pass


output = model.tokenizer.decode(invoker.output[0].argmax(-1).squeeze())
print(output)
