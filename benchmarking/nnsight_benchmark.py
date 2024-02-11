from nnsight import LanguageModel

llama = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = LanguageModel(llama)

prompt = "The quick brown fox jumps over the lazy"

with model.generate(max_new_tokens=10) as generator:
    with generator.invoke(prompt) as _:
        pass


print(generator.output())
