from nnsight import LanguageModel

llama = "nickypro/tinyllama-110M"
model = LanguageModel(llama)

prompt = "The quick brown fox jumps over the lazy"

with model.generate(max_new_tokens=10) as generator:
    with generator.invoke(prompt) as _:
        pass


output = model.tokenizer.decode(generator.output[0])
print(output)
