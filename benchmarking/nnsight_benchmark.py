from __future__ import annotations

from custom_timeit import custom_timeit
from nnsight import LanguageModel


def setup(model_id: str) -> LanguageModel:
    """Setup model and return it"""
    return LanguageModel(model_id)


def generate(prompt: str, model: LanguageModel) -> str:
    """Generate new text from the given prompt and model.
    Return the generated text as a string"""

    with model.generate(max_new_tokens=20) as generator:
        with generator.invoke(prompt) as _:
            pass

    # return generated text
    return model.tokenizer.decode(generator.output[0])


def run_benchmark(prompt: str, model_id: str) -> tuple[str, float]:
    """Setup model and run text generation on the prompt, estimating the average run time.
    Return the generated text and the average runtime"""

    model = setup(model_id)
    return custom_timeit(generate, 3, 10, prompt, model)
