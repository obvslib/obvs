from __future__ import annotations

import torch
from custom_timeit import custom_timeit
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup(model_id: str, compile_model: bool = False) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup model and tokenizer and return them.
    If compile==True -> compile the model with torch.compile"""

    model = AutoModelForCausalLM.from_pretrained(model_id)
    if compile_model:
        model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def generate(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    """Generate new text from the given prompt and model.
    Return the generated text as a string"""

    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    prompt_length = input_tokens.shape[-1]

    # generate text
    generated_output = model.generate(input_tokens, max_length=prompt_length + 20)
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return generated_text


def run_benchmark(prompt: str, model_id: str, compile_model: bool = False) -> tuple[str, float]:
    """Setup model and run text generation on the prompt, estimating the average run time.
    Return the generated text and the average runtime"""

    model, tokenizer = setup(model_id, compile_model)
    return custom_timeit(generate, 3, 10, prompt, model, tokenizer)
