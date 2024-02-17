import timeit
import custom_timeit
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple


def setup(model_id: str, compile_model: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """ Setup model and tokenizer and return them.
        If compile==True -> compile the model with torch.compile"""

    model = AutoModelForCausalLM.from_pretrained(model_id)
    if compile_model:
        model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def generate(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    """ Generate new text from the given prompt and model.
        Return the generated text as a string """

    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    prompt_length = input_tokens.shape[-1]

    # generate text
    generated_output = model.generate(input_tokens, max_length=prompt_length + 10)
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    return generated_text


def run_benchmark(prompt: str, model_id: str, compile_model: bool = False) -> Tuple[str, float]:
    """ Setup model and run text generation on the prompt, estimating the average run time.
        Return the generated text and the average runtime """

    model, tokenizer = setup(model_id, compile_model)

    # create a Timer object for estimating the runtime
    timer = timeit.Timer(lambda: generate(prompt, model, tokenizer))
    total_runtime, generated_text = timer.timeit(number=5)
    return total_runtime / 5, generated_text
