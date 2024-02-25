from __future__ import annotations

import random
import string
import time

import numpy as np
import torch
import typer
from tqdm import tqdm
from pathlib import Path

from obvspython.logging import logger
from obvspython.patchscope import Patchscope, SourceContext, TargetContext
from obvspython.vis import create_heatmap, plot_surprisal

app = typer.Typer()


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma": "google/gemma-2b"
}


def run_over_all_layers(patchscope, target_tokens, values):
    source_layers = list(range(patchscope.n_layers))
    target_layers = list(range(patchscope.n_layers))
    iterations = len(source_layers) * len(target_layers)

    # with tqdm(total=iterations) as pbar:
    #     outputs = patchscope.over_pairs(source_layers, target_layers)
    #     pbar.update(1)

    with tqdm(total=iterations) as pbar:
        outputs = patchscope.over(source_layers, target_layers)
        pbar.update(1)

    logger.info("Computing surprisal")
    target_output = 0

    # for i in source_layers:
    #     # Get the output of the run
    #     probs = torch.softmax(outputs[i][target_output], dim=-1)
    #     values[i] = patchscope.compute_surprisal(probs[-1], target_tokens)

    for i in source_layers:
        for j in target_layers:
            # Get the output of the run
            probs = torch.softmax(outputs[i][j][target_output].value, dim=-1)
            values[i, j] = patchscope.compute_surprisal(probs[-1], target_tokens)
    logger.info("Done")

    return source_layers, target_layers, values, outputs


def upate_saved_values(values):
    # Save the values to a file
    np.save("scripts/values.npy", values)


@app.command()
def main(
    word: str = typer.Argument(" boat", help="The expected next token."),
    model: str = "gpt2",
    prompt: str = typer.Option(
        "if its on the road, its a car. if its in the air, its a plane. if its on the sea, its a",
        help="Source Prompt",
    ),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating definition for word: {word} using model: {model}")
    if model in model_names:
        model = model_names[model]

    model_name = model.replace("/", "-")
    filename = f"{model_name}_{word}"

    # Setup source and target context with the simplest configuration
    source_context = SourceContext(
        prompt=prompt,  # Example input text
        model_name=model,  # Model name
        position=-1,
        device=device,
    )

    target_context = TargetContext.from_source(source_context)
    target_context.prompt = (
        "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"
    )
    target_context.max_new_tokens = 1
    patchscope = Patchscope(source=source_context, target=target_context)

    try:
        patchscope.source.position, target_tokens = patchscope.source_position_tokens(word)
        patchscope.target.position, _ = patchscope.target_position_tokens("X")

        assert (
            patchscope.source_words[patchscope.source.position].strip() == word
        ), patchscope.source_words[patchscope.source.position]
        assert (
            patchscope.target_words[patchscope.target.position].strip() == "X"
        ), patchscope.target_words[patchscope.target.position]

    except ValueError:
        target_tokens = patchscope.tokenizer.encode(" boat", add_special_tokens=False)
        patchscope.source.position = -1
        patchscope.target.position = -1

    if Path(f"scripts/{filename}.npy").exists():
        values = np.load(f"scripts/{filename}.npy")
    else:
        values = np.zeros((patchscope.n_layers_source, patchscope.n_layers_target))

    start = time.time()
    source_layers, target_layers, values, outputs = run_over_all_layers(patchscope, target_tokens, values)
    print(f"Elapsed time: {time.time() - start:.2f}s. Layers: {source_layers}, {target_layers}")

    # Save the values to a file
    np.save(f"scripts/{filename}.npy", values)

    # fig = plot_surprisal(source_layers, [value[0] for value in values], title=f"Token Identity: Surprisal by Layer {model_name}")
    # fig.write_image(f"scripts/{filename}.png")
    # fig.show()

    fig = create_heatmap(source_layers, target_layers, values, title=f"Token Identity: Surprisal by Layer {model_name} {prompt}")
    # Save as png
    fig.write_image(f"scripts/{filename}.png")
    fig.show()


if __name__ == "__main__":
    app()
