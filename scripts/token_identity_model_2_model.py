from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import typer
from tqdm import tqdm

from obvspython.logging import logger
from obvspython.patchscope import Patchscope, SourceContext, TargetContext
from obvspython.vis import create_heatmap

app = typer.Typer()


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    # "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    # "gpt2": "gpt2",
    "gptj": "EleutherAI/gpt-j-6B",
}


def run_over_all_layers(patchscope, target_tokens, values):
    source_layers = list(range(patchscope.n_layers_source))
    target_layers = list(range(patchscope.n_layers_target))
    iterations = len(source_layers) * len(target_layers)
    with tqdm(total=iterations) as pbar:
        for i in source_layers:
            for j in target_layers:
                patchscope.source.layer = i
                patchscope.target.layer = j

                patchscope.run()
                logger.info(
                    f"Source Layer-{i}, Target Layer-{j}: "
                    + "".join(patchscope.full_output_words()[len(patchscope.target_tokens) - 2 :]),
                )

                probs = patchscope.probabilities()[-1]
                values[i, j] = patchscope.compute_surprisal(probs, target_tokens)

                pbar.update(1)
    return source_layers, target_layers, values


@app.command()
def main(
    word: str = typer.Argument("USA", help="The word to generate a definition for."),
    source_model: str = "gpt2",
    target_model: str = "gpt2",
    prompt: str = typer.Option(
        "I went to the store but I didn't have any cash, so I had to use the ATM. Thankfully, this is the USA so I found one easy.",
        help="Must contain X, which will be replaced with the word",
    ),
):
    print(f"Generating definition for word: {word} using models {source_model} and {target_model}.")
    if source_model in model_names:
        source_model = model_names[source_model]
    if target_model in model_names:
        target_model = model_names[target_model]

    source_model_name = source_model.replace("/", "-")
    target_model_name = target_model.replace("/", "-")
    filename = f"{source_model_name}_2_{target_model_name}_{word}"

    # prompt = "For a long time, the largest and most famous building in New York was"
    prompt = "I went to the store but I didn't have any cash, so I had to use the ATM. Thankfully, this is the USA so I found one easy."
    # Setup source and target context with the simplest configuration
    source_context = SourceContext(
        prompt=prompt,  # Example input text
        model_name=source_model,  # Model name
        position=-1,
        device="cuda:0",
    )

    target_context = TargetContext.from_source(source_context)
    target_context.model_name = target_model
    target_context.device = "cuda:1"
    target_context.prompt = (
        "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; X is"
    )
    target_context.max_new_tokens = 1
    patchscope = Patchscope(source=source_context, target=target_context)

    target_word = "USA"
    patchscope.source.position, target_tokens = patchscope.source_position_tokens(target_word)
    patchscope.target.position, _ = patchscope.target_position_tokens("X")
    assert (
        patchscope.source_words[patchscope.source.position].strip() == target_word
    ), patchscope.source_words[patchscope.source.position]
    assert (
        patchscope.target_words[patchscope.target.position].strip() == "X"
    ), patchscope.target_words[patchscope.target.position]

    if Path(f"scripts/{filename}.npy").exists():
        values = np.load(f"scripts/{filename}.npy")
    else:
        values = np.zeros((patchscope.n_layers_source, patchscope.n_layers_target))

    start = time.time()
    source_layers, target_layers, values = run_over_all_layers(patchscope, target_tokens, values)
    print(f"Elapsed time: {time.time() - start:.2f}s. Layers: {source_layers}, {target_layers}")

    # Save the values to a file
    np.save(f"scripts/{filename}.npy", values)

    fig = create_heatmap(source_layers, target_layers, values)
    fig.update_layout(
        title="Token Identity: Surprisal by Layer",
        xaxis_title="Target Layer",
        yaxis_title="Source Layer",
    )

    source_model = source_model.replace("/", "-")
    target_model = target_model.replace("/", "-")
    filename = f"{source_model}_2_{target_model}_{word}"

    # Save as png
    fig.write_image(f"scripts/{filename}.png")
    fig.show()


if __name__ == "__main__":
    app()
