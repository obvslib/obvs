from __future__ import annotations

from datasets import load_dataset

from obvs.lenses import TokenIdentity
from obvs.vis import create_heatmap, plot_surprisal

model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma": "google/gemma-2b",
}


def main(model_name, n_samples=5, full=False):
    dataset = load_dataset("oscar-corpus/OSCAR-2201", "en", split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=43, buffer_size=n_samples)

    samples = []
    for example in shuffled_dataset.take(n_samples):
        samples.append(example["text"])

    # Trim the samples to the first 300 characters
    samples = [sample[:1000] for sample in samples]

    # Make sure it ends on a space
    samples = [sample[: sample.rfind(" ")] for sample in samples]

    # Strip the spaces
    samples = [sample.strip() for sample in samples]

    ti = TokenIdentity("", model_names[model_name], device="cpu")

    source_layers = range(ti._patchscope.n_layers_source)
    target_layers = range(ti._patchscope.n_layers_target)

    surprisals = []
    for prompt in samples:
        ti.filename = (
            f"{'full' if full else ''}token_identity_{model_name}_{prompt.replace(' ', '')[:10]}"
        )
        ti._patchscope.source.prompt = prompt
        ti.run_and_compute(
            source_layers=source_layers,
            target_layers=target_layers if full else None,
        ).visualize()
        surprisals.append(ti.surprisal)

    # Average the surprisals, calculate the standard deviation and plot with plotly
    import numpy as np

    if len(surprisals[0].shape) == 1:
        # Its a single set of layers
        mean_surprisal = np.mean(surprisals, axis=0)
        std_surprisal = np.std(surprisals, axis=0)

        fig = plot_surprisal(
            ti.source_layers,
            mean_surprisal,
            std_surprisal,
            f"{model_name} Surprisal of the first 1000 characters of {n_samples} random samples from the OSCAR corpus",
        )
        fig.write_html(f"mean_surprisal_heatmap_{model_name}_{len(samples)}_samples.html")
        fig.show()

    elif len(surprisals[0].shape) == 2:
        # Its a set of layers for each token, meaning a heatmap. We dont botther with the std
        mean_surprisal = np.mean(surprisals, axis=0)

        fig = create_heatmap(
            ti.source_layers,
            ti.target_layers,
            mean_surprisal,
            f"{model_name} Surprisal of the first 1000 characters of {n_samples} random samples from the OSCAR corpus",
        )
        fig.write_html(f"mean_surprisal_heatmap_{model_name}_{len(samples)}_samples.html")
        fig.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the surprisal of a set of samples using a model",
    )
    parser.add_argument("model_name", type=str, help="The name of the model to use")
    parser.add_argument("--n", type=int, default=5, help="The number of samples to average over")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Whether to run over all layers or pairs",
    )
    args = parser.parse_args()

    print(args.model_name, args.n, args.full)
    main(args.model_name, args.n, args.full)
