from obvs.lenses import TokenIdentity
from obvs.vis import create_heatmap, plot_surprisal

from datasets import load_dataset

dataset = load_dataset('oscar-corpus/OSCAR-2201', 'en', split='train', streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)

samples = []
for example in shuffled_dataset.take(5):
    samples.append(example['text'])


model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma": "google/gemma-2b"
}


# Trim the samples to the first 300 characters
samples = [sample[:1000] for sample in samples]

# Make sure it ends on a space
samples = [sample[:sample.rfind(' ')] for sample in samples]

# Strip the spaces
samples = [sample.strip() for sample in samples]

surprisals = []
model_name = "gpt2"
ti = TokenIdentity("", model_names[model_name])
for prompt in samples:
    ti.filename = f"token_identity_{model_name}_{prompt.replace(' ', '')[:10]}"
    ti.patchscope.source.prompt = prompt
    ti.run(
        range(ti.patchscope.n_layers_source),
        # range(ti.patchscope.n_layers_target)
    ).compute_surprisal().visualize()
    surprisals.append(ti.surprisal)

# Average the surprisals, calculate the standard deviation and plot with plotly
import numpy as np

if len(surprisals[0].shape) == 1:
    # Its a single set of layers
    mean_surprisal = np.mean(surprisals, axis=0)
    std_surprisal = np.std(surprisals, axis=0)

    fig = plot_surprisal(ti.source_layers, mean_surprisal, std_surprisal, "Surprisal of the first 1000 characters of 10 random samples from the OSCAR corpus")
    fig.write_html(f"mean_surprisal_heatmap_{model_names[model_name]}_{len(samples)}_samples.html")
    fig.show()

elif len(surprisals[0].shape) == 2:
    # Its a set of layers for each token, meaning a heatmap. We dont botther with the std
    mean_surprisal = np.mean(surprisals, axis=0)

    fig = create_heatmap(ti.source_layers, ti.target_layers, mean_surprisal, "Surprisal of the first 1000 characters of 10 random samples from the OSCAR corpus")
    fig.write_html(f"mean_surprisal_heatmap_{model_names[model_name]}_{len(samples)}_samples.html")
    fig.show()
