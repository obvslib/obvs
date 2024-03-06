from __future__ import annotations

from datasets import load_dataset

from obvs.lenses import TokenIdentity

dataset = load_dataset("oscar-corpus/OSCAR-2201", "en", split="train", streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10)

samples = []
for example in shuffled_dataset.take(10):
    samples.append(example["text"])


# Trim the samples to the first 300 characters
samples = [sample[:1000] for sample in samples]

# Make sure it ends on a space
samples = [sample[: sample.rfind(" ")] for sample in samples]

# Strip the spaces
samples = [sample.strip() for sample in samples]

surprisals = []
for prompt in samples:
    ti = TokenIdentity(prompt)
    ti.run().compute_surprisal().visualize()
    surprisals.append(ti.surprisal)

# Average the surprisals, calculate the standard deviation and plot with plotly
import numpy as np
import plotly.graph_objects as go

mean_surprisal = np.mean(surprisals, axis=0)
std_surprisal = np.std(surprisals, axis=0)

fig = go.Figure(
    data=go.Scatter(
        x=ti.source_layers,
        y=mean_surprisal,
        mode="lines+markers",
        error_y=dict(
            type="data",
            array=std_surprisal,
            visible=True,
        ),
    ),
)

fig.update_layout(
    title="Surprisal of the first 1000 characters of 10 random samples from the OSCAR corpus",
    xaxis_title="Layer",
    yaxis_title="Surprisal",
)

fig.show()
