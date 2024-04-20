from __future__ import annotations

import json
import random
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from transformers import AutoConfig

from obvs.lenses import ClassicLogitLens, PatchscopeLogitLens
from obvs.patchscope import ModelLoader

# Construct the path to the file
file_path = "data/processed/sentences.json"

# Open and read the file
with open(file_path) as file:
    data = json.load(file)

device = "gpu"
model_name = "gpt2"
model = ModelLoader.load(model_name, device)
print(model.tokenizer)
# Two options - redo all of the preprocessing and follow the paper method exactly
# Or, skip it, shortcut to surprisal and precision calculations per sentence, see if the plot looks sensible.
# Option 2, then go back and do 1?
config = AutoConfig.from_pretrained("gpt2")
num_layers = config.num_hidden_layers
layers = list(range(0, num_layers))
test_amount = 100
surprisals_log_classic_ll = np.zeros((test_amount, num_layers))
surprisals_log_patchscope_ll = np.zeros((test_amount, num_layers))

warnings.filterwarnings("ignore")  # Ignore all warnings

for index, (sentence, _split) in enumerate(tqdm(data[0:test_amount])):
    tokenized_sentence = model.tokenizer.encode(sentence)
    rand_position = random.randrange(0, len(tokenized_sentence) - 2)

    ClassicLL = ClassicLogitLens(model_name, sentence, "auto")
    PatchscopeLL = PatchscopeLogitLens(model_name, sentence, "auto", layers, sentence)

    ClassicLL.run(sentence, layers)
    PatchscopeLL.run(rand_position)

    # Prep for surprisal calc
    ClassicLL.target_token_id = tokenized_sentence[rand_position + 1]
    ClassicLL.target_token_position = rand_position
    PatchscopeLL.target_token_id = tokenized_sentence[rand_position + 1]
    PatchscopeLL.target_token_position = rand_position
    surprisals_across_layers_classic_LL = ClassicLL.compute_surprisal_at_position()
    surprisals_across_layers_patchscope_LL = PatchscopeLL.compute_surprisal_at_position()

    # Logging the surprisal values
    surprisals_log_classic_ll[index, :] = surprisals_across_layers_classic_LL
    surprisals_log_patchscope_ll[index, :] = surprisals_across_layers_patchscope_LL


def save_surprisals_to_csv(surprisals_log_classic_ll, surprisals_log_patchscope_ll, num_layers):
    df_classic = pd.DataFrame(
        surprisals_log_classic_ll,
        columns=[f"Layer {i}" for i in range(num_layers)],
    )
    df_patchscope = pd.DataFrame(
        surprisals_log_patchscope_ll,
        columns=[f"Layer {i}" for i in range(num_layers)],
    )
    df_classic["Lens Type"] = "Classic"
    df_patchscope["Lens Type"] = "Patchscope"
    df_total = pd.concat([df_classic, df_patchscope], axis=0)
    df_total.to_csv("surprisals.csv", index=False)


def plot_surprisals_and_save_fig(input_csv_path, output_html_path):
    """
    Plots surprisal values as a line plot across layers for different lenses and saves the plot to an HTML file.

    Parameters:
    - csv_filepath: str, the path to the CSV file containing surprisal data.
    - output_html_filepath: str, the path where the HTML file will be saved.
    """
    # Read data from CSV
    df_total = pd.read_csv(input_csv_path)

    # Separate data for plotting
    df_classic = df_total[df_total["Lens Type"] == "Classic"]
    df_patchscope = df_total[df_total["Lens Type"] == "Patchscope"]

    # Average surprisal values across sentences for each layer
    classic_means = df_classic.drop(columns=["Lens Type"]).mean()
    patchscope_means = df_patchscope.drop(columns=["Lens Type"]).mean()

    # Create line traces
    trace_classic = go.Scatter(
        x=np.arange(len(classic_means)),
        y=classic_means,
        mode="lines+markers",
        name="Classic Lens",
        marker=dict(color="red"),
        line=dict(width=2),
    )
    trace_patchscope = go.Scatter(
        x=np.arange(len(patchscope_means)),
        y=patchscope_means,
        mode="lines+markers",
        name="Patchscope Lens",
        marker=dict(color="blue"),
        line=dict(width=2),
    )

    # Configure the layout of the plot
    layout = go.Layout(
        title="Average Surprisal Values Across Layers",
        xaxis_title="Layer Index",
        yaxis_title="Average Surprisal",
        legend_title="Lens Type",
        xaxis=dict(tickmode="linear"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Create figure with data and layout
    fig = go.Figure(data=[trace_classic, trace_patchscope], layout=layout)

    # Save the plot
    fig.write_html(output_html_path)


save_surprisals_to_csv(surprisals_log_classic_ll, surprisals_log_patchscope_ll, num_layers)
plot_surprisals_and_save_fig("surprisals.csv", "surprisal_plot.html")
