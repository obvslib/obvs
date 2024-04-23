from __future__ import annotations

import json
import random
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from tqdm import tqdm
from transformers import AutoConfig

from obvs.lenses import ClassicLogitLens, PatchscopeLogitLens, TokenIdentity
from obvs.patchscope import ModelLoader

torch.cuda.empty_cache()

model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma": "google/gemma-2b",
}

# Construct the path to the file
file_path = "data/processed/sentences.json"

# Open and read the file
with open(file_path) as file:
    data = json.load(file)

# Define the device and model name
device = "gpu"
model_name = "gpt2"
# Load the model
model = ModelLoader.load(model_names[model_name], device)
# Load the model configuration
config = AutoConfig.from_pretrained(model_names[model_name])
# Get the number of layers
num_layers = config.num_hidden_layers
# Define the layers
layers = list(range(0, num_layers))
# Define the number of sentences to run
test_amount = 100
# Initialize the arrays to store the surprisal values
surprisals_log_classic_ll = np.zeros((test_amount, num_layers))
surprisals_log_patchscope_ll = np.zeros((test_amount, num_layers))
surprisals_log_ti = np.zeros((test_amount, num_layers))
warnings.filterwarnings("ignore")  # Ignore all warnings

# Loop through the data
for index, (sentence, _split) in enumerate(tqdm(data[0:test_amount])):
    # Token Identity pre-processing
    # Truncate sentence for token identity
    truncated_sentence = sentence
    # Make sure it ends on a space
    truncated_sentence = truncated_sentence[: truncated_sentence.rfind(" ")]
    # Strip the spaces
    truncated_sentence = truncated_sentence.strip()

    # Tokenize the sentence
    tokenized_sentence = model.tokenizer.encode(sentence)
    # Get a random position in the sentence
    rand_position = random.randrange(0, len(tokenized_sentence) - 2)

    # Initialize the Token Identity Lens
    ti = TokenIdentity("", model_names[model_name], device="cpu")
    source_layers = range(ti._patchscope.n_layers_source)
    target_layers = range(ti._patchscope.n_layers_target)
    full = False
    ti._patchscope.source.prompt = truncated_sentence
    ti.run_and_compute(
        source_layers=source_layers,
        target_layers=target_layers if full else None,
    )
    surprisals_log_ti[index, :] = ti.surprisal

    # Initialize the Classic and Patchscope Logit Lenses
    ClassicLL = ClassicLogitLens(model_names[model_name], sentence, "auto", layers, sentence)
    PatchscopeLL = PatchscopeLogitLens(model_names[model_name], sentence, "auto", layers, sentence)

    # Run the Classic and Patchscope Logit Lenses
    ClassicLL.run(sentence, layers)
    PatchscopeLL.run(rand_position)

    # Compute the surprisal values at the target position
    ClassicLL.target_token_id = tokenized_sentence[rand_position + 1]
    ClassicLL.target_token_position = rand_position
    PatchscopeLL.target_token_id = tokenized_sentence[rand_position + 1]
    PatchscopeLL.target_token_position = rand_position
    surprisals_across_layers_classic_LL = ClassicLL.compute_surprisal_at_position()
    surprisals_across_layers_patchscope_LL = PatchscopeLL.compute_surprisal_at_position()

    # Store the surprisal values
    surprisals_log_classic_ll[index, :] = surprisals_across_layers_classic_LL
    surprisals_log_patchscope_ll[index, :] = surprisals_across_layers_patchscope_LL


def save_surprisals_to_csv(
    surprisals_log_classic_ll,
    surprisals_log_patchscope_ll,
    surprisals_log_ti,
    num_layers,
):
    df_classic = pd.DataFrame(
        surprisals_log_classic_ll,
        columns=[f"Layer {i}" for i in range(num_layers)],
    )
    df_patchscope = pd.DataFrame(
        surprisals_log_patchscope_ll,
        columns=[f"Layer {i}" for i in range(num_layers)],
    )
    df_ti = pd.DataFrame(
        surprisals_log_ti,
        columns=[f"Layer {i}" for i in range(num_layers)],
    )
    df_classic["Lens Type"] = "Classic"
    df_patchscope["Lens Type"] = "Patchscope"
    df_ti["Lens Type"] = "Token Identity"

    df_total = pd.concat([df_classic, df_patchscope, df_ti], axis=0)
    df_total.to_csv("surprisals_fig_2.csv", index=False)


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
    df_ti = df_total[df_total["Lens Type"] == "Token Identity"]

    # Average surprisal values across sentences for each layer
    classic_means = df_classic.drop(columns=["Lens Type"]).mean()
    patchscope_means = df_patchscope.drop(columns=["Lens Type"]).mean()
    ti_means = df_ti.drop(columns=["Lens Type"]).mean()

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

    trace_ti = go.Scatter(
        x=np.arange(len(ti_means)),
        y=ti_means,
        mode="lines+markers",
        name="Token Identity Lens",
        marker=dict(color="green"),
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
    fig = go.Figure(data=[trace_classic, trace_patchscope, trace_ti], layout=layout)

    # Save the plot
    fig.write_html(output_html_path)


save_surprisals_to_csv(
    surprisals_log_classic_ll,
    surprisals_log_patchscope_ll,
    surprisals_log_ti,
    num_layers,
)
plot_surprisals_and_save_fig("surprisals_fig_2.csv", "surprisal_plot.html")
