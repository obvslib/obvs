from __future__ import annotations

import plotly.graph_objects as go


def create_heatmap(model_1_layers, model_2_layers, values):
    """
    Create a heatmap of the values between the layers of two models.

    Args:
        model_1_layers (list): The layers of the first model.
        model_2_layers (list): The layers of the second model.
        values (list): The values to be plotted.

    Returns:
        None
    """
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=model_2_layers,  # columns
            y=model_1_layers,  # rows
            colorscale="Viridis",  # or any other colorscale
        ),
    )

    # Update the layout of the figure
    fig.update_layout(
        title="Layer by Layer Comparison between Two Models",
        xaxis_title="Model 2 Layers",
        yaxis_title="Model 1 Layers",
    )

    return fig
