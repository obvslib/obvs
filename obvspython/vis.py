from __future__ import annotations

import plotly.graph_objects as go


def create_heatmap(source_layers, target_layers, values, title="Layer by Layer Comparison between Two Models") -> go.Figure:
    """
    Create a heatmap of the values between the layers of two models.

    Args:
        model_1_layers (list): The layers of the first model.
        model_2_layers (list): The layers of the second model.
        values (list): The values to be plotted.

    Returns:
        go.Figure: The heatmap figure.
    """
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=target_layers,  # columns
            y=source_layers,  # rows
            colorscale="Viridis",  # or any other colorscale
        ),
    )

    # Update the layout of the figure
    fig.update_layout(
        title=title,
        xaxis_title="Target Layers",
        yaxis_title="Source Layers",
    )

    return fig


def plot_surprisal(layers, values, title="Surprisal") -> go.Figure:
    """
    Create a line plot of the surprisal values.

    Args:
        surprisal_values (list): The surprisal values.
        layers (list): The layers.
        title (str, optional): The title of the plot. Defaults to "Surprisal".

    Returns:
        go.Figure: The bar plot figure.
    """
    # Create the scatter plot
    fig = go.Figure(
        data=go.Scatter(
            x=layers,
            y=values,
            mode="lines+markers",
        ),
    )

    # Pin the y-axis range to 0-15
    fig.update_yaxes(range=[0, 17])

    # Update the layout of the figure
    fig.update_layout(
        title=title,
        xaxis_title="Layers",
        yaxis_title="Surprisal",
    )

    return fig
