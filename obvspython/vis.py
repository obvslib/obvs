from __future__ import annotations
from typing import List

import plotly.graph_objects as go


def create_heatmap(model_1_layers, model_2_layers, values) -> go.Figure:
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


def create_annotated_heatmap(values: List[float], cell_annotations: List[str], x_ticks: List[str],
                             y_ticks: List[str], x_label: str = '', y_label: str = '',
                             title: str = '') -> go.Figure:
    """
    Create a heatmap with annotated cells. Set the x_ticks, y_ticks and title accordingly.

    Args:
        values (list): Cell values for the heatmap. Should have shape (nxn)
        cell_annotations (list): Text printed inside the cells. Should have shape (nxn)
        x_ticks (list): Tick labels for the x-axis. Should have shape (n)
        y_ticks (list): Labels for the y-axis. Should have shape (n)
        x_label (str): Label for the x-axis, default = ''
        y_label (str): Label for the yaxis, default = ''
        title (str): Title for the plot, default = ''
    Returns:
        go.Figure: The heatmap figure.
    """

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=x_ticks,
        y=y_ticks,
        hoverongaps=False,
        text=cell_annotations,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='Viridis'))

    fig.update_layout(
        title=title,
        xaxis=dict(title=x_label, tickfont=dict(size=16), titlefont=dict(size=18), tickangle=-45),
        yaxis=dict(title=y_label, tickfont=dict(size=16), titlefont=dict(size=18)),
        titlefont=dict(size=20)
    )

    return fig