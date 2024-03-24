from __future__ import annotations

import plotly.graph_objects as go
from typing import List


def create_heatmap(x_ticks: List[str], y_ticks: List[str], values: List[float], title: str = '',
                   cell_annotations: List[str] = None, x_label: str = '',
                   y_label: str = '') -> go.Figure:
    """
    Create a heatmap with annotated cells. Set the x_ticks, y_ticks and title accordingly.

    Args:
        x_ticks (list): Tick labels for the x-axis. Should have shape (n)
        y_ticks (list): Labels for the y-axis. Should have shape (n)
        values (list): Cell values for the heatmap. Should have shape (nxn)
        title (str): Title for the plot, default = ''
        cell_annotations (list): Text printed inside the cells.
            If given, should have shape (nxn), default = None
        x_label (str): Label for the x-axis, default = ''
        y_label (str): Label for the yaxis, default = ''
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


def plot_surprisal(layers, values, std=None, title="Surprisal") -> go.Figure:
    """
    Create a line plot of the surprisal values.

    Args:
        surprisal_values (list): The surprisal values.
        layers (list): The layers.
        title (str, optional): The title of the plot. Defaults to "Surprisal".

    Returns:
        go.Figure: The bar plot figure.
    """
    if not isinstance(layers, list):
        layers = list(layers)

    # Create the scatter plot
    fig = go.Figure(
        data=go.Scatter(
            x=layers,
            y=values,
            mode="lines+markers",
        ),
    )

    # If there are standard deviation values, add them to the plot
    if std is not None:
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=values + std,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=values - std,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(width=0),
                showlegend=False,
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


