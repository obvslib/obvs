from __future__ import annotations

import plotly.graph_objects as go


def create_heatmap(
    x_data: list[str | int | float],
    y_data: list[str | int | float],
    values: list[float],
    title: str = "",
    cell_annotations: list[str] = None,
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """
    Create a heatmap with annotated cells. Set the x_ticks, y_ticks and title accordingly.

    Args:
        x_data (list): Data on the x-axis of the heatmap.
        y_data (list): Data on the y-axis of the heatmap.
        values (list): Cell values for the heatmap. Should have shape (nxn)
        title (str): Title for the plot, default = ''
        cell_annotations (list): Text printed inside the cells.
            If given, should have shape (nxn), default = None
        x_label (str): Label for the x-axis, default = ''
        y_label (str): Label for the yaxis, default = ''
    Returns:
        go.Figure: The heatmap figure.
    """

    # Ensure the outer list of values matches the length of y_data
    assert len(values) == len(y_data), "Length of values must match length of y_data"
    for row in values:
        assert len(row) == len(x_data), "Each row in values must match the length of x_data"

    # Use ordered indexing to accommodate non-unique labels
    x_ticks = list(range(len(x_data)))
    y_ticks = list(range(len(y_data)))

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=x_ticks,
            y=y_ticks,
            hoverongaps=False,
            text=cell_annotations,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale="Viridis",
        ),
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_label,
            tickfont=dict(size=16),
            titlefont=dict(size=18),
            tickangle=-45,
            tickvals=x_ticks,
            ticktext=x_data,
        ),
        yaxis=dict(
            title=y_label,
            tickfont=dict(size=16),
            titlefont=dict(size=18),
            tickvals=y_ticks,
            ticktext=y_data,
        ),
        titlefont=dict(size=20),
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
