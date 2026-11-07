from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Categorical colours in a fixed, colour-vision-deficiency-safe order. Assign
# series to slots in this order (never by rank) so an entity keeps its colour
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
]

# Status colours are reserved for good / bad states and never used for series
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# Colours of the two W-state constructions, shared by every report
METHOD_COLORS = {"linear": SERIES[0], "tree": SERIES[1]}

# Sequential blue ramp (light = low, dark = high) for heatmaps
SEQUENTIAL = [
    [0.0, "#f4f8fe"],
    [0.15, "#cde2fb"],
    [0.35, "#86b6ef"],
    [0.55, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]

INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", "Hiragino Sans", sans-serif'


def style(
    fig: go.Figure, *, height: int = 380, x_title: str | None = None, y_title: str | None = None,
    hovermode: str = "closest", legend: bool | None = None,
) -> go.Figure:
    """
    Function that applies the shared look to a Plotly figure in place

    Parameters
    ----------
    fig: go.Figure
        Figure to style
    height: int = 380
        Figure height in pixels
    x_title: str | None = None
        Title of the x axes (unchanged when None)
    y_title: str | None = None
        Title of the y axes (unchanged when None)
    hovermode: str = "closest"
        Plotly hover mode, e.g. "x unified" for line charts
    legend: bool | None = None
        Show or hide the legend (Plotly decides when None)

    Returns
    -------
    fig: go.Figure
        The same figure, styled
    """
    # Transparent background, system font and a horizontal legend above the plot
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=24, t=36, b=52),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK),
        hovermode=hovermode,
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        barcornerradius=4,
        bargap=0.35,
    )
    if legend is not None:
        fig.update_layout(showlegend=legend)

    # Recessive hairline axes; horizontal gridlines only
    axis = dict(
        gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS,
        tickfont=dict(color=MUTED), title_font=dict(color=INK), automargin=True,
    )
    fig.update_xaxes(**axis, showgrid=False)
    fig.update_yaxes(**axis, showgrid=True)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)

    # 2px lines and 8px markers with a ring in the surface colour
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=2))
    fig.update_traces(selector=dict(type="scatter"), marker=dict(size=8, line=dict(width=2, color="#fcfcfb")))

    return fig


def lines(
    x: Sequence, series: dict[str, Sequence[float]], *, colors: dict[str, str] | None = None,
    x_title: str = "", y_title: str = "", hover_format: str = ".3g", height: int = 380,
    log_y: bool = False, dash: dict[str, str] | None = None,
) -> go.Figure:
    """
    Function that draws a line chart with one trace per series

    Parameters
    ----------
    x: Sequence
        Values on the x axis
    series: dict[str, Sequence[float]]
        Series name -> y values, drawn in insertion order
    colors: dict[str, str] | None = None
        Series name -> colour (default: SERIES in order)
    x_title: str = ""
        Title of the x axis
    y_title: str = ""
        Title of the y axis
    hover_format: str = ".3g"
        d3 format of hovered values
    height: int = 380
        Figure height in pixels
    log_y: bool = False
        Use a logarithmic y axis
    dash: dict[str, str] | None = None
        Series name -> Plotly dash style ("solid", "dash", "dot", ...)

    Returns
    -------
    fig: go.Figure
        Styled line chart
    """
    # Add one line per series, keeping its colour and dash style
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        color = (colors or {}).get(name, SERIES[i % len(SERIES)])
        fig.add_trace(
            go.Scatter(
                x=list(x), y=list(ys), name=name, mode="lines+markers",
                line=dict(color=color, dash=(dash or {}).get(name, "solid")), marker=dict(color=color),
                hovertemplate=f"%{{y:{hover_format}}}<extra>{name}</extra>",
            )
        )

    # Apply the shared look; a single series needs no legend
    style(fig, height=height, x_title=x_title, y_title=y_title, hovermode="x unified", legend=len(series) > 1)
    if log_y:
        fig.update_yaxes(type="log")

    return fig


def small_multiples(
    x: Sequence, panels: dict[str, dict[str, Sequence[float]]], *, colors: dict[str, str] | None = None,
    x_title: str = "", y_title: str = "", hover_format: str = ".3g", height: int = 360,
) -> go.Figure:
    """
    Function that draws one line chart per panel, side by side, on a shared y axis

    Parameters
    ----------
    x: Sequence
        Values on the x axis, shared by every panel
    panels: dict[str, dict[str, Sequence[float]]]
        Panel title -> {series name -> y values}; every panel has the same series
    colors: dict[str, str] | None = None
        Series name -> colour; series keep their colour in every panel
    x_title: str = ""
        Title of every x axis
    y_title: str = ""
        Title of the leftmost y axis
    hover_format: str = ".3g"
        d3 format of hovered values
    height: int = 360
        Figure height in pixels

    Returns
    -------
    fig: go.Figure
        Styled figure with one legend entry per series
    """
    # Lay the panels out in one row and share the y axis so heights compare directly
    fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True,
                        subplot_titles=list(panels), horizontal_spacing=0.04)

    # Add the same series to every panel, listing each series once in the legend
    for col, (_, series) in enumerate(panels.items(), start=1):
        for i, (name, ys) in enumerate(series.items()):
            color = (colors or {}).get(name, SERIES[i % len(SERIES)])
            fig.add_trace(
                go.Scatter(
                    x=list(x), y=list(ys), name=name, legendgroup=name, showlegend=col == 1,
                    mode="lines+markers", line=dict(color=color), marker=dict(color=color),
                    hovertemplate=f"%{{y:{hover_format}}}<extra>{name}</extra>",
                ),
                row=1, col=col,
            )

    # Apply the shared look, leaving room for the legend above the panel titles
    style(fig, height=height, hovermode="x unified", legend=True)
    fig.update_layout(margin=dict(t=72), legend=dict(y=1.14))
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_annotations(font=dict(size=13, color=INK))

    return fig


def bars(
    categories: Sequence, series: dict[str, Sequence[float]], *, colors: dict[str, str] | None = None,
    x_title: str = "", y_title: str = "", hover_format: str = ".3g", height: int = 360,
    reference: float | None = None, reference_label: str = "",
) -> go.Figure:
    """
    Function that draws a grouped column chart, optionally with a reference line

    Parameters
    ----------
    categories: Sequence
        Category on each column group
    series: dict[str, Sequence[float]]
        Series name -> one value per category
    colors: dict[str, str] | None = None
        Series name -> colour (default: SERIES in order)
    x_title: str = ""
        Title of the x axis
    y_title: str = ""
        Title of the y axis
    hover_format: str = ".3g"
        d3 format of hovered values
    height: int = 360
        Figure height in pixels
    reference: float | None = None
        y value of a dotted horizontal reference line (none when None)
    reference_label: str = ""
        Text next to the reference line

    Returns
    -------
    fig: go.Figure
        Styled column chart
    """
    # Add one group of columns per series
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        color = (colors or {}).get(name, SERIES[i % len(SERIES)])
        fig.add_trace(
            go.Bar(
                x=[str(c) for c in categories], y=list(ys), name=name,
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate=f"%{{x}}: %{{y:{hover_format}}}<extra>{name}</extra>",
            )
        )

    # Apply the shared look and keep categories in the given order
    style(fig, height=height, x_title=x_title, y_title=y_title, legend=len(series) > 1)
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_xaxes(type="category")

    # Draw the reference line as a dotted trace named in the legend, so its label
    # can never collide with the bars
    if reference is not None:
        labels = [str(c) for c in categories]
        fig.add_trace(
            go.Scatter(
                x=labels, y=[reference] * len(labels), name=reference_label or "reference", mode="lines",
                line=dict(color=MUTED, width=1.5, dash="dot", shape="hvh"), hoverinfo="skip",
            )
        )
        fig.update_layout(showlegend=True)

    return fig


def heatmap(
    z, *, x: Sequence, y: Sequence, x_title: str = "", y_title: str = "", z_title: str = "",
    zmin: float | None = 0.0, zmax: float | None = None, hover_format: str = ".3f", height: int = 380,
    colorscale=None,
) -> go.Figure:
    """
    Function that draws a heatmap on the sequential blue ramp

    Parameters
    ----------
    z: array-like
        2-D values, one row per y value
    x: Sequence
        Labels of the columns
    y: Sequence
        Labels of the rows
    x_title: str = ""
        Title of the x axis
    y_title: str = ""
        Title of the y axis
    z_title: str = ""
        Title of the colour bar
    zmin: float | None = 0.0
        Value mapped to the lightest colour
    zmax: float | None = None
        Value mapped to the darkest colour (data maximum when None)
    hover_format: str = ".3f"
        d3 format of hovered values
    height: int = 380
        Figure height in pixels
    colorscale: list | None = None
        Plotly colour scale (default: SEQUENTIAL)

    Returns
    -------
    fig: go.Figure
        Styled heatmap
    """
    # Draw the cells with a 1px gap so neighbours stay distinct
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(x), y=list(y), zmin=zmin, zmax=zmax, colorscale=colorscale or SEQUENTIAL,
            colorbar=dict(title=dict(text=z_title, side="right"), thickness=12, outlinewidth=0),
            xgap=1, ygap=1,
            hovertemplate=f"{x_title} %{{x}}<br>{y_title} %{{y}}<br>{z_title} %{{z:{hover_format}}}<extra></extra>",
        )
    )

    # Apply the shared look without gridlines over the cells
    style(fig, height=height, x_title=x_title, y_title=y_title)
    fig.update_yaxes(showgrid=False)

    return fig
