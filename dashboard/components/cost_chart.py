import requests
from dash import dcc, html
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"


def build_cost_chart():
    """Building cost charts from API trace data."""
    try:
        traces = requests.get(f"{API_BASE}/traces?n=50", timeout=5).json()
    except Exception as e:
        print(f"Fetching cost data failed — {e}.")
        traces = []

    # Bar chart: cost per query
    bar_fig = go.Figure()
    if traces:
        queries = [t["query"][:30] + "..." for t in reversed(traces)]
        costs   = [t["confidence"] * 0.000057 for t in reversed(traces)]
        bar_fig.add_trace(go.Bar(
            x=list(range(len(queries))),
            y=costs,
            marker_color="#059669",
            hovertext=queries,
        ))
    bar_fig.update_layout(
        title="Cost per Query",
        xaxis_title="Query Index",
        yaxis_title="Cost (USD)",
        height=300,
        margin={"t": 40, "b": 40},
    )

    # Area chart: cumulative cost over time
    area_fig = go.Figure()
    if traces:
        sorted_traces = list(reversed(traces))
        cumulative    = []
        running       = 0.0
        dates         = []
        for t in sorted_traces:
            running += t["confidence"] * 0.000057
            cumulative.append(round(running, 6))
            dates.append(t["created_at"][:19])
        area_fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative,
            fill="tozeroy",
            mode="lines",
            line={"color": "#059669"},
            name="Cumulative Cost",
        ))
    area_fig.update_layout(
        title="Cumulative Cost Over Time",
        xaxis_title="Time",
        yaxis_title="Total Cost (USD)",
        height=300,
        margin={"t": 40, "b": 40},
    )

    return html.Div([
        html.H3("Cost Analysis", style={"marginBottom": "10px"}),
        dcc.Graph(figure=bar_fig),
        dcc.Graph(figure=area_fig),
    ])