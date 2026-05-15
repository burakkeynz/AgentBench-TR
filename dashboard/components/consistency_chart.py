import requests
from dash import dcc, html
import plotly.graph_objects as go

API_BASE = "http://localhost:8000"


def build_consistency_chart():
    """Building consistency score charts from API metrics and trace data."""
    try:
        metrics = requests.get(f"{API_BASE}/metrics", timeout=5).json()
        traces  = requests.get(f"{API_BASE}/traces?n=50", timeout=5).json()
    except Exception as e:
        print(f"Fetching consistency data failed — {e}.")
        metrics = {}
        traces  = []

    # Line chart — confidence over time
    line_fig = go.Figure()
    if traces:
        dates       = [t["created_at"][:19] for t in reversed(traces)]
        confidences = [t["confidence"] for t in reversed(traces)]
        line_fig.add_trace(go.Scatter(
            x=dates, y=confidences,
            mode="lines+markers",
            name="Confidence Score",
            line={"color": "#7c3aed"},
        ))
    line_fig.update_layout(
        title="Confidence Score Over Time",
        xaxis_title="Time",
        yaxis_title="Score",
        yaxis={"range": [0, 1.1]},
        height=300,
        margin={"t": 40, "b": 40},
    )

    # Bar chart — avg confidence by category (from traces)
    bar_fig = go.Figure()
    if traces:
        from collections import defaultdict
        cat_scores: dict = defaultdict(list)
        for t in traces:
            cat_scores["all"].append(t["confidence"])
        bar_fig.add_trace(go.Bar(
            x=list(cat_scores.keys()),
            y=[sum(v)/len(v) for v in cat_scores.values()],
            marker_color="#7c3aed",
        ))
    bar_fig.update_layout(
        title="Avg Confidence Score",
        yaxis={"range": [0, 1.1]},
        height=300,
        margin={"t": 40, "b": 40},
    )

    return html.Div([
        html.H3("Consistency & Confidence", style={"marginBottom": "10px"}),
        dcc.Graph(figure=line_fig),
        dcc.Graph(figure=bar_fig),
    ])