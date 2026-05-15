"""
Building failure type heatmap from trace data.
"""
import requests
from dash import dcc, html
import plotly.graph_objects as go
from collections import defaultdict

from eval.failure_taxonomy import (
    LOOP_DETECTED, TOOL_CALL_FAILED, TIMEOUT, NO_SOURCE_FOUND, LOW_CONFIDENCE
)

API_BASE = "http://localhost:8000"

FAILURE_TYPES = [LOOP_DETECTED, TOOL_CALL_FAILED, TIMEOUT, NO_SOURCE_FOUND, LOW_CONFIDENCE]
CATEGORIES    = ["factual", "conceptual", "multi-hop", "unknown"]


def _infer_category(query: str) -> str:
    """Inferring question category from query text heuristically."""
    q = query.lower()
    if any(w in q for w in ["ne zaman", "kaç", "hangi yıl", "kim"]):
        return "factual"
    if any(w in q for w in ["nasıl", "neden", "nedir", "ne işe"]):
        return "conceptual"
    if any(w in q for w in ["arasındaki", "ile", "fark", "ilişki"]):
        return "multi-hop"
    return "unknown"


def _infer_failure(trace: dict) -> str:
    """Inferring failure type from trace data."""
    # Checking low confidence
    if trace["confidence"] == 0.0:
        return LOW_CONFIDENCE
    if trace["confidence"] < 0.3:
        return LOW_CONFIDENCE
    # Checking no source
    if "bulunamadı" in trace["answer"].lower():
        return NO_SOURCE_FOUND
    return "none"


def build_failure_heatmap():
    """Building category × failure_type heatmap from trace data."""
    try:
        traces = requests.get(f"{API_BASE}/traces?n=100", timeout=5).json()
    except Exception as e:
        print(f"Fetching traces for heatmap failed — {e}.")
        traces = []

    # Counting category × failure_type
    counts: dict = defaultdict(lambda: defaultdict(int))
    for t in traces:
        cat     = _infer_category(t["query"])
        failure = _infer_failure(t)
        if failure != "none":
            counts[cat][failure] += 1

    z      = []
    labels = []
    for cat in CATEGORIES:
        row = [counts[cat][ft] for ft in FAILURE_TYPES]
        z.append(row)
        labels.append(cat)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=FAILURE_TYPES,
        y=labels,
        colorscale="Reds",
        text=z,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Failure Type by Question Category",
        xaxis_title="Failure Type",
        yaxis_title="Category",
        height=350,
        margin={"t": 50, "b": 60},
    )

    # Finding most frequent failure
    all_failures = [_infer_failure(t) for t in traces if _infer_failure(t) != "none"]
    most_common  = max(set(all_failures), key=all_failures.count) if all_failures else "none"
    print(f"Building failure heatmap — {len(traces)} traces, most common failure: {most_common}.")

    return html.Div([
        html.H3("Failure Analysis", style={"marginBottom": "10px"}),
        html.P(f"Most frequent failure: {most_common}", style={"color": "#dc2626", "fontWeight": "bold"}),
        dcc.Graph(figure=fig),
    ])