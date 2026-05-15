import requests
from dash import dash_table, html

API_BASE = "http://localhost:8000"


def build_trace_table():
    """Building trace table component from API data."""
    try:
        response = requests.get(f"{API_BASE}/traces?n=50", timeout=5)
        traces = response.json()
    except Exception as e:
        print(f"Fetching traces failed — {e}.")
        traces = []

    rows = [
        {
            "trace_id":   t["trace_id"][:8],
            "query":      t["query"][:60],
            "answer":     t["answer"][:80],
            "confidence": t["confidence"],
            "created_at": t["created_at"],
        }
        for t in traces
    ]

    table = dash_table.DataTable(
        id="trace-table",
        columns=[
            {"name": "Trace ID",    "id": "trace_id"},
            {"name": "Query",       "id": "query"},
            {"name": "Answer",      "id": "answer"},
            {"name": "Confidence",  "id": "confidence"},
            {"name": "Created At",  "id": "created_at"},
        ],
        data=rows,
        row_selectable="single",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={"backgroundColor": "#1e1e2e", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"}
        ],
        page_size=20,
    )

    return html.Div([
        html.H3("Trace History", style={"marginBottom": "10px"}),
        table,
    ])