"""
Combining all dashboard components into a unified Plotly Dash application.
"""
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

from dashboard.components.trace_table import build_trace_table
from dashboard.components.consistency_chart import build_consistency_chart
from dashboard.components.cost_chart import build_cost_chart
from dashboard.components.failure_heatmap import build_failure_heatmap

app = dash.Dash(
    __name__,
    title="AgentBench-TR Dashboard",
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    style={"fontFamily": "Inter, sans-serif", "padding": "20px", "backgroundColor": "#f4f4f8"},
    children=[
        html.H1("AgentBench-TR", style={"color": "#1e1e2e", "marginBottom": "4px"}),
        html.P("Multi-agent QA evaluation dashboard", style={"color": "#6b7280", "marginBottom": "24px"}),

        # Auto-refresh interval — refreshing every 30 seconds
        dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),

        # Trace table
        html.Div(id="trace-table-container", style={"marginBottom": "32px"}),

        # Charts row
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px", "marginBottom": "32px"},
            children=[
                html.Div(id="consistency-chart-container"),
                html.Div(id="cost-chart-container"),
            ],
        ),

        # Failure heatmap
        html.Div(id="failure-heatmap-container"),
    ],
)


@app.callback(
    Output("trace-table-container",    "children"),
    Output("consistency-chart-container", "children"),
    Output("cost-chart-container",     "children"),
    Output("failure-heatmap-container","children"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_dashboard(n_intervals: int):
    """Refreshing all dashboard components on interval tick."""
    print(f"Refreshing dashboard — tick {n_intervals}.")
    return (
        build_trace_table(),
        build_consistency_chart(),
        build_cost_chart(),
        build_failure_heatmap(),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)