from dotenv import load_dotenv
load_dotenv()

from storage.trace_store import get_agent_logs


def calculate_latency_breakdown(trace_id: str) -> dict:
    """Calculating latency breakdown per agent for a given trace."""
    logs = get_agent_logs(trace_id)

    if not logs:
        print(f"Calculating latency breakdown — no logs found for trace {trace_id[:8]}.")
        return {
            "trace_id":        trace_id,
            "total_ms":        0.0,
            "breakdown":       [],
            "bottleneck_agent": None,
        }

    breakdown = []
    for log in logs:
        breakdown.append({
            "agent":      log["agent_name"],
            "latency_ms": log["latency_ms"] or 0.0,
        })

    total_ms = sum(b["latency_ms"] for b in breakdown)

    # Calculating each agent's share
    for b in breakdown:
        b["share_pct"] = round((b["latency_ms"] / total_ms) * 100, 2) if total_ms > 0 else 0.0

    bottleneck = max(breakdown, key=lambda x: x["latency_ms"])

    print(f"Calculating latency breakdown — trace {trace_id[:8]}: total={total_ms:.1f}ms, bottleneck={bottleneck['agent']} ({bottleneck['share_pct']}%).")

    return {
        "trace_id":         trace_id,
        "total_ms":         round(total_ms, 2),
        "breakdown":        breakdown,
        "bottleneck_agent": bottleneck["agent"],
    }