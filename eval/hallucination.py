from dotenv import load_dotenv
load_dotenv()

from storage.trace_store import get_agent_logs


def calculate_hallucination_rate(trace_id: str) -> dict:
    """Calculating hallucination rate from validator agent logs for a given trace."""
    logs = get_agent_logs(trace_id)

    validator_logs = [l for l in logs if l["agent_name"] == "validator"]

    if not validator_logs:
        print(f"Calculating hallucination rate — no validator logs found for trace {trace_id[:8]}.")
        return {
            "trace_id":          trace_id,
            "total_claims":      0,
            "flagged_claims":    0,
            "hallucination_rate": 0.0,
        }

    total_claims   = 0
    flagged_claims = 0

    for log in validator_logs:
        output = log.get("output", "")
        # Parsing "N claims, M hallucination flags" format
        try:
            parts         = output.split(",")
            total_claims  += int(parts[0].strip().split()[0])
            flagged_claims += int(parts[1].strip().split()[0])
        except (IndexError, ValueError):
            continue

    rate = round(flagged_claims / total_claims, 4) if total_claims > 0 else 0.0

    print(f"Calculating hallucination rate — trace {trace_id[:8]}: {flagged_claims}/{total_claims} flagged, rate={rate}.")

    return {
        "trace_id":           trace_id,
        "total_claims":       total_claims,
        "flagged_claims":     flagged_claims,
        "hallucination_rate": rate,
    }