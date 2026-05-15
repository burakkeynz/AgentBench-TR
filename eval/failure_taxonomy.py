from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

from graph.state import AgentState

# Failure type constants
LOOP_DETECTED    = "LOOP_DETECTED"
TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
TIMEOUT          = "TIMEOUT"
NO_SOURCE_FOUND  = "NO_SOURCE_FOUND"
LOW_CONFIDENCE   = "LOW_CONFIDENCE"

LATENCY_THRESHOLD_MS = 60_000  # 60 seconds


def classify_failure(state: AgentState) -> str | None:
    """Classifying failure type from agent state — returning None if no failure detected."""

    # Checking loop
    if state.retry_count >= 3:
        print(f"Classifying failure — {LOOP_DETECTED}: retry_count={state.retry_count}.")
        return LOOP_DETECTED

    # Checking no source found
    if not state.retrieved_docs:
        print(f"Classifying failure — {NO_SOURCE_FOUND}: retrieved_docs is empty.")
        return NO_SOURCE_FOUND

    # Checking tool call failed
    retrieval_logs = [l for l in state.agent_logs if l.get("agent") == "search"]
    if retrieval_logs:
        last_search = retrieval_logs[-1]
        output = str(last_search.get("output", ""))
        if "0 unique chunks" in output:
            print(f"Classifying failure — {TOOL_CALL_FAILED}: retrieval returned 0 chunks.")
            return TOOL_CALL_FAILED

    # Checking timeout
    total_latency = sum(l.get("latency_ms", 0) for l in state.agent_logs)
    if total_latency > LATENCY_THRESHOLD_MS:
        print(f"Classifying failure — {TIMEOUT}: total_latency={total_latency:.0f}ms.")
        return TIMEOUT

    # Checking low confidence
    if state.confidence_score < 0.3:
        print(f"Classifying failure — {LOW_CONFIDENCE}: confidence_score={state.confidence_score}.")
        return LOW_CONFIDENCE

    return None