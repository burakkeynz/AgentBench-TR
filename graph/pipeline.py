from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from graph.state import AgentState
from agents.planner import planner_agent
from agents.search import search_agent
from agents.validator import validator_agent
from agents.synthesizer import synthesizer_agent


def _increment_retry(state: AgentState) -> AgentState:
    """Incrementing retry counter before re-entering search."""
    state.retry_count += 1
    print(f"Incrementing retry count — now at {state.retry_count}.")
    return state


def _route_validator(state: AgentState) -> str:
    """Routing after validator — retrying if low confidence, else synthesizing."""
    validated = [c for c in state.validated_claims if c.get("validated")]
    total = len(state.validated_claims)
    score = round(len(validated) / total, 2) if total > 0 else 0.0

    if score < 0.5 and state.retry_count < 3:
        print(f"Routing to retry — score={score}, retry={state.retry_count}.")
        return "retry"
    print(f"Routing to synthesizer — score={score}, retry={state.retry_count}.")
    return "synthesizer"


def build_graph() -> StateGraph:
    """Building and compiling the AgentBench-TR LangGraph pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("planner",     planner_agent)
    graph.add_node("search",      search_agent)
    graph.add_node("validator",   validator_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.add_node("retry",       _increment_retry)

    graph.set_entry_point("planner")
    graph.add_edge("planner",  "search")
    graph.add_edge("search",   "validator")
    graph.add_conditional_edges(
        "validator",
        _route_validator,
        {"retry": "retry", "synthesizer": "synthesizer"},
    )
    graph.add_edge("retry", "search")
    graph.add_edge("synthesizer", END)

    return graph.compile()


pipeline = build_graph()
