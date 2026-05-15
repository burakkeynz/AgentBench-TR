import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter
from graph.pipeline import pipeline
from graph.state import AgentState
from storage.database import init_db
from storage.trace_store import save_trace, save_agent_log
from api.schemas import QueryRequest, QueryResponse

router = APIRouter()
init_db()


@router.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest) -> QueryResponse:
    """Receiving query request, running pipeline, saving trace, returning response."""
    state = AgentState(query=request.query)
    result = pipeline.invoke(state)

    trace_id = save_trace(
        query_text=request.query,
        final_answer=result["final_answer"],
        confidence_score=result["confidence_score"],
    )

    for log in result["agent_logs"]:
        save_agent_log(
            trace_id=trace_id,
            agent_name=log["agent"],
            input=str(log["input"]),
            output=str(log["output"]),
            latency_ms=log["latency_ms"],
        )

    print(f"Processing query — trace_id={trace_id[:8]}, confidence={result['confidence_score']}.")

    return QueryResponse(
        answer=result["final_answer"],
        confidence=result["confidence_score"],
        trace_id=trace_id,
    )