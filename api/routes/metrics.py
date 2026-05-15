from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter, HTTPException
from storage.database import get_db
from storage.models import Trace, EvalResult
from storage.trace_store import get_agent_logs
from api.schemas import MetricsResponse, Tracesummary

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Calculating and returning aggregated metrics across all traces."""
    with get_db() as db:
        eval_results = db.query(EvalResult).all()

        if not eval_results:
            return MetricsResponse(
                consistency=0.0,
                hallucination_rate=0.0,
                avg_cost=0.0,
                avg_latency=0.0,
            )

        consistency   = sum(r.consistency_score or 0 for r in eval_results) / len(eval_results)
        hallucination = sum(r.hallucination_rate or 0 for r in eval_results) / len(eval_results)
        avg_cost      = sum(r.cost_usd           or 0 for r in eval_results) / len(eval_results)
        trace_ids     = [r.trace_id for r in eval_results]

    total_latencies = []
    for tid in trace_ids:
        logs  = get_agent_logs(tid)
        total = sum(l.get("latency_ms") or 0 for l in logs)
        if total > 0:
            total_latencies.append(total)

    avg_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0

    print(f"Calculating metrics — {len(eval_results)} eval results, avg_cost=${avg_cost:.6f}.")

    return MetricsResponse(
        consistency=round(consistency, 4),
        hallucination_rate=round(hallucination, 4),
        avg_cost=round(avg_cost, 6),
        avg_latency=round(avg_latency, 2),
    )


@router.get("/traces", response_model=list[Tracesummary])
def get_traces(n: int = 20) -> list[Tracesummary]:
    """Listing last N traces ordered by creation time."""
    with get_db() as db:
        traces = (
            db.query(Trace)
            .order_by(Trace.created_at.desc())
            .limit(n)
            .all()
        )

        print(f"Listing traces — returning {len(traces)} records.")

        return [
            Tracesummary(
                trace_id=t.query_id,
                query=t.query_text,
                answer=t.final_answer or "",
                confidence=t.confidence_score or 0.0,
                created_at=t.created_at.isoformat(),
            )
            for t in traces
        ]


@router.get("/traces/{trace_id}", response_model=Tracesummary)
def get_trace_detail(trace_id: str) -> Tracesummary:
    """Retrieving single trace detail by trace_id."""
    with get_db() as db:
        trace = db.query(Trace).filter(Trace.query_id == trace_id).first()

        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found.")

        print(f"Retrieving trace detail — trace_id={trace_id[:8]}.")

        return Tracesummary(
            trace_id=trace.query_id,
            query=trace.query_text,
            answer=trace.final_answer or "",
            confidence=trace.confidence_score or 0.0,
            created_at=trace.created_at.isoformat(),
        )