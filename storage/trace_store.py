"""
Providing CRUD operations for Trace, AgentLog, and EvalResult models.
"""

import uuid
from datetime import datetime
from typing import Optional

from storage.database import get_db
from storage.models import AgentLog, EvalResult, Trace


# ── Trace ──────────────────────────────────────────────────────────────────

def save_trace(
    query_text: str,
    final_answer: Optional[str] = None,
    confidence_score: Optional[float] = None,
    query_id: Optional[str] = None,
) -> str:
    """Saving a new trace record and returning its query_id."""
    qid = query_id or str(uuid.uuid4())
    with get_db() as db:
        trace = Trace(
            query_id=qid,
            query_text=query_text,
            final_answer=final_answer,
            confidence_score=confidence_score,
            created_at=datetime.utcnow(),
        )
        db.add(trace)
    print(f"Saving trace {qid[:8]}... — done.")
    return qid


def get_trace(query_id: str) -> Optional[dict]:
    """Retrieving a single trace by query_id."""
    with get_db() as db:
        trace = db.query(Trace).filter(Trace.query_id == query_id).first()
        if not trace:
            return None
        return {
            "query_id":        trace.query_id,
            "query_text":      trace.query_text,
            "final_answer":    trace.final_answer,
            "confidence_score": trace.confidence_score,
            "created_at":      trace.created_at.isoformat(),
        }


def get_all_traces() -> list[dict]:
    """Retrieving all traces ordered by creation time descending."""
    with get_db() as db:
        traces = db.query(Trace).order_by(Trace.created_at.desc()).all()
        return [
            {
                "query_id":        t.query_id,
                "query_text":      t.query_text,
                "final_answer":    t.final_answer,
                "confidence_score": t.confidence_score,
                "created_at":      t.created_at.isoformat(),
            }
            for t in traces
        ]


# ── AgentLog ───────────────────────────────────────────────────────────────

def save_agent_log(
    trace_id: str,
    agent_name: str,
    input: Optional[str] = None,
    output: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> int:
    """Saving an agent execution log and returning its id."""
    with get_db() as db:
        log = AgentLog(
            trace_id=trace_id,
            agent_name=agent_name,
            input=input,
            output=output,
            latency_ms=latency_ms,
            created_at=datetime.utcnow(),
        )
        db.add(log)
        db.flush()
        log_id = log.id
    print(f"Saving agent log [{agent_name}] for trace {trace_id[:8]}... — done.")
    return log_id


def get_agent_logs(trace_id: str) -> list[dict]:
    """Retrieving all agent logs for a given trace."""
    with get_db() as db:
        logs = db.query(AgentLog).filter(AgentLog.trace_id == trace_id).all()
        return [
            {
                "id":         l.id,
                "trace_id":   l.trace_id,
                "agent_name": l.agent_name,
                "input":      l.input,
                "output":     l.output,
                "latency_ms": l.latency_ms,
                "created_at": l.created_at.isoformat(),
            }
            for l in logs
        ]


# ── EvalResult ─────────────────────────────────────────────────────────────

def save_eval_result(
    trace_id: str,
    consistency_score: Optional[float] = None,
    hallucination_rate: Optional[float] = None,
    cost_usd: Optional[float] = None,
) -> int:
    """Saving evaluation results for a trace and returning its id."""
    with get_db() as db:
        result = EvalResult(
            trace_id=trace_id,
            consistency_score=consistency_score,
            hallucination_rate=hallucination_rate,
            cost_usd=cost_usd,
            created_at=datetime.utcnow(),
        )
        db.add(result)
        db.flush()
        result_id = result.id
    print(f"Saving eval result for trace {trace_id[:8]}... — done.")
    return result_id


def get_eval_result(trace_id: str) -> Optional[dict]:
    """Retrieving the latest eval result for a given trace."""
    with get_db() as db:
        result = (
            db.query(EvalResult)
            .filter(EvalResult.trace_id == trace_id)
            .order_by(EvalResult.created_at.desc())
            .first()
        )
        if not result:
            return None
        return {
            "id":                 result.id,
            "trace_id":           result.trace_id,
            "consistency_score":  result.consistency_score,
            "hallucination_rate": result.hallucination_rate,
            "cost_usd":           result.cost_usd,
            "created_at":         result.created_at.isoformat(),
        }