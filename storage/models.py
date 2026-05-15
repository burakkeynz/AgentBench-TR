"""
Defining SQLAlchemy ORM models for traces, agent logs, and eval results.
"""

from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from storage.base import Base


class Trace(Base):
    """Storing top-level query traces."""
    __tablename__ = "traces"

    query_id         = Column(String(64), primary_key=True)
    query_text       = Column(Text, nullable=False)
    final_answer     = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    agent_logs   = relationship("AgentLog",   back_populates="trace", cascade="all, delete-orphan")
    eval_results = relationship("EvalResult", back_populates="trace", cascade="all, delete-orphan")


class AgentLog(Base):
    """Storing per-agent execution logs."""
    __tablename__ = "agent_logs"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    trace_id   = Column(String(64), ForeignKey("traces.query_id"), nullable=False)
    agent_name = Column(String(64), nullable=False)
    input      = Column(Text, nullable=True)
    output     = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    trace = relationship("Trace", back_populates="agent_logs")


class EvalResult(Base):
    """Storing evaluation metrics per trace."""
    __tablename__ = "eval_results"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    trace_id           = Column(String(64), ForeignKey("traces.query_id"), nullable=False)
    consistency_score  = Column(Float, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    cost_usd           = Column(Float, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow)

    trace = relationship("Trace", back_populates="eval_results")