# tests/test_eval.py
"""
Testing consistency, hallucination, and cost evaluation functions.
"""
import pytest
from unittest.mock import patch, MagicMock
from eval.hallucination import calculate_hallucination_rate
from eval.cost_tracker import calculate_cost, log_token_usage, MODEL_PRICES
from eval.latency import calculate_latency_breakdown
from eval.failure_taxonomy import classify_failure, LOOP_DETECTED, LOW_CONFIDENCE, NO_SOURCE_FOUND
from graph.state import AgentState


# ── Hallucination ──────────────────────────────────────────────────────────

def test_hallucination_rate_calculation():
    """Testing hallucination rate calculates correctly from validator logs."""
    mock_logs = [
        {"agent_name": "validator", "output": "3 claims, 1 hallucination flags", "latency_ms": 100},
    ]
    with patch("eval.hallucination.get_agent_logs", return_value=mock_logs):
        result = calculate_hallucination_rate("trace-001")
    assert result["hallucination_rate"] == round(1/3, 4)
    assert result["total_claims"] == 3
    assert result["flagged_claims"] == 1


def test_hallucination_rate_no_logs():
    """Testing hallucination rate returns zero when no validator logs found."""
    with patch("eval.hallucination.get_agent_logs", return_value=[]):
        result = calculate_hallucination_rate("trace-000")
    assert result["hallucination_rate"] == 0.0


# ── Cost ───────────────────────────────────────────────────────────────────

def test_log_and_calculate_cost():
    """Testing token logging and cost calculation for a trace."""
    log_token_usage("trace-cost-01", "gpt-4o-mini", input_tokens=1000, output_tokens=500)
    result = calculate_cost("trace-cost-01", model="gpt-4o-mini")
    expected = (1000 / 1_000_000) * 0.150 + (500 / 1_000_000) * 0.600
    assert abs(result["cost_usd"] - round(expected, 6)) < 1e-6


def test_model_prices_exist():
    """Testing MODEL_PRICES contains required models."""
    assert "gpt-4o-mini" in MODEL_PRICES
    assert "input" in MODEL_PRICES["gpt-4o-mini"]
    assert "output" in MODEL_PRICES["gpt-4o-mini"]


# ── Latency ────────────────────────────────────────────────────────────────

def test_latency_breakdown():
    """Testing latency breakdown sums correctly and identifies bottleneck."""
    mock_logs = [
        {"agent_name": "planner",     "latency_ms": 1000},
        {"agent_name": "search",      "latency_ms": 3000},
        {"agent_name": "validator",   "latency_ms": 500},
        {"agent_name": "synthesizer", "latency_ms": 500},
    ]
    with patch("eval.latency.get_agent_logs", return_value=mock_logs):
        result = calculate_latency_breakdown("trace-lat-01")
    assert result["total_ms"] == 5000.0
    assert result["bottleneck_agent"] == "search"


# ── Failure Taxonomy ───────────────────────────────────────────────────────

def test_classify_loop_detected():
    """Testing LOOP_DETECTED classification when retry_count >= 3."""
    state = AgentState(query="test", retry_count=3)
    assert classify_failure(state) == LOOP_DETECTED


def test_classify_no_source_found():
    """Testing NO_SOURCE_FOUND classification when retrieved_docs is empty."""
    state = AgentState(query="test", retrieved_docs=[], retry_count=0)
    assert classify_failure(state) == NO_SOURCE_FOUND


def test_classify_low_confidence():
    """Testing LOW_CONFIDENCE classification when confidence_score < 0.3."""
    state = AgentState(
        query="test",
        retrieved_docs=[{"chunk_id": "c1"}],
        agent_logs=[{"agent": "search", "output": "1 unique chunks retrieved", "latency_ms": 100}],
        confidence_score=0.1,
        retry_count=0,
    )
    assert classify_failure(state) == LOW_CONFIDENCE


def test_classify_no_failure():
    """Testing None returned when no failure condition is met."""
    state = AgentState(
        query="test",
        retrieved_docs=[{"chunk_id": "c1"}],
        agent_logs=[{"agent": "search", "output": "1 unique chunks retrieved", "latency_ms": 100}],
        confidence_score=0.8,
        retry_count=0,
    )
    assert classify_failure(state) is None