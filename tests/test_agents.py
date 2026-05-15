# tests/test_agents.py
"""
Testing each agent with mock LLM responses.
"""
import pytest
from unittest.mock import patch, MagicMock
from graph.state import AgentState


# ── PlannerAgent ───────────────────────────────────────────────────────────

def test_planner_agent_populates_sub_tasks():
    """Testing PlannerAgent populates sub_tasks from LLM response."""
    mock_response = MagicMock()
    mock_response.content = "1. Alt görev bir\n2. Alt görev iki\n3. Alt görev üç"

    with patch("agents.planner._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        from agents.planner import planner_agent
        state = AgentState(query="BERTurk nedir?")
        result = planner_agent(state)

    assert len(result.sub_tasks) == 3
    assert len(result.agent_logs) == 1
    assert result.agent_logs[0]["agent"] == "planner"


# ── SearchAgent ────────────────────────────────────────────────────────────

def test_search_agent_populates_retrieved_docs():
    """Testing SearchAgent populates retrieved_docs via hybrid search."""
    mock_docs = [
        {"chunk_id": "c1", "source": "doc", "text": "BERTurk açıklaması", "tokens": 3, "score": 0.9, "rank": 1, "rrf_score": 0.03},
    ]

    with patch("agents.search.hybrid_search", return_value=mock_docs):
        from agents.search import search_agent
        state = AgentState(query="test", sub_tasks=["BERTurk nedir?"])
        result = search_agent(state)

    assert len(result.retrieved_docs) == 1
    assert result.agent_logs[0]["agent"] == "search"


def test_search_agent_deduplicates_docs():
    """Testing SearchAgent deduplicates identical chunks across sub-tasks."""
    mock_doc = {"chunk_id": "c1", "source": "doc", "text": "metin", "tokens": 1, "score": 0.9, "rank": 1, "rrf_score": 0.03}

    with patch("agents.search.hybrid_search", return_value=[mock_doc]):
        from agents.search import search_agent
        state = AgentState(query="test", sub_tasks=["görev 1", "görev 2"])
        result = search_agent(state)

    assert len(result.retrieved_docs) == 1


# ── ValidatorAgent ─────────────────────────────────────────────────────────

def test_validator_agent_populates_claims():
    """Testing ValidatorAgent populates validated_claims from LLM JSON response."""
    mock_response = MagicMock()
    mock_response.content = '{"claims": [{"claim": "BERTurk Türkçe modelidir", "source_index": 1, "validated": true, "hallucination": false}]}'

    with patch("agents.validator._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        from agents.validator import validator_agent
        state = AgentState(
            query="BERTurk nedir?",
            retrieved_docs=[{"source": "doc", "text": "BERTurk Türkçe modelidir"}]
        )
        result = validator_agent(state)

    assert len(result.validated_claims) == 1
    assert result.validated_claims[0]["validated"] is True


# ── SynthesizerAgent ───────────────────────────────────────────────────────

def test_synthesizer_agent_calculates_confidence():
    """Testing SynthesizerAgent calculates confidence from validated claims ratio."""
    mock_response = MagicMock()
    mock_response.content = "BERTurk, Türkçe için geliştirilmiş bir BERT modelidir."

    with patch("agents.synthesizer._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        from agents.synthesizer import synthesizer_agent
        state = AgentState(
            query="BERTurk nedir?",
            validated_claims=[
                {"claim": "c1", "validated": True},
                {"claim": "c2", "validated": True},
                {"claim": "c3", "validated": False},
            ]
        )
        result = synthesizer_agent(state)

    assert result.confidence_score == round(2/3, 2)
    assert result.final_answer != ""


def test_synthesizer_agent_no_validated_claims():
    """Testing SynthesizerAgent handles empty validated claims gracefully."""
    from agents.synthesizer import synthesizer_agent
    state = AgentState(query="test", validated_claims=[])
    result = synthesizer_agent(state)
    assert result.final_answer == "Yeterli doğrulanmış bilgi bulunamadı."
    assert result.confidence_score == 0.0