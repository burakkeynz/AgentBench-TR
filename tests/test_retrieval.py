# tests/test_retrieval.py
"""
Testing BM25, vector, and hybrid retrieval functions.
"""
import pytest
from unittest.mock import patch, MagicMock
from retrieval.bm25_index import chunk_document, BM25Index, search as bm25_search
from retrieval.hybrid import reciprocal_rank_fusion


# ── Chunking ───────────────────────────────────────────────────────────────

def test_chunk_document_basic():
    """Testing basic document chunking into correct sizes."""
    text = " ".join([f"word{i}" for i in range(600)])
    chunks = chunk_document(text, source="test_doc", chunk_size=512, overlap=50)
    assert len(chunks) >= 1
    assert chunks[0]["source"] == "test_doc"
    assert chunks[0]["tokens"] <= 512


def test_chunk_document_overlap():
    """Testing that overlapping chunks share tokens."""
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_document(text, source="test", chunk_size=512, overlap=50)
    assert len(chunks) >= 2


def test_chunk_document_short_text():
    """Testing single chunk output for short documents."""
    text = "Bu kısa bir metin."
    chunks = chunk_document(text, source="short", chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


# ── BM25 ───────────────────────────────────────────────────────────────────

def test_bm25_index_search_returns_results():
    """Testing BM25 index search returns top_k results."""
    chunks = [
        {"chunk_id": f"doc__chunk_{i}", "source": "doc", "text": f"Bu bir test metnidir {i}", "tokens": 5}
        for i in range(10)
    ]
    index = BM25Index(chunks)
    results = index.search("test metni", top_k=3)
    assert len(results) == 3
    assert "score" in results[0]
    assert "rank" in results[0]


def test_bm25_search_rank_order():
    """Testing BM25 results are ordered by score descending."""
    chunks = [
        {"chunk_id": "c1", "source": "doc", "text": "BERTurk Türkçe dil modeli", "tokens": 4},
        {"chunk_id": "c2", "source": "doc", "text": "hava durumu bugün güzel", "tokens": 4},
        {"chunk_id": "c3", "source": "doc", "text": "BERTurk BERT modeli Türkçe NLP", "tokens": 5},
    ]
    index = BM25Index(chunks)
    results = index.search("BERTurk", top_k=3)
    assert results[0]["score"] >= results[1]["score"]


# ── RRF ────────────────────────────────────────────────────────────────────

def test_reciprocal_rank_fusion_combines_results():
    """Testing RRF correctly combines BM25 and vector results."""
    bm25 = [
        {"chunk_id": "c1", "rank": 1},
        {"chunk_id": "c2", "rank": 2},
    ]
    vector = [
        {"chunk_id": "c2", "rank": 1},
        {"chunk_id": "c3", "rank": 2},
    ]
    scores = reciprocal_rank_fusion(bm25, vector)
    assert "c1" in scores
    assert "c2" in scores
    assert "c3" in scores
    # c2 appears in both — should have highest score
    assert scores["c2"] > scores["c1"]
    assert scores["c2"] > scores["c3"]


def test_reciprocal_rank_fusion_empty():
    """Testing RRF with empty inputs returns empty dict."""
    scores = reciprocal_rank_fusion([], [])
    assert scores == {}