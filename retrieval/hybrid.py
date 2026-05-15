"""
Combining BM25 and dense retrieval results using Reciprocal Rank Fusion.
"""

from retrieval.bm25_index import search as bm25_search
from retrieval.vector_store import search as vector_search

RRF_K = 60  # standard RRF constant


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
) -> dict[str, float]:
    """Computing RRF scores by merging two ranked lists."""
    scores: dict[str, float] = {}

    for result in bm25_results:
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + result["rank"])

    for result in vector_results:
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + result["rank"])

    return scores


def search(query: str, top_k: int = 5) -> list[dict]:
    """Searching with hybrid BM25 + dense retrieval and returning RRF-ranked top_k chunks."""
    bm25_results   = bm25_search(query,   top_k=top_k * 2)
    vector_results = vector_search(query, top_k=top_k * 2)

    # Building chunk lookup from both result sets
    chunk_lookup: dict[str, dict] = {}
    for r in bm25_results + vector_results:
        cid = r["chunk_id"]
        if cid not in chunk_lookup:
            chunk_lookup[cid] = r

    # Computing RRF scores
    rrf_scores = reciprocal_rank_fusion(bm25_results, vector_results)

    # Sorting by RRF score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for rank, (cid, rrf_score) in enumerate(ranked):
        chunk = chunk_lookup[cid].copy()
        chunk["rrf_score"] = round(rrf_score, 6)
        chunk["rank"]      = rank + 1
        results.append(chunk)

    return results