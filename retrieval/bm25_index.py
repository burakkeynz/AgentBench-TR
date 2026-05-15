"""
Building BM25 index from processed documents and providing keyword search.
"""

import json
import re
import os
from pathlib import Path
from typing import Optional


from rank_bm25 import BM25Okapi

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHUNK_SIZE    = 512   # tokens
CHUNK_OVERLAP = 50    # tokens




# ── Chunking ───────────────────────────────────────────────────────────────

def chunk_document(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Splitting a document into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end        = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "chunk_id": f"{source}__chunk_{idx}",
            "source":   source,
            "text":     chunk_text,
            "tokens":   end - start,
        })

        idx   += 1
        start += chunk_size - overlap

    return chunks


def load_all_chunks(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Loading all processed documents and returning chunked list."""
    all_chunks = []
    for filepath in sorted(PROCESSED_DIR.glob("*.txt")):
        print(f"Loading chunks from {filepath.name}...")
        text   = filepath.read_text(encoding="utf-8")
        chunks = chunk_document(text, source=filepath.stem, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        print(f"  Generated {len(chunks)} chunks.")
    print(f"Loading complete — {len(all_chunks)} total chunks.")
    return all_chunks


# ── BM25 Index ─────────────────────────────────────────────────────────────

def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizing text into lowercase words for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Wrapping BM25Okapi with chunk metadata for search."""

    def __init__(self, chunks: list[dict]):
        print("Building BM25 index...")
        self.chunks    = chunks
        tokenized      = [_tokenize_for_bm25(c["text"]) for c in chunks]
        self.bm25      = BM25Okapi(tokenized)
        print(f"Building complete — {len(chunks)} chunks indexed.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Searching index and returning top_k chunks with scores."""
        query_tokens = _tokenize_for_bm25(query)
        scores       = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for rank, (idx, score) in enumerate(ranked):
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["rank"]  = rank + 1
            results.append(chunk)

        return results


# ── Singleton loader ───────────────────────────────────────────────────────

_index: Optional[BM25Index] = None


def get_bm25_index(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> BM25Index:
    """Returning cached BM25 index, building it if not yet initialized."""
    global _index
    if _index is None:
        chunks = load_all_chunks(chunk_size=chunk_size, overlap=overlap)
        _index = BM25Index(chunks)
    return _index


def search(query: str, top_k: int = 5) -> list[dict]:
    """Searching BM25 index and returning top_k results."""
    return get_bm25_index().search(query, top_k=top_k)