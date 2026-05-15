"""
Building ChromaDB vector store with multilingual embeddings for semantic search.
"""

import os
from pathlib import Path

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).parent.parent / "storage" / "chroma"))
COLLECTION_NAME    = "agentbench_tr"
EMBED_MODEL        = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_collection = None


# ── Embedding function ─────────────────────────────────────────────────────

def _get_embedding_function():
    """Loading multilingual sentence-transformer embedding function."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


# ── Collection ─────────────────────────────────────────────────────────────

def get_collection():
    """Returning cached ChromaDB collection, creating it if not initialized."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    print(f"Initializing ChromaDB collection at {CHROMA_PERSIST_DIR}...")
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Collection '{COLLECTION_NAME}' ready — {_collection.count()} docs stored.")
    return _collection


# ── Indexing ───────────────────────────────────────────────────────────────

def index_chunks(chunks: list[dict], batch_size: int = 64) -> None:
    """Embedding and storing chunks into ChromaDB collection."""
    collection = get_collection()

    existing_ids = set(collection.get()["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("Indexing skipped — all chunks already stored.")
        return

    print(f"Indexing {len(new_chunks)} new chunks into ChromaDB...")
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        collection.add(
            ids        = [c["chunk_id"] for c in batch],
            documents  = [c["text"]     for c in batch],
            metadatas  = [{"source": c["source"], "tokens": c["tokens"]} for c in batch],
        )
        print(f"  Indexing batch {i // batch_size + 1} — {len(batch)} chunks.")

    print(f"Indexing complete — {collection.count()} total chunks in collection.")


# ── Search ─────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 5) -> list[dict]:
    """Searching ChromaDB with dense embeddings and returning top_k results."""
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    output = []
    for rank, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        output.append({
            "chunk_id": doc_id,
            "source":   meta["source"],
            "text":     doc,
            "tokens":   meta.get("tokens", 0),
            "score":    float(1 - dist),  # cosine distance → similarity
            "rank":     rank + 1,
        })

    return output


# ── Build helper ───────────────────────────────────────────────────────────

def build_vector_store() -> None:
    """Loading all chunks and indexing into ChromaDB."""
    from retrieval.bm25_index import load_all_chunks
    chunks = load_all_chunks()
    index_chunks(chunks)