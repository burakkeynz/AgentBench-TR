import time
from dotenv import load_dotenv
load_dotenv()

from graph.state import AgentState
from retrieval.hybrid import search as hybrid_search


def search_agent(state: AgentState) -> AgentState:
    """Running SearchAgent: retrieving docs for each sub-task via hybrid search."""
    t0 = time.time()

    all_docs = []
    seen_ids = set()

    for task in state.sub_tasks:
        results = hybrid_search(task, top_k=5)
        for doc in results:
            doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("text", "")[:60]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    latency_ms = (time.time() - t0) * 1000

    state.retrieved_docs = all_docs
    state.agent_logs.append({
        "agent": "search",
        "input": state.sub_tasks,
        "output": f"{len(all_docs)} unique chunks retrieved",
        "latency_ms": round(latency_ms, 2),
    })

    print(f"Running SearchAgent — retrieved {len(all_docs)} unique chunks in {latency_ms:.0f}ms.")
    return state