import time
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from graph.state import AgentState

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _build_context(docs: list) -> str:
    """Building context string from retrieved docs."""
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"[{i+1}] (kaynak: {doc['source']})\n{doc['text']}")
    return "\n\n".join(parts)


def validator_agent(state: AgentState) -> AgentState:
    """Running ValidatorAgent — generating claims and grounding each against retrieved docs."""
    t0 = time.time()

    context = _build_context(state.retrieved_docs)

    prompt = f"""Aşağıdaki bağlam belgelerini kullanarak soruyu yanıtla.
Yanıtını JSON formatında ver. Sadece JSON döndür, başka hiçbir şey yazma.

Format:
{{
  "claims": [
    {{
      "claim": "iddia metni",
      "source_index": 1,
      "validated": true,
      "hallucination": false
    }}
  ]
}}

Soru: {state.query}

Bağlam:
{context}

Kurallar:
- Her claim için bağlamda gerçekten geçen bilgileri kullan.
- Bağlamda geçmiyorsa validated=false, hallucination=true yap.
- source_index bağlamdaki [N] numarasına karşılık gelir."""

    response = _llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Parsing JSON response
    try:
        raw_clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_clean)
        claims = data.get("claims", [])
    except json.JSONDecodeError:
        claims = [{
            "claim": raw,
            "source_index": 0,
            "validated": False,
            "hallucination": True,
        }]

    latency_ms = (time.time() - t0) * 1000
    flagged = sum(1 for c in claims if c.get("hallucination"))

    state.validated_claims = claims
    state.agent_logs.append({
        "agent": "validator",
        "input": f"{len(state.retrieved_docs)} docs",
        "output": f"{len(claims)} claims, {flagged} hallucination flags",
        "latency_ms": round(latency_ms, 2),
    })

    print(f"Running ValidatorAgent — {len(claims)} claims, {flagged} flagged in {latency_ms:.0f}ms.")
    return state