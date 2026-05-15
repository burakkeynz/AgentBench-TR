import time
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from graph.state import AgentState

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def synthesizer_agent(state: AgentState) -> AgentState:
    """Running SynthesizerAgent — building final answer from validated claims."""
    t0 = time.time()

    validated = [c for c in state.validated_claims if c.get("validated")]
    total = len(state.validated_claims)
    confidence_score = round(len(validated) / total, 2) if total > 0 else 0.0

    if not validated:
        state.final_answer = "Yeterli doğrulanmış bilgi bulunamadı."
        state.confidence_score = 0.0
        state.agent_logs.append({
            "agent": "synthesizer",
            "input": f"{total} claims",
            "output": "no validated claims",
            "latency_ms": 0.0,
        })
        print("Running SynthesizerAgent — no validated claims, skipping LLM.")
        return state

    claims_text = "\n".join(f"- {c['claim']}" for c in validated)

    prompt = f"""Aşağıdaki doğrulanmış bilgileri kullanarak soruya kısa ve net bir Türkçe cevap yaz.
Sadece cevabı yaz, başka hiçbir şey ekleme.

Soru: {state.query}

Doğrulanmış bilgiler:
{claims_text}"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    final_answer = response.content.strip()
    latency_ms = (time.time() - t0) * 1000

    state.final_answer = final_answer
    state.confidence_score = confidence_score
    state.agent_logs.append({
        "agent": "synthesizer",
        "input": f"{len(validated)}/{total} validated claims",
        "output": final_answer,
        "latency_ms": round(latency_ms, 2),
    })

    print(f"Running SynthesizerAgent — confidence={confidence_score}, latency={latency_ms:.0f}ms.")
    return state