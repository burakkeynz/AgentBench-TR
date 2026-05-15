import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from graph.state import AgentState
import os
from dotenv import load_dotenv
load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_SYSTEM = """Sen bir görev planlayıcısısın. Kullanıcının sorusunu analiz et ve soruyu yanıtlamak için gereken alt görevleri listele.
Sadece alt görevleri numaralı liste olarak döndür. Başka hiçbir şey yazma."""


def planner_agent(state: AgentState) -> AgentState:
    """Running PlannerAgent — decomposing query into sub-tasks."""
    t0 = time.time()

    prompt = f"{_SYSTEM}\n\nSoru: {state.query}"
    response = _llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Parsing numbered list into sub_tasks
    sub_tasks = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Stripping leading number+dot: "1. foo" → "foo"
        if line[0].isdigit() and len(line) > 2 and line[1] in ".):":
            line = line[2:].strip()
        sub_tasks.append(line)

    latency_ms = (time.time() - t0) * 1000

    state.sub_tasks = sub_tasks
    state.agent_logs.append({
        "agent": "planner",
        "input": state.query,
        "output": sub_tasks,
        "latency_ms": round(latency_ms, 2),
    })

    print(f"Running PlannerAgent — generated {len(sub_tasks)} sub-tasks in {latency_ms:.0f}ms.")
    return state