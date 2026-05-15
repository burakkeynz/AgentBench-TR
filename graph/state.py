from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Storing the full pipeline state across all agents."""
    query: str = ""
    sub_tasks: list = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)
    validated_claims: list = field(default_factory=list)
    final_answer: str = ""
    confidence_score: float = 0.0
    agent_logs: list = field(default_factory=list)
    error: str = ""
    retry_count: int = 0