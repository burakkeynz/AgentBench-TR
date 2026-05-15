from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Storing incoming query request."""
    query: str


class QueryResponse(BaseModel):
    """Storing query response with answer and metadata."""
    answer: str
    confidence: float
    trace_id: str


class MetricsResponse(BaseModel):
    """Storing aggregated evaluation metrics."""
    consistency: float
    hallucination_rate: float
    avg_cost: float
    avg_latency: float


class Tracesummary(BaseModel):
    """Storing a single trace summary for listing."""
    trace_id: str
    query: str
    answer: str
    confidence: float
    created_at: str