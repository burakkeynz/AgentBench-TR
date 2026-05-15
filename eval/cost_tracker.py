from dotenv import load_dotenv
load_dotenv()

from storage.trace_store import get_agent_logs
from storage.database import get_db
from storage.models import EvalResult

# Pricing per 1M tokens (USD)
MODEL_PRICES = {
    "gpt-4o-mini": {
        "input":  0.150,
        "output": 0.600,
    },
    "gpt-4o": {
        "input":  2.50,
        "output": 10.00,
    },
    "gpt-4-turbo": {
        "input":  10.00,
        "output": 30.00,
    },
    "claude-haiku-4-5": {
        "input":  0.80,
        "output": 4.00,
    },
    "claude-sonnet-4-6": {
        "input":  3.00,
        "output": 15.00,
    },
}

# In-memory token usage log: {trace_id: [{model, input_tokens, output_tokens}]}
_token_log: dict[str, list] = {}


def log_token_usage(trace_id: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Logging token usage for a trace."""
    if trace_id not in _token_log:
        _token_log[trace_id] = []
    _token_log[trace_id].append({
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    })
    print(f"Logging token usage — trace {trace_id[:8]}: {model} in={input_tokens} out={output_tokens}.")


def calculate_cost(trace_id: str, model: str = "gpt-4o-mini") -> dict:
    """Calculating total cost for a trace from logged token usage."""
    entries = _token_log.get(trace_id, [])

    if not entries:
        # Estimating from agent_logs if no token log exists
        logs = get_agent_logs(trace_id)
        total_input  = sum(len(l.get("input",  "").split()) * 1.3 for l in logs)
        total_output = sum(len(l.get("output", "").split()) * 1.3 for l in logs)
        entries = [{"model": model, "input_tokens": int(total_input), "output_tokens": int(total_output)}]

    prices     = MODEL_PRICES.get(model, MODEL_PRICES["gpt-4o-mini"])
    total_cost = 0.0
    total_in   = 0
    total_out  = 0

    for entry in entries:
        p    = MODEL_PRICES.get(entry["model"], prices)
        cost = (entry["input_tokens"] / 1_000_000) * p["input"] + \
               (entry["output_tokens"] / 1_000_000) * p["output"]
        total_cost += cost
        total_in   += entry["input_tokens"]
        total_out  += entry["output_tokens"]

    total_cost = round(total_cost, 6)

    print(f"Calculating cost — trace {trace_id[:8]}: in={total_in} out={total_out} cost=${total_cost}.")

    return {
        "trace_id":      trace_id,
        "model":         model,
        "input_tokens":  total_in,
        "output_tokens": total_out,
        "cost_usd":      total_cost,
    }