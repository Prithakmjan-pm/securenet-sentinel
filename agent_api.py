from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from graph import (
    context_enricher_node,
    log_correlator_node,
    risk_evaluator_node,
    summary_node,
)

app = FastAPI(title="SecureNet LangGraph Agent API")


class StateIn(BaseModel):
    """Accepts whatever state has accumulated so far. Every field is
    optional because each endpoint only needs the subset its node reads —
    n8n just forwards the previous node's full JSON output as the next
    node's request body, so the shape grows as it moves through the chain."""
    alert_id: str
    raw_alert: Dict[str, Any] | None = None
    logs: list | None = None
    context: Dict[str, Any] | None = None
    log_evidence: Dict[str, Any] | None = None
    risk_evaluation: Dict[str, Any] | None = None
    final_report: Dict[str, Any] | None = None
    error: str | None = None

    class Config:
        extra = "allow"  # tolerate extra fields n8n may add (timestamps, etc.)


def _merge(state_in: StateIn, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Each endpoint returns the FULL merged state, not just its own delta —
    so the n8n HTTP Request node can pass its raw output straight to the
    next node's body with no extra Set/Merge node needed in between."""
    merged = state_in.model_dump()
    merged.update(updates)
    return merged


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/agents/context")
def run_context(state: StateIn) -> Dict[str, Any]:
    updates = context_enricher_node(state.model_dump())
    return _merge(state, updates)


@app.post("/agents/correlate")
def run_correlate(state: StateIn) -> Dict[str, Any]:
    updates = log_correlator_node(state.model_dump())
    return _merge(state, updates)


@app.post("/agents/risk")
def run_risk(state: StateIn) -> Dict[str, Any]:
    updates = risk_evaluator_node(state.model_dump())
    return _merge(state, updates)


@app.post("/agents/report")
def run_report(state: StateIn) -> Dict[str, Any]:
    updates = summary_node(state.model_dump())
    return _merge(state, updates)


@app.post("/agents/investigate")
def run_full_pipeline(state: StateIn) -> Dict[str, Any]:
    """Convenience endpoint: runs all 4 agents in one call, for cases where
    n8n should treat the whole investigation as a single node instead of
    four chained ones."""
    from graph import app_graph
    return app_graph.invoke(state.model_dump())
