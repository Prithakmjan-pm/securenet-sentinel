# SecureNet Alert Investigation System

A 4-agent LangGraph pipeline (Context → Correlator → Risk → Report) that investigates
security alerts, exposed two ways:
1. A Streamlit dashboard for an analyst to run investigations directly.
2. A FastAPI service so an n8n workflow can call each agent as its own node.

## Files

| File | Role |
|---|---|
| `data.json` | Alerts, logs, user/asset reference, severity definitions, risk factors, escalation rules |
| `state.py` | Pydantic schemas for each agent's structured output + the shared LangGraph state |
| `graph.py` | The 4 agent node functions + the compiled LangGraph pipeline (`app_graph`) |
| `app.py` | Streamlit dashboard — run investigations, review agent-by-agent output, HITL decisions |
| `agent_api.py` | FastAPI service exposing each agent as its own HTTP endpoint, for n8n |
| `requirements.txt` | All dependencies for both entry points |
| `.env.example` | Copy to `.env` and fill in your own values |

## LLM provider: OpenRouter (free tier)

This project calls the LLM through **OpenRouter** instead of OpenAI directly, using
a free model that supports tool calling / structured output (default:
`openai/gpt-oss-120b:free`). Because OpenRouter's endpoint is OpenAI-compatible,
the code still uses `langchain_openai.ChatOpenAI` — just pointed at
`https://openrouter.ai/api/v1` with your OpenRouter key. See `get_llm()` in
`graph.py`.

Agents 2 and 3 use `with_structured_output()`, which relies on the model
supporting function/tool calling. Not every free OpenRouter model does this
reliably — if you swap `SECURENET_MODEL`, pick one that explicitly advertises
tool calling and structured outputs (e.g. `nex-agi/nex-n2-pro:free`,
`baidu/cobuddy:free`), or use the auto-selecting `openrouter/free` router,
which filters for models that support the features a request needs, tool
calling and structured outputs included.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your OPENROUTER_API_KEY
```

## Running the Streamlit dashboard

```bash
streamlit run app.py
```
Opens at http://localhost:8501 — select an alert, run the pipeline, review each
agent's output, and approve/dismiss/request more info.

## Running the API for n8n

```bash
uvicorn agent_api:app --reload --port 8000
```
Runs at http://localhost:8000. Endpoints:

- `POST /agents/context` — Context & Baseline agent
- `POST /agents/correlate` — Evidence & Log Correlator agent
- `POST /agents/risk` — Threat & Risk Evaluator agent
- `POST /agents/report` — Incident Report agent
- `POST /agents/investigate` — runs all 4 agents in one call
- `GET /health` — health check

Each endpoint returns the **full merged state**, so in n8n every HTTP Request node's
body can simply be `{{ $json }}` — the previous node's output — with no extra
Set/Merge node needed in between.

## n8n workflow shape

```
Webhook → Context agent (HTTP) → Correlator agent (HTTP) → Risk agent (HTTP)
        → Report agent (HTTP) → Severity router (IF) → Slack tool / Sheets tool
        → Respond to webhook
```

Both `app.py` and `agent_api.py` import from the same `graph.py` — there is only
one implementation of the agent logic, used by both the dashboard and n8n.

## Notes

- Governance rule enforced in code, not just prompted: `severity` in `{High, Critical}`,
  or `uncertainty_flag=true`, or `evidence_completeness != "Complete"` always forces
  `requires_human_review=True` (see `_requires_human_review()` in `graph.py`). No
  automated action is ever taken — every path ends in a recommendation for a human.
- `context_enricher_node` (Agent 1) is intentionally deterministic, not an LLM call —
  working-hours comparison and reference lookups are factual operations where an LLM
  adds risk of error without adding value. Agents 2–4 do the actual judgment-based
  reasoning.
- Free-tier OpenRouter models can be slower or occasionally queued compared to a
  paid tier, and some free models don't reliably support structured/JSON output —
  that showed up as `'NoneType' object is not iterable` when a model returned no
  tool call for Agent 2/3's structured schema. Agents 2, 3, and 4's LLM calls are
  now wrapped in try/except, so a model failure surfaces as a clean
  `{"error": "..."}` in the UI (and stops the pipeline) instead of an unhandled
  crash. If you see that error, it almost always means `SECURENET_MODEL` needs to
  be swapped to another `:free` model with solid tool-calling support.
