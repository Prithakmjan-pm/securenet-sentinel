import os
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from state import InvestigationState, LogEvidenceOutput, RiskOutput, ReportOutput

DATA_PATH = Path(__file__).parent / "data.json"

with open(DATA_PATH, "r") as f:
    DATASET = json.load(f)

_ALERTS_BY_ID = {a["id"]: a for a in DATASET["alerts"]}


def get_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    """Lazy LLM init so importing this module (e.g. from Streamlit) doesn't
    crash at import time if OPENROUTER_API_KEY isn't set yet — the error only
    surfaces when an investigation is actually run, with a clear message."""
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running an investigation, "
            "e.g. `export OPENROUTER_API_KEY=sk-or-v1-...`"
        )
    model_name = model_name or os.getenv("SECURENET_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def fallback_models() -> list[str]:
    configured = os.getenv("SECURENET_FALLBACK_MODELS")
    if configured:
        return [model.strip() for model in configured.split(",") if model.strip()]
    return [
        os.getenv("SECURENET_FALLBACK_MODEL", "google/gemma-4-31b-it:free"),
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ]


def invoke_with_fallback(prompt, request, output_schema=None):
    models = [os.getenv("SECURENET_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")]
    models.extend(model for model in fallback_models() if model not in models)
    errors = []
    for model_name in models:
        llm = get_llm(model_name)
        for attempt in range(2):
            try:
                prompt_to_use = prompt
                if output_schema is not None:
                    prompt_to_use = prompt.partial(
                        output_format=(
                            "Return only one valid JSON object, with no Markdown or explanation, "
                            f"matching this schema: {json.dumps(output_schema.model_json_schema())}"
                        )
                    )
                result = (prompt_to_use | llm).invoke(request)
                if result is None or not getattr(result, "content", None):
                    raise ValueError("Model returned no output.")
                if output_schema is not None:
                    content = result.content.strip()
                    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
                    if fenced:
                        content = fenced.group(1)
                    else:
                        start, end = content.find("{"), content.rfind("}")
                        if start < 0 or end <= start:
                            raise ValueError("Model did not return a JSON object.")
                        content = content[start:end + 1]
                    result = output_schema.model_validate(json.loads(content))
                return result
            except Exception as error:
                errors.append(f"{model_name}: {error}")
                if attempt == 0:
                    time.sleep(1)
    raise RuntimeError("; ".join(errors))


# --------------------------------------------------------------------------
# Deterministic helpers — kept OUT of the LLM so they can't hallucinate.
# Time arithmetic and rule-boolean logic should never be left to a model
# when a plain function can compute it correctly every time.
# --------------------------------------------------------------------------

def _parse_alert_time(timestamp: str) -> Optional[datetime]:
    try:
        return datetime.strptime(timestamp, "%d-%b-%Y %H:%M")
    except ValueError:
        return None


def _is_outside_working_hours(alert_time: Optional[datetime], working_hours: Optional[str]) -> Optional[bool]:
    """Returns True/False, or None if it can't be determined (e.g. on-call text)."""
    if alert_time is None or not working_hours or "to" not in working_hours:
        return None
    try:
        start_str, end_str = [s.strip() for s in working_hours.split("to")]
        start = datetime.strptime(start_str, "%H:%M").time()
        end = datetime.strptime(end_str, "%H:%M").time()
        return not (start <= alert_time.time() <= end)
    except ValueError:
        return None


def _recommended_action(severity: str, uncertainty_flag: bool) -> str:
    if uncertainty_flag:
        return "Request More Information"
    return {
        "Low": "Monitor",
        "Medium": "Investigate",
        "High": "Escalate to Security Team",
        "Critical": "Escalate to Security Team",
    }.get(severity, "Investigate")


def _requires_human_review(severity: str, uncertainty_flag: bool, evidence_completeness: str) -> bool:
    """Enforces the company rules programmatically rather than trusting the
    LLM to remember them: 'Critical or High-risk recommendation: Security
    personnel must review the case' and 'Evidence is incomplete or
    conflicting: must not make a confident unsupported conclusion.'"""
    return severity in ("High", "Critical") or uncertainty_flag or evidence_completeness != "Complete"


# --------------------------------------------------------------------------
# NODE 1: Context & Baseline Enricher (deterministic — no LLM call)
# --------------------------------------------------------------------------

def context_enricher_node(state: InvestigationState) -> Dict[str, Any]:
    alert_id = state["alert_id"]
    alert_data = _ALERTS_BY_ID.get(alert_id)

    if alert_data is None:
        return {"error": f"Alert ID '{alert_id}' was not found in the dataset."}

    target = alert_data["target"]
    user_info = DATASET["user_reference"].get(target)
    asset_info = DATASET["asset_reference"].get(target)

    alert_time = _parse_alert_time(alert_data["timestamp"])
    working_hours = user_info.get("working_hours") if user_info else None
    outside_hours = _is_outside_working_hours(alert_time, working_hours)

    is_sensitive = False
    if user_info and "Sensitive" in user_info.get("access_level", ""):
        is_sensitive = True
    if asset_info and asset_info.get("sensitivity") == "Confidential":
        is_sensitive = True

    context = {
        "target": target,
        "target_type": "user" if user_info else ("asset" if asset_info else "unresolved"),
        "user_profile": user_info,
        "asset_profile": asset_info,
        "outside_working_hours": outside_hours,  # True / False / None (undetermined, e.g. on-call)
        "is_sensitive_context": is_sensitive,
    }
    return {"raw_alert": alert_data, "logs": alert_data["logs"], "context": context}


# --------------------------------------------------------------------------
# NODE 2: Evidence & Log Correlator
# --------------------------------------------------------------------------

def log_correlator_node(state: InvestigationState) -> Dict[str, Any]:
    if state.get("error"):
        return {}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Agent 2 (Evidence & Log Correlator) for SecureNet. Analyze the supporting logs for "
         "alert {alert_id}. Build a chronological timeline, determine whether a normal business "
         "explanation accounts for the activity (e.g. a scheduled backup), and rate evidence "
         "completeness as Complete, Incomplete, or Conflicting. Base every claim only on the logs "
         "provided — do not assume events that are not in the log list. {output_format}"),
        ("human", "Alert: {raw_alert}\nLogs: {logs}"),
    ])

    request = {
        "alert_id": state["alert_id"],
        "raw_alert": state["raw_alert"],
        "logs": state["logs"],
    }
    try:
        evidence = invoke_with_fallback(prompt, request, LogEvidenceOutput)
    except Exception as error:
        return {"error": f"Evidence correlation (Agent 2) failed across available models: {error}"}

    return {"log_evidence": evidence.model_dump()}


# --------------------------------------------------------------------------
# NODE 3: Threat & Risk Evaluator
# --------------------------------------------------------------------------

def risk_evaluator_node(state: InvestigationState) -> Dict[str, Any]:
    if state.get("error"):
        return {}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Agent 3 (Threat & Risk Evaluator) for SecureNet. Compare the correlated log evidence "
         "and target context against the company escalation rules and severity definitions below. "
         "Select the single best-matching rule, assign a severity of exactly Low, Medium, High, or "
         "Critical, and set uncertainty_flag to true if the evidence was rated Incomplete or "
         "Conflicting. Never assign a confident severity when evidence is insufficient — under those "
         "conditions, prefer a lower severity plus uncertainty_flag=true over a high-confidence guess.\n\n"
         "Severity definitions: {severity_reference}\n"
         "Risk factors to weigh: {risk_factors}\n"
         "Company escalation rules: {rules}\n{output_format}"),
        ("human", "Alert: {raw_alert}\nTarget Context: {context}\nLog Evidence: {log_evidence}"),
    ])

    request = {
            "severity_reference": json.dumps(DATASET["severity_reference"]),
            "risk_factors": json.dumps(DATASET["risk_factors"]),
            "rules": json.dumps(DATASET["escalation_rules"]),
            "raw_alert": state["raw_alert"],
            "context": state["context"],
            "log_evidence": state["log_evidence"],
    }
    try:
        risk = invoke_with_fallback(prompt, request, RiskOutput)
    except Exception as error:
        return {"error": f"Risk evaluation (Agent 3) failed across available models: {error}"}

    return {"risk_evaluation": risk.model_dump()}


# --------------------------------------------------------------------------
# NODE 4: Incident Report & HITL Agent
# --------------------------------------------------------------------------

def summary_node(state: InvestigationState) -> Dict[str, Any]:
    if state.get("error"):
        return {}

    alert = state["raw_alert"]
    context = state["context"]
    evidence = state["log_evidence"]
    risk = state["risk_evaluation"]

    severity = risk["severity"]
    uncertainty_flag = risk.get("uncertainty_flag", False)
    evidence_completeness = evidence.get("evidence_completeness", "Incomplete")

    # Severity, review-flag, and recommended action are DERIVED deterministically
    # from Node 3's structured output rather than re-asked of an LLM here — a
    # policy-mandated escalation shouldn't be left to a second probabilistic call.
    requires_human_review = _requires_human_review(severity, uncertainty_flag, evidence_completeness)
    recommended_action = _recommended_action(severity, uncertainty_flag)

    # Only the free-text narrative is generated by the LLM.
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Agent 4 (Incident Report Writer) for SecureNet. Write a concise, factual "
         "investigation summary (3-5 sentences) for a security analyst, grounded only in the alert, "
         "context, evidence, and risk evaluation provided. Do not recommend or imply any automated "
         "action — analysis and recommendations only. If evidence was incomplete or conflicting, say "
         "so plainly rather than resolving the ambiguity yourself."),
        ("human",
         "Alert: {raw_alert}\nContext: {context}\nEvidence: {evidence}\nRisk Evaluation: {risk}"),
    ])
    try:
        narrative = invoke_with_fallback(prompt, {
            "raw_alert": alert, "context": context, "evidence": evidence, "risk": risk,
        }).content
        if not narrative:
            raise ValueError("Model returned an empty narrative.")
    except Exception as error:
        return {"error": f"Report writing (Agent 4) failed across available models: {error}"}

    report = ReportOutput(
        investigation_summary=narrative,
        severity=severity,
        requires_human_review=requires_human_review,
        recommended_action=recommended_action,
        evidence_reference=evidence.get("logs_examined", []),
        limitations=("Evidence rated as " + evidence_completeness + "; conclusion should be treated as provisional."
                     if evidence_completeness != "Complete" else None),
    )

    return {
        "final_report": report.model_dump(),
        "severity": severity,
        "requires_human_review": requires_human_review,
    }


# --------------------------------------------------------------------------
# BUILD THE LANGGRAPH WORKFLOW
# --------------------------------------------------------------------------

workflow = StateGraph(InvestigationState)

workflow.add_node("context_enricher", context_enricher_node)
workflow.add_node("log_correlator", log_correlator_node)
workflow.add_node("risk_evaluator", risk_evaluator_node)
workflow.add_node("summary_generator", summary_node)

workflow.set_entry_point("context_enricher")
workflow.add_edge("context_enricher", "log_correlator")
workflow.add_edge("log_correlator", "risk_evaluator")
workflow.add_edge("risk_evaluator", "summary_generator")
workflow.add_edge("summary_generator", END)

app_graph = workflow.compile()
