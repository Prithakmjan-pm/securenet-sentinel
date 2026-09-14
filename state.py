from typing import TypedDict, List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Pydantic Schemas for Structured Agent Outputs
# --------------------------------------------------------------------------

class ContextOutput(BaseModel):
    """Output of the Context/Baseline agent: resolves the alert's target
    against user_reference / asset_reference before any risk judgment is made."""
    target_type: Literal["user", "asset", "unresolved"] = Field(
        description="Whether the alert target resolved to a known user, a known asset, or neither"
    )
    baseline_summary: str = Field(
        description="Plain-language summary of what 'normal' looks like for this target "
                    "(role, access level, working hours, data sensitivity)"
    )
    is_sensitive_context: bool = Field(
        description="True if the target has elevated access or touches confidential/sensitive data"
    )


class LogEvidenceOutput(BaseModel):
    logs_examined: List[str] = Field(description="IDs of examined logs, e.g. ['L018', 'L019']")
    timeline_summary: str = Field(description="Chronological summary of what occurred in the logs")
    has_benign_explanation: bool = Field(description="True if a normal business explanation exists")
    benign_explanation_details: Optional[str] = Field(
        default=None, description="Details of the benign explanation, if applicable"
    )
    evidence_completeness: Literal["Complete", "Incomplete", "Conflicting"] = Field(
        description="Whether the available evidence is complete, incomplete, or conflicting"
    )


class RiskOutput(BaseModel):
    applied_rule_situation: str = Field(description="Matching situation from the company escalation rules")
    applied_rule_action: str = Field(description="Action mandated by the matched escalation rule")
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Severity level per the company severity reference — must be exactly one of these four values"
    )
    justification: str = Field(description="Detailed, evidence-grounded risk justification")
    uncertainty_flag: bool = Field(
        description="True if evidence was incomplete or conflicting and the conclusion is provisional"
    )


class ReportOutput(BaseModel):
    """Output of the final Report/HITL agent. Kept structured (not free text)
    so the Streamlit UI can render badges/flags without parsing prose."""
    investigation_summary: str = Field(description="Human-readable investigation narrative")
    severity: Literal["Low", "Medium", "High", "Critical"]
    requires_human_review: bool = Field(
        description="True whenever severity is High or Critical, or evidence is incomplete/conflicting "
                    "— per company rule that Critical/High cases must be reviewed by security personnel"
    )
    recommended_action: Literal[
        "Monitor", "Verify Account", "Investigate", "Escalate to Security Team", "Request More Information"
    ] = Field(description="Recommended next step — a recommendation only, never an automated action")
    evidence_reference: List[str] = Field(description="Log/user/asset IDs that support the conclusion")
    limitations: Optional[str] = Field(
        default=None, description="Any gaps, assumptions, or ambiguity the system could not resolve"
    )


# --------------------------------------------------------------------------
# Shared LangGraph State
# --------------------------------------------------------------------------
# total=False: LangGraph nodes typically return only the keys they update,
# not the full state dict, so every key must be allowed to be absent.

class InvestigationState(TypedDict, total=False):
    alert_id: str
    raw_alert: Dict[str, Any]
    logs: List[Dict[str, Any]]

    context: Dict[str, Any]              # ContextOutput.model_dump()
    log_evidence: Dict[str, Any]         # LogEvidenceOutput.model_dump()
    risk_evaluation: Dict[str, Any]      # RiskOutput.model_dump()
    final_report: Dict[str, Any]         # ReportOutput.model_dump()

    # convenience top-level fields, mirrored from final_report so the
    # Streamlit UI and test harness can read them without digging into
    # nested dicts or parsing free text
    severity: str
    requires_human_review: bool

    # populated only if a node raises/handles an error, so the UI can
    # surface a failure state instead of silently showing a stale report
    error: Optional[str]
