import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook-test/incident-approval")

import streamlit as st
import requests
from graph import app_graph, DATASET

st.set_page_config(page_title="SecureNet SOC Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ SecureNet Security Operations Center (SOC)")
st.caption("Multi-Agent AI Investigation System — analysis and recommendations only; "
           "all security actions remain under human control.")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is not set. Add it to your .env file, then restart the app.")
    st.stop()

# Initialize session state
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "current_alert_id" not in st.session_state:
    st.session_state.current_alert_id = None
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

SEVERITY_COLOR = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("🔍 Alert Selector")
alert_ids = [a["id"] for a in DATASET["alerts"]]
alert_labels = {a["id"]: f"{a['id']} - {a['type']} ({a['target']})" for a in DATASET["alerts"]}
selected_id = st.sidebar.selectbox("Select Alert to Investigate", alert_ids, format_func=lambda i: alert_labels[i])

if st.sidebar.button("⚡ Run 4-Agent Pipeline", use_container_width=True):
    with st.spinner(f"Running Context → Correlator → Risk → Report agents for {selected_id}..."):
        try:
            final_state = app_graph.invoke({"alert_id": selected_id})
            st.session_state.current_result = final_state
            st.session_state.current_alert_id = selected_id
        except Exception as e:
            st.session_state.current_result = {"error": f"Pipeline failed: {e}"}
            st.session_state.current_alert_id = selected_id

st.sidebar.divider()
st.sidebar.subheader("📊 System Status")
st.sidebar.write(f"Model: `{os.getenv('SECURENET_MODEL', 'nvidia/nemotron-3-ultra-550b-a55b:free')}`")
st.sidebar.write(f"LangSmith tracing: {'configured' if LANGCHAIN_API_KEY else 'not configured'}")
st.sidebar.caption("Governance: AI recommends only — a human analyst decides every action.")

# --------------------------------------------------------------------------
# Main display
# --------------------------------------------------------------------------
res = st.session_state.current_result

if res is None:
    st.info("👈 Select an alert from the sidebar and click **Run 4-Agent Pipeline** to start an investigation.")

elif res.get("error"):
    st.error(f"Investigation could not be completed: {res['error']}")
    if res.get("n8n_response"):
        st.write("n8n returned:")
        st.json(res["n8n_response"])

else:
    report = res["final_report"]
    severity = res.get("severity", report["severity"])
    requires_review = res.get("requires_human_review", report["requires_human_review"])

    # --- 1. Alert & context summary ---
    alert = res["raw_alert"]
    context = res["context"]
    st.subheader(f"Alert {alert['id']} — {alert['type']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Severity", f"{SEVERITY_COLOR.get(severity, '')} {severity}")
    c2.metric("Target", context["target"])
    c3.metric("Human Review Required", "Yes" if requires_review else "No")

    with st.expander("🧭 Agent 1 — Target Context & Baseline", expanded=False):
        st.json(context)

    with st.expander("🧩 Agent 2 — Evidence & Log Correlation", expanded=False):
        st.json(res["log_evidence"])

    with st.expander("⚖️ Agent 3 — Risk Evaluation", expanded=False):
        st.json(res["risk_evaluation"])

    # --- 2. Final report ---
    st.divider()
    st.subheader("📄 Investigation Report — Agent 4")
    st.write(report["investigation_summary"])
    st.write(f"**Recommended action:** {report['recommended_action']}")
    if report.get("evidence_reference"):
        st.write(f"**Supporting log IDs:** {', '.join(report['evidence_reference'])}")
    if report.get("limitations"):
        st.warning(f"⚠️ {report['limitations']}")
    if requires_review:
        st.error("🔒 This case meets the company threshold for mandatory security-analyst review "
                  "before any action is taken.")

    # --- 3. Human-in-the-loop decision ---
    st.divider()
    st.subheader("👨‍💻 Human Security Analyst Decision")
    st.caption("These buttons record an analyst decision for the audit trail. "
               "No automated security action is ever taken by this system.")

    col1, col2, col3 = st.columns(3)

    def log_decision(action: str):
        payload = {
            "alert_id": res.get("alert_id", st.session_state.current_alert_id),
            "action": action,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        try:
            requests.post(N8N_WEBHOOK_URL, json=payload, timeout=3)
            payload["logged_to"] = "n8n webhook"
        except Exception:
            payload["logged_to"] = "local session only (n8n unreachable)"
        st.session_state.audit_log.append(payload)
        return payload

    with col1:
        if st.button("✅ Approve Escalation", use_container_width=True):
            p = log_decision("Approved Escalation")
            st.success(f"Logged: Approved Escalation ({p['logged_to']}).")

    with col2:
        if st.button("❌ Dismiss as False Positive", use_container_width=True):
            p = log_decision("Dismissed as False Positive")
            st.warning(f"Logged: Dismissed as False Positive ({p['logged_to']}).")

    with col3:
        if st.button("❓ Request Additional Evidence", use_container_width=True):
            p = log_decision("Requested Additional Evidence")
            st.info(f"Logged: Requested Additional Evidence ({p['logged_to']}).")

    if st.session_state.audit_log:
        with st.expander("📝 Session Audit Trail"):
            st.json(st.session_state.audit_log)

# --------------------------------------------------------------------------
# Batch test runner — satisfies the "testing across scenarios" requirement
# --------------------------------------------------------------------------
st.sidebar.divider()
if st.sidebar.button("🧪 Run All 12 Alerts (Test Suite)", use_container_width=True):
    rows = []
    progress = st.sidebar.progress(0)
    for i, aid in enumerate(alert_ids):
        try:
            out = app_graph.invoke({"alert_id": aid})
            if out.get("error"):
                rows.append({"alert_id": aid, "severity": "ERROR", "requires_human_review": "-",
                             "recommended_action": out["error"]})
            else:
                r = out["final_report"]
                rows.append({
                    "alert_id": aid,
                    "severity": r["severity"],
                    "requires_human_review": r["requires_human_review"],
                    "recommended_action": r["recommended_action"],
                })
        except Exception as e:
            rows.append({"alert_id": aid, "severity": "ERROR", "requires_human_review": "-",
                         "recommended_action": str(e)})
        progress.progress((i + 1) / len(alert_ids))
    st.session_state.test_results = rows

if st.session_state.get("test_results"):
    st.divider()
    st.subheader("🧪 Testing Record — All 12 Alerts")
    st.table(st.session_state.test_results)
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["alert_id", "severity", "requires_human_review", "recommended_action"])
    writer.writeheader()
    writer.writerows(st.session_state.test_results)
    st.download_button("Download testing_record.csv", buf.getvalue(), file_name="testing_record.csv")
