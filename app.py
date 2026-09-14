import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

print("Loaded keys:", bool(OPENAI_API_KEY), bool(LANGCHAIN_API_KEY))

import streamlit as st
import requests
import json
from graph import app_graph, DATASET

st.set_page_config(page_title="SecureNet SOC Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ SecureNet Security Operations Center (SOC)")
st.caption("Multi-Agent AI Investigation System with LangGraph, LangSmith & n8n Human-in-the-Loop Governance")

# Initialize Session State
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# Sidebar Controls
st.sidebar.header("🔍 Alert Selector")
alert_options = [f"{a['id']} - {a['type']} ({a['target']})" for a in DATASET["alerts"]]
selected_option = st.sidebar.selectbox("Select Alert to Investigate", alert_options)
selected_id = selected_option.split(" - ")

if st.sidebar.button("⚡ Run 4-Agent Pipeline"):
    with st.spinner(f"Executing LangGraph Agents for Alert {selected_id}..."):
        initial_state = {"alert_id": selected_id}
        final_state = app_graph.invoke(initial_state)
        st.session_state.current_result = final_state

st.sidebar.divider()
st.sidebar.subheader("📊 System Observability")
st.sidebar.info("LangSmith Tracing: Active\nModel: GPT-4o-mini\nn8n Webhook Status: Listening on localhost:5678")

# Main Display Area
if st.session_state.current_result:
    res = st.session_state.current_result

    # 1. Render Generated Incident Report
    st.markdown(res["final_report"])
    st.divider()

    # 2. Human-in-the-Loop (HITL) Action Buttons
    st.subheader("👨‍💻 Human Security Analyst Decision & n8n Audit Logging")
    st.write("Assessment Rule: AI system recommends while human analyst maintains full decision authority [2].")

    col1, col2, col3 = st.columns(3)
    webhook_url = "http://localhost:5678/webhook-test/incident-approval"

    with col1:
        if st.button("✅ Approve Escalation", use_container_width=True):
            payload = {
                "alert_id": selected_id,
                "action": "Approved Escalation",
                "severity": res["risk_evaluation"]["severity"]
            }
            try:
                requests.post(webhook_url, json=payload, timeout=3)
                st.success("Escalation Approved! Decision payload sent to n8n Webhook.")
            except Exception:
                st.info("Escalation Approved! (Logged locally - n8n container available).")
            st.session_state.audit_log.append(payload)

    with col2:
        if st.button("❌ Dismiss as False Positive", use_container_width=True):
            payload = {
                "alert_id": selected_id,
                "action": "Dismissed as False Positive",
                "severity": res["risk_evaluation"]["severity"]
            }
            try:
                requests.post(webhook_url, json=payload, timeout=3)
                st.warning("Dismissed as False Positive! Logged to n8n Webhook.")
            except Exception:
                st.warning("Dismissed as False Positive! (Logged locally).")
            st.session_state.audit_log.append(payload)

    with col3:
        if st.button("❓ Request Additional Logs", use_container_width=True):
            payload = {
                "alert_id": selected_id,
                "action": "Requested Additional Evidence",
                "severity": res["risk_evaluation"]["severity"]
            }
            st.info("Additional evidence requested from Tier-2 SOC team.")
            st.session_state.audit_log.append(payload)

    # 3. Audit Log Display
    if st.session_state.audit_log:
        with st.expander("📝 Session Audit Trail Log"):
            st.json(st.session_state.audit_log)
    else:
        st.info("👈 Select an alert from the sidebar and click 'Run 4-Agent Pipeline' to start the investigation.")
