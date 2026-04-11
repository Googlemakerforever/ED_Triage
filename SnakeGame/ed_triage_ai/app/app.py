"""Streamlit UI for ED triage AI."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure package imports work even when launched from inside app/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ed_triage_ai.models.predict import TriagePredictor
from ed_triage_ai.utils.ai_summary import ai_summary_available, generate_triage_summary
from ed_triage_ai.utils.config import DEFAULT_MODEL_PATH

st.set_page_config(page_title="ED Triage AI", page_icon="🏥", layout="wide")

st.title("Emergency Department Triage AI Assistant")
st.caption("Educational clinical decision support prototype. Not for direct patient care.")

if ai_summary_available():
    st.info("AI narrative summaries are enabled for this deployment.")
else:
    st.warning("AI summary is not configured. Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in secrets.")

model_path = Path(DEFAULT_MODEL_PATH)
if not model_path.exists():
    st.error("Model artifact not found. Train first: `python -m ed_triage_ai.models.train`")
    st.stop()

predictor = TriagePredictor(str(model_path))

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=110, value=52)
    sex = st.selectbox("Sex", options=["Female", "Male", "Other"], index=0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=20.0, max_value=240.0, value=98.0)
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=40.0, max_value=250.0, value=118.0)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=20.0, max_value=150.0, value=72.0)

with col2:
    respiratory_rate = st.number_input("Respiratory Rate", min_value=4.0, max_value=60.0, value=18.0)
    oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=50.0, max_value=100.0, value=97.0)
    temperature = st.number_input("Temperature (C)", min_value=33.0, max_value=43.0, value=37.0)
    pain_score = st.slider("Pain Score", min_value=0, max_value=10, value=4)
    chief_complaint = st.text_area(
        "Chief Complaint",
        value="chest pain and shortness of breath",
        height=90,
    )

if st.button("Predict Triage", type="primary"):
    patient = {
        "age": age,
        "sex": sex,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate,
        "oxygen_saturation": oxygen_saturation,
        "temperature": temperature,
        "pain_score": pain_score,
        "chief_complaint": chief_complaint,
    }

    result = predictor.predict(patient)
    ai_summary = ""
    if ai_summary_available():
        try:
            with st.spinner("Generating AI summary..."):
                ai_summary = generate_triage_summary(patient, result)
        except Exception as exc:
            ai_summary = f"AI summary unavailable: {exc}"

    color = {"High": "#b91c1c", "Medium": "#b45309", "Low": "#15803d"}[result.risk_category]

    st.subheader("Prediction")
    st.markdown(f"**Triage Level (ESI):** `{result.triage_level}`")
    st.markdown(
        f"**Risk Score (high-acuity probability):** <span style='color:{color};font-weight:700'>{result.risk_score:.3f}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Risk Category:** <span style='color:{color};font-weight:700'>{result.risk_category}</span>", unsafe_allow_html=True)

    st.subheader("Top 3 Reasons")
    for reason in result.explanation:
        st.write(f"- {reason}")

    if result.override_triggered:
        st.warning("Clinical safety override triggered.")
        for reason in result.override_reasons:
            st.write(f"- {reason}")

    st.subheader("Decision Source")
    st.write(result.prediction_source)

    if ai_summary:
        st.subheader("AI Clinical Summary")
        st.write(ai_summary)
