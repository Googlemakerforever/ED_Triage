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


DEFAULT_INPUTS = {
    "age": 52,
    "sex": "Female",
    "heart_rate": 98.0,
    "systolic_bp": 118.0,
    "diastolic_bp": 72.0,
    "respiratory_rate": 18.0,
    "oxygen_saturation": 97.0,
    "temperature": 37.0,
    "pain_score": 4,
    "chief_complaint": "chest pain and shortness of breath",
}

NUMERIC_FIELDS = {
    "age": int,
    "heart_rate": float,
    "systolic_bp": float,
    "diastolic_bp": float,
    "respiratory_rate": float,
    "oxygen_saturation": float,
    "temperature": float,
    "pain_score": int,
}


def _coerce_query_value(field: str, raw_value: str):
    if raw_value is None:
        return DEFAULT_INPUTS[field]

    if field == "sex":
        value = str(raw_value)
        return value if value in {"Female", "Male", "Other"} else DEFAULT_INPUTS[field]

    if field == "chief_complaint":
        return str(raw_value)

    caster = NUMERIC_FIELDS[field]
    try:
        return caster(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_INPUTS[field]


def _initialize_input_state() -> None:
    for field in DEFAULT_INPUTS:
        if field in st.session_state:
            continue
        st.session_state[field] = _coerce_query_value(field, st.query_params.get(field))


def _sync_inputs_to_query_params() -> None:
    # Persist current inputs in the URL so a browser reload restores them.
    for field in DEFAULT_INPUTS:
        value = st.session_state[field]
        serialized = str(value)
        if st.query_params.get(field) != serialized:
            st.query_params[field] = serialized

st.title("Emergency Department Triage AI Assistant")
st.caption("Educational clinical decision support prototype. Not for direct patient care.")

if ai_summary_available():
    st.info("AI narrative summaries are enabled.")
else:
    st.caption("Optional AI summaries are unavailable until `OPENROUTER_API_KEY` or `OPENAI_API_KEY` is configured.")

model_path = Path(DEFAULT_MODEL_PATH)


@st.cache_resource
def load_predictor() -> TriagePredictor:
    return TriagePredictor(str(model_path))


predictor = load_predictor()
_initialize_input_state()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=110, key="age")
    sex = st.selectbox("Sex", options=["Female", "Male", "Other"], key="sex")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=20.0, max_value=240.0, key="heart_rate")
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=40.0, max_value=250.0, key="systolic_bp")
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=20.0, max_value=150.0, key="diastolic_bp")

with col2:
    respiratory_rate = st.number_input("Respiratory Rate", min_value=4.0, max_value=60.0, key="respiratory_rate")
    oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=50.0, max_value=100.0, key="oxygen_saturation")
    temperature = st.number_input("Temperature (C)", min_value=33.0, max_value=43.0, key="temperature")
    pain_score = st.slider("Pain Score", min_value=0, max_value=10, key="pain_score")
    chief_complaint = st.text_area(
        "Chief Complaint",
        key="chief_complaint",
        height=90,
    )

_sync_inputs_to_query_params()

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
