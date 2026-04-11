"""Google Gemini integration for optional narrative summaries."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def get_google_api_key() -> str:
    """Resolve Google AI API key from environment or Streamlit secrets."""
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    try:
        import streamlit as st

        for secret_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
            value = str(st.secrets.get(secret_name, "")).strip()
            if value:
                return value
    except Exception:
        pass

    return ""


def google_ai_available() -> bool:
    return bool(get_google_api_key())


def _extract_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_chunks = [part.get("text", "") for part in parts if part.get("text")]
    return "\n".join(text_chunks).strip()


def generate_triage_summary(patient: Dict[str, Any], prediction: Any) -> str:
    """Generate a concise AI narrative for the triage result."""
    api_key = get_google_api_key()
    if not api_key:
        return ""

    prompt = f"""
You are assisting with an emergency department triage prototype.
Write a concise clinical summary in 4 short bullet points.
Do not invent new facts.
Do not mention being an AI.
State clearly that this is educational support, not medical advice.

Patient inputs:
- Age: {patient.get('age')}
- Sex: {patient.get('sex')}
- Heart rate: {patient.get('heart_rate')} bpm
- Systolic BP: {patient.get('systolic_bp')} mmHg
- Diastolic BP: {patient.get('diastolic_bp')} mmHg
- Respiratory rate: {patient.get('respiratory_rate')}
- Oxygen saturation: {patient.get('oxygen_saturation')}%
- Temperature: {patient.get('temperature')} C
- Pain score: {patient.get('pain_score')}
- Chief complaint: {patient.get('chief_complaint')}

Model/system output:
- Triage level: {prediction.triage_level}
- Risk score: {prediction.risk_score:.3f}
- Risk category: {prediction.risk_category}
- Prediction source: {prediction.prediction_source}
- Explanation: {', '.join(prediction.explanation)}
- Override triggered: {prediction.override_triggered}
""".strip()

    try:
        response = requests.post(
            f"{GEMINI_API_BASE}/{DEFAULT_GEMINI_MODEL}:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 220,
                },
            },
            timeout=20,
        )
        response.raise_for_status()
        text = _extract_text(response.json())
        return text or "AI summary unavailable for this request."
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            return "Google AI summary unavailable because the API key is rate-limited or out of quota."
        if status == 403:
            return "Google AI summary unavailable because the API key is not authorized for this model."
        return f"Google AI summary unavailable due to API error ({status})."
    except requests.RequestException:
        return "Google AI summary unavailable because the network request failed."
