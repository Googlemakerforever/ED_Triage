"""OpenAI-compatible summary generation for triage results."""

from __future__ import annotations

import os
from typing import Any, Dict

import requests

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1")


def get_ai_api_key() -> str:
    """Resolve OpenAI-compatible API key from environment or Streamlit secrets."""
    for env_name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    try:
        import streamlit as st

        for secret_name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
            value = str(st.secrets.get(secret_name, "")).strip()
            if value:
                return value
    except Exception:
        pass

    return ""


def ai_summary_available() -> bool:
    return bool(get_ai_api_key())


def _extract_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()


def generate_triage_summary(patient: Dict[str, Any], prediction: Any) -> str:
    api_key = get_ai_api_key()
    if not api_key:
        return ""

    system_prompt = (
        "You are assisting with an emergency department triage prototype. "
        "Write exactly 4 short bullet points. "
        "Do not invent new facts. "
        "Do not mention being an AI. "
        "State clearly that this is educational support, not medical advice."
    )

    user_prompt = f"""
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

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "ED Triage AI"),
    }

    try:
        response = requests.post(
            f"{DEFAULT_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": DEFAULT_MODEL,
                "temperature": 0.2,
                "max_tokens": 220,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=20,
        )
        response.raise_for_status()
        text = _extract_text(response.json())
        return text or "AI summary unavailable for this request."
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            return "AI summary unavailable because the API key is invalid."
        if status == 402:
            return "AI summary unavailable because the provider account has no credit."
        if status == 429:
            return "AI summary unavailable because the API key is rate-limited or out of quota."
        return f"AI summary unavailable due to API error ({status})."
    except requests.RequestException:
        return "AI summary unavailable because the network request failed."
