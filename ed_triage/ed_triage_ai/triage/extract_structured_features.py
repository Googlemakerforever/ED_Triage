"""GenAI-assisted structured feature extraction with deterministic fallback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

from ed_triage_ai.triage.fallback_keyword_extractor import fallback_keyword_extractor
from ed_triage_ai.triage.normalize_complaint import normalize_complaint
from ed_triage_ai.utils.ai_summary import DEFAULT_BASE_URL, DEFAULT_MODEL, get_ai_api_key


class OpenAICompatibleFeatureExtractor:
    def __init__(self, api_key: str, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("OPENROUTER_FEATURE_MODEL", DEFAULT_MODEL)

    def extract(self, complaint: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "ED Triage AI"),
        }
        schema_prompt = (
            "Extract structured emergency-triage complaint features. "
            "Return JSON only. Do not assign a triage level. "
            "Use booleans for features and include confidence and ambiguity flags."
        )
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 400,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": schema_prompt},
                    {"role": "user", "content": complaint},
                ],
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        return json.loads(content)


def extract_structured_features(complaint: str, provider: Optional[Any] = None) -> Dict[str, Any]:
    normalized = normalize_complaint(complaint)
    provider = provider or (OpenAICompatibleFeatureExtractor(get_ai_api_key()) if get_ai_api_key() else None)

    if provider is not None:
        try:
            extracted = provider.extract(normalized)
            extracted.setdefault("features", {})
            extracted.setdefault("context", {})
            extracted["context"].setdefault("extractor", "genai")
            extracted["context"].setdefault("confidence", 0.75)
            extracted["context"].setdefault("ambiguity_flags", [])
            extracted["context"].setdefault("temporal_modifiers", [])
            extracted["context"].setdefault("negations", [])
            return extracted
        except Exception:
            pass

    return fallback_keyword_extractor(normalized)
