"""Simple clinical NLP utilities for ED chief complaints."""

from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd

KEYWORD_PATTERNS = {
    "kw_chest_pain": [r"chest pain", r"pressure in chest", r"angina"],
    "kw_shortness_of_breath": [r"shortness of breath", r"dyspnea", r"can'?t breathe"],
    "kw_stroke": [r"stroke", r"facial droop", r"slurred speech", r"weakness"],
    "kw_altered_mental_status": [r"altered mental", r"confusion", r"unresponsive"],
    "kw_syncope": [r"syncope", r"faint", r"passed out"],
    "kw_trauma": [r"trauma", r"accident", r"injury", r"fall"],
    "kw_bleeding": [r"bleeding", r"hemorrhage", r"blood loss"],
    "kw_abdominal_pain": [r"abdominal pain", r"stomach pain"],
    "kw_fever": [r"fever", r"febrile", r"chills"],
    "kw_vomiting": [r"vomit", r"emesis", r"nausea"],
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _contains_any(text: str, patterns: Iterable[str]) -> int:
    return int(any(re.search(pattern, text) for pattern in patterns))


def extract_keyword_flags(text: str) -> Dict[str, int]:
    normalized = normalize_text(text)
    return {
        feature: _contains_any(normalized, patterns)
        for feature, patterns in KEYWORD_PATTERNS.items()
    }


def append_keyword_features(df: pd.DataFrame, text_col: str = "chief_complaint") -> pd.DataFrame:
    flags = df[text_col].fillna("").map(extract_keyword_flags)
    keyword_df = pd.DataFrame(flags.tolist(), index=df.index)
    return pd.concat([df, keyword_df], axis=1)
