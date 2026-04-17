"""Deterministic feature extraction fallback when GenAI is unavailable."""

from __future__ import annotations

from typing import Any, Dict

from ed_triage_ai.triage.keyword_sets import FEATURE_ALIASES, FUZZY_CRITICAL_FEATURES
from ed_triage_ai.triage.normalize_complaint import detect_negations, detect_temporal_modifiers, find_phrase_matches, flatten_aliases


def fallback_keyword_extractor(normalized_complaint: str) -> Dict[str, Any]:
    features: Dict[str, bool] = {}
    matched_categories = []
    for feature, aliases in FEATURE_ALIASES.items():
        hits = find_phrase_matches(
            normalized_complaint,
            aliases,
            allow_fuzzy=feature in FUZZY_CRITICAL_FEATURES,
        )
        features[feature] = bool(hits)
        if hits:
            matched_categories.append(feature)

    negations = detect_negations(normalized_complaint, flatten_aliases(FEATURE_ALIASES))
    temporal_modifiers = detect_temporal_modifiers(normalized_complaint)
    ambiguity_flags = []
    if any(term in normalized_complaint for term in ["maybe", "possible", "unclear", "?"]):
        ambiguity_flags.append("ambiguous_language")
    if "resolved" in temporal_modifiers or "transient" in temporal_modifiers:
        ambiguity_flags.append("transient_or_resolved_symptom")

    return {
        "features": features,
        "context": {
            "negations": negations,
            "temporal_modifiers": temporal_modifiers,
            "confidence": 0.62 if matched_categories else 0.4,
            "ambiguity_flags": ambiguity_flags,
            "extractor": "keyword_fallback",
            "matched_categories": matched_categories,
        },
    }
