"""Conservative safety-floor handling for ambiguous but concerning cases."""

from __future__ import annotations

from typing import Any, Dict, Optional


RuleDict = Dict[str, Any]


def compute_safety_floor(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    features = extracted.get("features", {})
    context = extracted.get("context", {})
    critical_flags = context.get("critical_flags", {})
    confidence = float(context.get("confidence", 0.5) or 0.5)
    ambiguity_flags = list(context.get("ambiguity_flags", []))
    temporal_modifiers = list(context.get("temporal_modifiers", []))

    if critical_flags.get("semantic_loc"):
        return {
            "level": 2,
            "reason": "Semantic loss-of-consciousness event requires a non-negotiable safety floor.",
            "matched_rules": ["UNC_SEMANTIC_LOC"],
            "source": "uncertainty_escalation",
        }

    if critical_flags.get("semantic_stroke"):
        return {
            "level": 2,
            "reason": "Semantic transient neurologic symptoms require a non-negotiable safety floor.",
            "matched_rules": ["UNC_SEMANTIC_STROKE"],
            "source": "uncertainty_escalation",
        }

    concerning_features = [
        name
        for name in [
            "chest_pain",
            "shortness_of_breath",
            "stroke_like_symptoms",
            "head_injury",
            "loss_of_consciousness",
            "open_injury",
            "dangerous_mechanism",
            "neurovascular_compromise",
            "sepsis_concern",
            "pregnancy_related",
            "suicidal_risk",
            "homicidal_risk",
            "overdose_or_ingestion",
        ]
        if features.get(name)
    ]

    if "loss_of_consciousness" in concerning_features and any(flag in temporal_modifiers for flag in ["resolved", "transient", "earlier_today"]):
        return {
            "level": 2,
            "reason": "Transient or resolved high-risk symptoms still require conservative escalation.",
            "matched_rules": ["UNC_TRANSIENT_RED_FLAG"],
            "source": "uncertainty_escalation",
        }

    if len(concerning_features) >= 2:
        return {
            "level": 2,
            "reason": "Multiple moderate-risk signals coexist, so acuity is escalated conservatively.",
            "matched_rules": ["UNC_MULTI_SIGNAL"],
            "source": "uncertainty_escalation",
        }

    if confidence < 0.55 and concerning_features:
        return {
            "level": 2,
            "reason": "Parsing confidence is low but dangerous terms are present, so a safety floor is applied.",
            "matched_rules": ["UNC_LOW_CONFIDENCE_RED_FLAG"],
            "source": "uncertainty_escalation",
        }

    if ambiguity_flags and concerning_features:
        return {
            "level": 3,
            "reason": "Ambiguous but potentially serious symptoms require a conservative floor.",
            "matched_rules": ["UNC_AMBIGUOUS_CONCERNING_TEXT"],
            "source": "uncertainty_escalation",
        }

    if features.get("chest_pain") or features.get("stroke_like_symptoms") or features.get("head_injury"):
        return {
            "level": 2,
            "reason": "High-morbidity symptoms should not be reassured solely by normal vitals.",
            "matched_rules": ["UNC_HIGH_MORBIDITY_NORMAL_VITALS"],
            "source": "uncertainty_escalation",
        }

    return None


def apply_uncertainty_escalation(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    return compute_safety_floor(payload, extracted, derived)
