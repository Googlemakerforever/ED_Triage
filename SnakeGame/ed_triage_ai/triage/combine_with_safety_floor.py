"""Final no-downgrade combiner for safety floors and ML outputs."""

from __future__ import annotations

from typing import Any, Dict, Optional


RuleDict = Dict[str, Any]


def combine_with_safety_floor(
    hard_override: Optional[RuleDict],
    safety_floor: Optional[RuleDict],
    ml_level: Optional[int],
    ml_probabilities: Optional[Dict[str, float]],
) -> RuleDict:
    if hard_override is not None:
        return {**hard_override, "ml_probabilities": ml_probabilities or {}}

    if ml_level is None:
        return {**(safety_floor or {"level": 3, "reason": "No confident path was available.", "matched_rules": ["FALLBACK_NO_PATH"], "source": "uncertainty_escalation"}), "ml_probabilities": ml_probabilities or {}}

    if safety_floor is not None and ml_level > int(safety_floor["level"]):
        return {**safety_floor, "ml_probabilities": ml_probabilities or {}}

    if safety_floor is not None and ml_level < int(safety_floor["level"]):
        return {
            "level": ml_level,
            "reason": "ML predicted a higher-acuity gray-zone case above the uncertainty floor.",
            "matched_rules": ["ML_ESCALATED_ABOVE_FLOOR"],
            "source": "ml_prediction",
            "ml_probabilities": ml_probabilities or {},
        }

    return {
        "level": ml_level,
        "reason": "ML prediction used for a non-obvious case after safety screening.",
        "matched_rules": ["ML_GRAY_ZONE"],
        "source": "ml_prediction",
        "ml_probabilities": ml_probabilities or {},
    }
