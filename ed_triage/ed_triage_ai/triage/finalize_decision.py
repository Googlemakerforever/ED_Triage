"""Audit shaping for hybrid triage output."""

from __future__ import annotations

from typing import Any, Dict


def finalize_decision(
    decision: Dict[str, Any],
    normalized_complaint: str,
    abnormal_vitals: list[str],
    extracted_features: Dict[str, Any],
    derived_features: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "level": int(decision["level"]),
        "reason": decision["reason"],
        "source": decision["source"],
        "matched_rules": list(decision.get("matched_rules", [])),
        "audit": {
            "normalized_complaint": normalized_complaint,
            "abnormal_vitals": abnormal_vitals,
            "extracted_features": extracted_features,
            "derived_features": derived_features,
            "ml_probabilities": decision.get("ml_probabilities", {}),
            "critical_flags": extracted_features.get("context", {}).get("critical_flags", {}),
            "semantic_matches": extracted_features.get("context", {}).get("semantic_matches", {}),
            "safety_floor_reason": decision.get("safety_floor_reason"),
        },
    }
