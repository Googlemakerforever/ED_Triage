"""Validation helpers for hybrid triage input handling."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ed_triage_ai.triage.keyword_sets import REQUIRED_FIELDS, VALID_RANGES


def validate_input(payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    suspicious: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in payload or payload.get(field) in (None, ""):
            errors.append(f"missing required input: {field}")

    for field, (lower, upper) in VALID_RANGES.items():
        if field not in payload or payload.get(field) in (None, ""):
            continue
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric")
            continue
        if value < lower or value > upper:
            errors.append(f"{field} must be between {lower} and {upper}")
        elif field in {"heart_rate", "respiratory_rate", "systolic_bp", "oxygen_saturation"} and value in {lower, upper}:
            suspicious.append(f"boundary_{field}")

    if str(payload.get("sex", "")) not in {"Female", "Male", "Other"}:
        errors.append("sex must be Female, Male, or Other")

    return errors, suspicious


def summarize_abnormal_vitals(payload: Dict[str, Any]) -> List[str]:
    abnormal: List[str] = []
    try:
        if float(payload.get("oxygen_saturation", 100)) < 92:
            abnormal.append(f"oxygen_saturation={float(payload['oxygen_saturation']):.0f}%")
        if float(payload.get("systolic_bp", 120)) < 90:
            abnormal.append(f"systolic_bp={float(payload['systolic_bp']):.0f} mmHg")
        if float(payload.get("heart_rate", 80)) > 110:
            abnormal.append(f"heart_rate={float(payload['heart_rate']):.0f} bpm")
        if float(payload.get("respiratory_rate", 16)) > 24:
            abnormal.append(f"respiratory_rate={float(payload['respiratory_rate']):.0f}/min")
        if float(payload.get("temperature", 36.8)) >= 38.0:
            abnormal.append(f"temperature={float(payload['temperature']):.1f} C")
    except Exception:
        return abnormal
    return abnormal
