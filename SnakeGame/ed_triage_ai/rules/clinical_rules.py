"""Strict rule-based triage assignment for guaranteed clinical overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RuleResult:
    triage_level: int
    explanation: str


def _contains_any(text: str, keywords) -> bool:
    return any(keyword in text for keyword in keywords)


def _normal_vitals(hr: float, sbp: float, rr: float, spo2: float, temp: float) -> bool:
    return (
        60 <= hr <= 100
        and 100 <= sbp <= 140
        and 12 <= rr <= 20
        and spo2 >= 95
        and 36.0 <= temp < 38.0
    )


def apply_clinical_rules(input_data: Dict[str, float]) -> Optional[RuleResult]:
    """Return strict rule-based triage assignment, or None if no rule applies.

    Priority order is fixed: 1 -> 2 -> 3 -> 4 -> 5.
    First matching rule returns immediately.
    """
    hr = float(input_data.get("heart_rate", 0) or 0)
    sbp = float(input_data.get("systolic_bp", 0) or 0)
    rr = float(input_data.get("respiratory_rate", 0) or 0)
    spo2 = float(input_data.get("oxygen_saturation", 100) or 100)
    temp = float(input_data.get("temperature", 36.8) or 36.8)
    pain = float(input_data.get("pain_score", 0) or 0)
    complaint = str(input_data.get("chief_complaint", "")).lower().strip()

    # Level 1 (Immediate)
    if spo2 < 90:
        return RuleResult(1, "Level 1 assigned due to oxygen saturation below 90%.")
    if sbp < 90:
        return RuleResult(1, "Level 1 assigned due to systolic blood pressure below 90 mmHg.")
    if rr > 30:
        return RuleResult(1, "Level 1 assigned due to respiratory rate above 30.")
    if hr > 130 and ("chest pain" in complaint or "shortness of breath" in complaint):
        return RuleResult(1, "Level 1 assigned due to severe tachycardia with cardiopulmonary symptoms.")
    if _contains_any(
        complaint,
        ["cannot breathe", "unconscious", "not responsive", "cardiac arrest"],
    ):
        return RuleResult(1, "Level 1 assigned due to life-threatening chief complaint.")

    # Level 2 (High Risk)
    if "chest pain" in complaint:
        return RuleResult(2, "Level 2 assigned due to chest pain.")
    if _contains_any(complaint, ["stroke", "slurred speech", "weakness"]):
        return RuleResult(2, "Level 2 assigned due to neurologic high-risk symptoms.")
    if hr > 110:
        return RuleResult(2, "Level 2 assigned due to heart rate above 110.")
    if temp >= 39:
        return RuleResult(2, "Level 2 assigned due to temperature at or above 39 C.")
    if pain >= 7:
        return RuleResult(2, "Level 2 assigned due to severe pain score (>=7).")

    # Level 3 (Moderate)
    if 38 <= temp < 39:
        return RuleResult(3, "Level 3 assigned due to fever between 38 C and 39 C.")
    if _contains_any(complaint, ["vomiting", "abdominal pain"]):
        return RuleResult(3, "Level 3 assigned due to gastrointestinal moderate-risk complaint.")
    if _normal_vitals(hr, sbp, rr, spo2, temp) and _contains_any(
        complaint,
        ["multiple resources", "labs and imaging", "needs labs and imaging"],
    ):
        return RuleResult(3, "Level 3 assigned due to expected multi-resource evaluation with stable vitals.")

    # Level 4 (Low Risk)
    if _contains_any(complaint, ["sprain", "cut", "minor injury"]) and _normal_vitals(hr, sbp, rr, spo2, temp) and 2 <= pain <= 4:
        return RuleResult(4, "Level 4 assigned due to minor injury with normal vitals and low pain.")

    # Level 5 (Non-Urgent)
    if _normal_vitals(hr, sbp, rr, spo2, temp) and _contains_any(
        complaint,
        ["cold", "runny nose", "med refill", "medication refill"],
    ) and pain <= 1:
        return RuleResult(5, "Level 5 assigned due to non-urgent complaint and normal vitals.")

    return None


def validate_required_test_case() -> None:
    """Hard validation for required safety scenario."""
    test_case = {
        "oxygen_saturation": 88,
        "systolic_bp": 85,
        "heart_rate": 120,
        "respiratory_rate": 24,
        "temperature": 37.2,
        "pain_score": 6,
        "chief_complaint": "chest pain",
    }
    result = apply_clinical_rules(test_case)
    if result is None or result.triage_level != 1:
        raise RuntimeError(
            "Clinical rule validation failed: Test Case 1 must return Level 1 "
            "(O2=88, BP=85, chest pain)."
        )
