"""Compatibility wrapper for the safety-first ED guardrail layer."""

from ed_triage_ai.rules.safety_guardrails import (
    RuleResult,
    apply_clinical_rules,
    apply_trauma_overrides,
    detect_high_risk_mechanism,
    detect_neurovascular_compromise,
    evaluate_safety_guardrails,
    has_any_keyword,
    normalize_complaint,
    validate_required_test_case,
    validate_trauma_override_cases,
)

__all__ = [
    "RuleResult",
    "apply_clinical_rules",
    "apply_trauma_overrides",
    "detect_high_risk_mechanism",
    "detect_neurovascular_compromise",
    "evaluate_safety_guardrails",
    "has_any_keyword",
    "normalize_complaint",
    "validate_required_test_case",
    "validate_trauma_override_cases",
]
