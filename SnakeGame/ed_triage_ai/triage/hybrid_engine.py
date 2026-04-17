"""Hybrid safety-first triage engine.

GenAI is used only to extract structured complaint features. Deterministic rules
run before ML, and ML is only used for non-obvious gray-zone cases.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ed_triage_ai.triage.apply_hard_overrides import apply_hard_overrides
from ed_triage_ai.triage.apply_uncertainty_escalation import apply_uncertainty_escalation
from ed_triage_ai.triage.combine_with_safety_floor import combine_with_safety_floor
from ed_triage_ai.triage.compute_derived_vitals import compute_derived_vitals
from ed_triage_ai.triage.extract_structured_features import extract_structured_features
from ed_triage_ai.triage.finalize_decision import finalize_decision
from ed_triage_ai.triage.normalize_complaint import normalize_complaint
from ed_triage_ai.triage.predict_acuity_ml import predict_acuity_ml
from ed_triage_ai.triage.validate_input import summarize_abnormal_vitals, validate_input


class HybridTriageEngine:
    def __init__(self, model: Any, feature_extractor: Optional[Any] = None):
        self.model = model
        self.feature_extractor = feature_extractor

    def run(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        errors, suspicious = validate_input(patient)
        normalized = normalize_complaint(str(patient.get("chief_complaint", "")))
        abnormal_vitals = summarize_abnormal_vitals(patient)

        if errors:
            return finalize_decision(
                {
                    "level": 3,
                    "reason": "Input validation error. Please review the entered values.",
                    "source": "validation_error",
                    "matched_rules": ["VAL_INPUT_ERROR"],
                    "ml_probabilities": {},
                },
                normalized,
                abnormal_vitals,
                {"features": {}, "context": {"errors": errors, "suspicious_inputs": suspicious, "extractor": "validation_only"}},
                {"suspicious_inputs": suspicious},
            )

        extracted = extract_structured_features(normalized, provider=self.feature_extractor)
        extracted.setdefault("context", {})
        extracted["context"].setdefault("suspicious_inputs", suspicious)
        derived = compute_derived_vitals(patient, extracted)

        hard_override = apply_hard_overrides(patient, extracted, derived)
        safety_floor = None if hard_override is not None else apply_uncertainty_escalation(patient, extracted, derived)

        ml_level = None
        ml_probabilities = {}
        if hard_override is None:
            ml_level, ml_probabilities = predict_acuity_ml(self.model, patient, normalized)

        final_decision = combine_with_safety_floor(hard_override, safety_floor, ml_level, ml_probabilities)
        return finalize_decision(final_decision, normalized, abnormal_vitals, extracted, derived)
