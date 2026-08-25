"""Deterministic hard overrides for obvious high-risk presentations."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ed_triage_ai.triage.keyword_sets import THRESHOLDS


RuleDict = Dict[str, Any]


def _result(level: int, reason: str, rule_id: str) -> RuleDict:
    return {"level": level, "reason": reason, "matched_rules": [rule_id], "source": "hard_override"}


def evaluate_level1_rules(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    f = extracted["features"]
    complaint = str(payload.get("chief_complaint", "")).lower()
    critical_flags = extracted.get("context", {}).get("critical_flags", {})
    spo2 = float(payload.get("oxygen_saturation", 100) or 100)
    sbp = float(payload.get("systolic_bp", 120) or 120)
    rr = float(payload.get("respiratory_rate", 16) or 16)

    # Safety-critical semantic checks do not depend on extractor confidence.
    if critical_flags.get("semantic_airway_compromise"):
        return _result(1, "Immediate life threat due to semantic airway-compromise detection.", "L1_SEMANTIC_AIRWAY")
    if "not breathing" in complaint or "apnea" in complaint:
        return _result(1, "Immediate life threat due to absent breathing.", "L1_APNEA")
    if "no pulse" in complaint or "pulseless" in complaint:
        return _result(1, "Immediate life threat due to pulseless presentation.", "L1_PULSELESS")
    if f.get("altered_mental_status") and ("unresponsive" in complaint or "unconscious" in complaint):
        return _result(1, "Immediate life threat due to unresponsive state.", "L1_UNRESPONSIVE")
    if f.get("airway_compromise"):
        return _result(1, "Immediate life threat due to airway compromise.", "L1_AIRWAY")
    if f.get("shortness_of_breath") and (spo2 < THRESHOLDS["critical_spo2"] or rr > THRESHOLDS["critical_rr"]):
        return _result(1, "Immediate life threat due to severe respiratory compromise.", "L1_RESPIRATORY_FAILURE")
    if spo2 < THRESHOLDS["critical_spo2"]:
        return _result(1, "Immediate life threat due to critical hypoxia.", "L1_CRITICAL_HYPOXIA")
    if sbp < THRESHOLDS["critical_sbp"]:
        return _result(1, "Immediate life threat due to profound hypotension.", "L1_CRITICAL_HYPOTENSION")
    if f.get("severe_bleeding"):
        return _result(1, "Immediate life threat due to severe bleeding.", "L1_MASSIVE_BLEEDING")
    if f.get("active_seizure"):
        return _result(1, "Immediate life threat due to ongoing seizure without recovery.", "L1_ACTIVE_SEIZURE")
    if critical_flags.get("semantic_severe_trauma") and any([spo2 < THRESHOLDS["critical_spo2"], sbp < THRESHOLDS["critical_sbp"], rr > THRESHOLDS["critical_rr"]]):
        return _result(1, "Immediate life threat due to semantic severe trauma with instability.", "L1_SEMANTIC_TRAUMA_INSTABILITY")
    if derived.get("shock_index") is not None and derived["shock_index"] >= 1.3 and f.get("dangerous_mechanism"):
        return _result(1, "Immediate life threat due to trauma with profound instability.", "L1_TRAUMA_SHOCK")
    return None


def evaluate_level2_rules(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    f = extracted["features"]
    critical_flags = extracted.get("context", {}).get("critical_flags", {})
    spo2 = float(payload.get("oxygen_saturation", 100) or 100)
    sbp = float(payload.get("systolic_bp", 120) or 120)
    hr = float(payload.get("heart_rate", 80) or 80)
    rr = float(payload.get("respiratory_rate", 16) or 16)

    if f.get("head_injury") and critical_flags.get("semantic_head_injury_red_flags"):
        return _result(2, "Head injury with semantic red-flag cluster requires urgent evaluation.", "L2_SEMANTIC_HEAD_INJURY")
    if critical_flags.get("semantic_loc"):
        return _result(2, "Semantic loss-of-consciousness event requires urgent evaluation.", "L2_SEMANTIC_LOC")
    if critical_flags.get("semantic_stroke"):
        return _result(2, "New focal neurologic symptoms are time-sensitive and require urgent stroke-level evaluation.", "L2_SEMANTIC_STROKE")
    if f.get("chest_pain"):
        return _result(2, "High-risk chest pain requires urgent evaluation.", "L2_CHEST_PAIN")
    if f.get("shortness_of_breath"):
        return _result(2, "Shortness of breath with deterioration risk requires urgent evaluation.", "L2_SHORTNESS_OF_BREATH")
    if f.get("stroke_like_symptoms"):
        return _result(2, "Stroke-like symptoms require time-sensitive evaluation.", "L2_STROKE_SIGNS")
    if f.get("head_injury") and any(f.get(flag) for flag in ["loss_of_consciousness", "vomiting", "confusion", "amnesia"]):
        return _result(2, "Head injury red flags require urgent evaluation.", "L2_HEAD_INJURY_RED_FLAGS")
    if f.get("open_injury"):
        return _result(2, "Open injury or exposed bone requires urgent evaluation.", "L2_OPEN_FRACTURE")
    if f.get("neurovascular_compromise"):
        return _result(2, "Neurovascular compromise requires urgent evaluation.", "L2_NEUROVASCULAR")
    if f.get("sepsis_concern") and (float(payload.get("temperature", 36.8) or 36.8) >= THRESHOLDS["high_fever"] or sbp < 90 or hr > 110):
        return _result(2, "Severe infection concern with concerning physiology.", "L2_SEPSIS")
    if f.get("overdose_or_ingestion"):
        return _result(2, "Overdose or ingestion concern requires urgent evaluation.", "L2_OVERDOSE")
    if f.get("suicidal_risk") or f.get("homicidal_risk"):
        return _result(2, "Psychiatric safety emergency requires urgent evaluation.", "L2_PSYCH_RISK")
    if f.get("dangerous_mechanism"):
        return _result(2, "Dangerous mechanism of injury requires urgent evaluation.", "L2_DANGEROUS_MECHANISM")
    if f.get("severe_allergic_reaction") or f.get("airway_compromise"):
        return _result(2, "Severe allergic reaction risk requires urgent evaluation.", "L2_ALLERGIC_REACTION")
    if f.get("pregnancy_related") and (f.get("severe_abdominal_pain") or f.get("severe_bleeding")):
        return _result(2, "Pregnancy-related emergency requires urgent evaluation.", "L2_PREGNANCY")
    if THRESHOLDS["critical_spo2"] <= spo2 <= THRESHOLDS["low_spo2"]:
        return _result(2, "Oxygen saturation is in a concerning range.", "L2_OXYGEN_RANGE")
    if THRESHOLDS["critical_sbp"] <= sbp <= THRESHOLDS["low_sbp"]:
        return _result(2, "Blood pressure is in a concerning range.", "L2_BP_RANGE")
    if hr > THRESHOLDS["critical_hr"]:
        return _result(2, "Heart rate is in a concerning range.", "L2_HR_RANGE")
    if rr > THRESHOLDS["critical_rr"]:
        return _result(2, "Respiratory rate is in a concerning range.", "L2_RR_RANGE")
    if f.get("shortness_of_breath") and "normal" == derived.get("hypoxia_bucket") and ("suddenly" in complaint or "chest" in complaint):
        return _result(2, "Potential for rapid respiratory deterioration despite near-normal vitals.", "L2_RAPID_DETERIORATION")
    return None


def evaluate_level3_rules(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    f = extracted["features"]
    pain = float(payload.get("pain_score", 0) or 0)
    complaint = str(payload.get("chief_complaint", "")).lower()
    critical_flags = extracted.get("context", {}).get("critical_flags", {})
    if critical_flags.get("semantic_migraine_pattern") and not critical_flags.get("semantic_stroke") and not critical_flags.get("semantic_possible_stroke"):
        return _result(3, "Classic recurrent migraine pattern without focal neurologic deficit should not auto-escalate to stroke acuity.", "L3_CLASSIC_MIGRAINE")
    if f.get("possible_fracture"):
        return _result(3, "Possible stable fracture likely needs multiple resources.", "L3_POSSIBLE_FRACTURE")
    if f.get("deformity"):
        return _result(3, "Traumatic deformity requires urgent workup.", "L3_DEFORMITY")
    if "unable to bear weight" in complaint or "cannot bear weight" in complaint:
        return _result(3, "Inability to bear weight requires urgent workup.", "L3_WEIGHT_BEARING")
    if "deep laceration" in complaint or "deep cut" in complaint:
        return _result(3, "Deep laceration likely needs repair and resources.", "L3_DEEP_LACERATION")
    if f.get("severe_abdominal_pain"):
        return _result(3, "Significant abdominal or pelvic pain requires urgent workup.", "L3_ABDOMINAL_PAIN")
    if pain >= 6 and any(f.get(name) for name in ["possible_fracture", "deformity", "severe_abdominal_pain", "minor_injury"]):
        return _result(3, "Significant pain supports urgent evaluation in this context.", "L3_PAIN_SUPPORT")
    return None


def evaluate_level4_rules(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    if extracted["features"].get("minor_injury") and derived.get("hypoxia_bucket") == "normal" and not derived.get("hypotension_flag"):
        return _result(4, "Minor stable problem likely needs only one simple resource.", "L4_MINOR_CASE")
    return None


def evaluate_level5_rules(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    if extracted["features"].get("low_resource_case") and derived.get("hypoxia_bucket") == "normal" and not derived.get("hypotension_flag") and float(payload.get("pain_score", 0) or 0) <= 1:
        return _result(5, "Trivial low-resource presentation.", "L5_LOW_RESOURCE")
    return None


def apply_hard_overrides(payload: Dict[str, Any], extracted: Dict[str, Any], derived: Dict[str, Any]) -> Optional[RuleDict]:
    for evaluator in (evaluate_level1_rules, evaluate_level2_rules, evaluate_level3_rules, evaluate_level4_rules, evaluate_level5_rules):
        result = evaluator(payload, extracted, derived)
        if result is not None:
            return result
    return None
