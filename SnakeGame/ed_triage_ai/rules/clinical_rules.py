"""Deterministic clinical rule layer with trauma-first hard overrides."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RuleResult:
    triage_level: int
    explanation: str
    source: str


TRAUMA_CORE_KEYWORDS = [
    "trauma",
    "injury",
    "injured",
    "accident",
    "crash",
    "collision",
    "fall",
    "fracture",
    "broken bone",
    "laceration",
    "bleeding",
    "burn",
    "sprain",
    "dislocation",
    "contusion",
    "bruise",
    "abrasion",
    "wound",
    "crush",
    "gunshot",
    "stab",
]

LEVEL1_TRAUMA_KEYWORDS = {
    "unresponsive": ["unresponsive", "unconscious", "not responsive", "failure to respond"],
    "apnea": ["not breathing", "apneic", "apnea", "stopped breathing"],
    "pulseless": ["no pulse", "pulseless", "without pulse"],
    "airway": ["airway compromise", "airway obstructed", "cannot maintain airway"],
    "respiratory_distress": ["severe respiratory distress", "agonal breathing", "gasping"],
    "hemorrhage": ["massive bleeding", "exsanguinating hemorrhage", "hemorrhaging badly", "bleeding out"],
    "major_trauma_ams": ["major trauma with altered mental status", "trauma altered mental status"],
    "no_bp": ["no measurable blood pressure", "unable to obtain blood pressure"],
    "no_pulse": ["no measurable pulse", "unable to obtain pulse"],
}

LEVEL2_TRAUMA_KEYWORDS = {
    "open_fracture": ["open fracture", "compound fracture", "bone exposed", "protruding bone"],
    "neurovascular": [
        "absent distal pulse",
        "no distal pulse",
        "numbness distal",
        "weakness distal",
        "cold limb",
        "cyanotic limb",
        "ischemic limb",
        "neurovascular compromise",
    ],
    "crush": ["crush injury", "crush accident"],
    "dislocation_nv": ["dislocation with neurovascular compromise"],
    "active_bleeding": ["significant active bleeding", "active bleeding", "heavy bleeding"],
    "head_red_flags": [
        "head injury with confusion",
        "head injury with vomiting",
        "head injury with loss of consciousness",
        "head injury with blackout",
        "confusion after head injury",
        "vomiting after head injury",
        "loss of consciousness after head injury",
        "blackout after head injury",
    ],
    "injury_chest_pain": ["chest pain after injury", "chest pain associated with injury", "chest pain after crash"],
    "injury_sob": [
        "shortness of breath after injury",
        "shortness of breath associated with injury",
        "difficulty breathing after injury",
    ],
    "compartment": ["compartment syndrome", "suspected compartment syndrome"],
    "burn": ["severe burn"],
    "mechanism": [
        "high speed collision",
        "high-speed collision",
        "rollover",
        "ejection",
        "pedestrian struck",
        "motorcycle crash",
        "major fall",
        "significant fall",
        "fall from height",
        "industrial accident",
        "weapon-related injury",
        "gunshot",
        "stab wound",
        "stabbing",
        "shooting",
    ],
}

LEVEL3_TRAUMA_KEYWORDS = {
    "fracture": ["closed fracture", "possible fracture", "possible broken bone", "suspected closed fracture", "broken bone"],
    "deformity": ["deformity", "limb deformity", "arm deformity", "leg deformity"],
    "weight_bearing": ["inability to bear weight", "cannot bear weight", "unable to bear weight"],
    "swelling": ["severe swelling after injury", "marked swelling", "significant swelling"],
    "laceration": ["deep laceration", "laceration needing repair", "deep cut"],
    "procedure": ["need imaging", "needs imaging", "likely need for imaging", "likely need for procedure"],
}

LEVEL4_TRAUMA_KEYWORDS = {
    "minor_injury": ["minor sprain", "finger injury", "toe injury", "localized mild trauma", "minor laceration", "bruise", "contusion", "mild burn"],
    "ambulatory": ["ambulatory", "walking", "functioning limb preserved", "moving limb"],
    "simple_resource": ["one simple resource needed", "single simple resource", "simple repair only"],
}

LEVEL5_TRAUMA_KEYWORDS = {
    "trivial": ["superficial abrasion", "tiny bruise", "very minor bump", "healed injury recheck", "recheck only", "paperwork only"],
    "intact": ["no deformity", "no functional loss", "full range of motion", "normal function"],
}

HEAD_INJURY_KEYWORDS = ["head injury", "head trauma", "hit head", "concussion", "blackout", "loss of consciousness", "loc"]
CHEST_INJURY_KEYWORDS = ["injury", "trauma", "crash", "collision", "fall", "assault", "hit in chest", "chest wall injury"]
SOB_INJURY_KEYWORDS = ["injury", "trauma", "crash", "collision", "fall", "assault", "hit in chest"]
NEUROVASCULAR_COMPROMISE_KEYWORDS = [
    "absent distal pulse",
    "no distal pulse",
    "numbness",
    "weakness",
    "cold limb",
    "cyanotic limb",
    "ischemic limb",
    "neurovascular compromise",
]


def normalize_complaint(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def has_any_keyword(text: str, keywords) -> bool:
    return any(keyword in text for keyword in keywords)


def detect_high_risk_mechanism(text: str) -> bool:
    return has_any_keyword(text, LEVEL2_TRAUMA_KEYWORDS["mechanism"])


def detect_neurovascular_compromise(text: str) -> bool:
    return has_any_keyword(text, NEUROVASCULAR_COMPROMISE_KEYWORDS)


def _contains_any(text: str, keywords) -> bool:
    return has_any_keyword(text, keywords)


def _contains_non_negated_keyword(text: str, keywords) -> bool:
    """Avoid false positives from phrases like 'no deformity' or 'without weakness'."""
    for keyword in keywords:
        pattern = rf"(?<!no\s)(?<!not\s)(?<!without\s){re.escape(keyword)}"
        if re.search(pattern, text):
            return True
    return False


def _normal_vitals(hr: float, sbp: float, rr: float, spo2: float, temp: float) -> bool:
    return (
        60 <= hr <= 100
        and 100 <= sbp <= 140
        and 12 <= rr <= 20
        and spo2 >= 95
        and 36.0 <= temp < 38.0
    )


def _is_trauma_context(text: str) -> bool:
    return has_any_keyword(text, TRAUMA_CORE_KEYWORDS) or detect_high_risk_mechanism(text)


def apply_trauma_overrides(input_data: Dict[str, float]) -> Optional[RuleResult]:
    """Trauma-specific hard override layer that always runs before default logic."""
    hr = float(input_data.get("heart_rate", 0) or 0)
    sbp = float(input_data.get("systolic_bp", 0) or 0)
    rr = float(input_data.get("respiratory_rate", 0) or 0)
    spo2 = float(input_data.get("oxygen_saturation", 100) or 100)
    temp = float(input_data.get("temperature", 36.8) or 36.8)
    pain = float(input_data.get("pain_score", 0) or 0)
    complaint = normalize_complaint(str(input_data.get("chief_complaint", "")))

    trauma_context = _is_trauma_context(complaint)
    nv_compromise = detect_neurovascular_compromise(complaint)
    head_injury = has_any_keyword(complaint, HEAD_INJURY_KEYWORDS)

    # Level 1 trauma overrides
    if has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["unresponsive"]):
        return RuleResult(1, "Level 1 trauma override triggered by unresponsive presentation.", "trauma_override")
    if has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["apnea"]):
        return RuleResult(1, "Level 1 trauma override triggered by apnea or absent breathing.", "trauma_override")
    if has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["pulseless"]):
        return RuleResult(1, "Level 1 trauma override triggered by pulseless presentation.", "trauma_override")
    if has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["airway"]):
        return RuleResult(1, "Level 1 trauma override triggered by airway compromise.", "trauma_override")
    if has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["respiratory_distress"]):
        return RuleResult(1, "Level 1 trauma override triggered by severe respiratory distress.", "trauma_override")
    if spo2 < 85 and trauma_context:
        return RuleResult(1, "Level 1 trauma override triggered by oxygen saturation below 85%.", "trauma_override")
    if sbp < 80 and trauma_context:
        return RuleResult(1, "Level 1 trauma override triggered by systolic blood pressure below 80 mmHg.", "trauma_override")
    if has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["hemorrhage"]):
        return RuleResult(1, "Level 1 trauma override triggered by exsanguinating hemorrhage concern.", "trauma_override")
    if has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["major_trauma_ams"]):
        return RuleResult(1, "Level 1 trauma override triggered by major trauma with altered mental status.", "trauma_override")
    if trauma_context and has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["no_bp"]):
        return RuleResult(1, "Level 1 trauma override triggered by no measurable blood pressure.", "trauma_override")
    if trauma_context and has_any_keyword(complaint, LEVEL1_TRAUMA_KEYWORDS["no_pulse"]):
        return RuleResult(1, "Level 1 trauma override triggered by no measurable pulse.", "trauma_override")

    # Level 2 trauma overrides
    if has_any_keyword(complaint, LEVEL2_TRAUMA_KEYWORDS["open_fracture"]):
        return RuleResult(2, "Level 2 trauma override triggered by open fracture concern.", "trauma_override")
    if nv_compromise and trauma_context:
        return RuleResult(2, "Level 2 trauma override triggered by neurovascular compromise.", "trauma_override")
    if has_any_keyword(complaint, LEVEL2_TRAUMA_KEYWORDS["crush"]):
        return RuleResult(2, "Level 2 trauma override triggered by crush injury.", "trauma_override")
    if has_any_keyword(complaint, LEVEL2_TRAUMA_KEYWORDS["dislocation_nv"]):
        return RuleResult(2, "Level 2 trauma override triggered by dislocation with neurovascular compromise.", "trauma_override")
    if has_any_keyword(complaint, LEVEL2_TRAUMA_KEYWORDS["active_bleeding"]):
        return RuleResult(2, "Level 2 trauma override triggered by significant active bleeding.", "trauma_override")
    if head_injury and has_any_keyword(complaint, ["confusion", "vomiting", "loss of consciousness", "blackout", "loc"]):
        return RuleResult(2, "Level 2 trauma override triggered by high-risk head injury features.", "trauma_override")
    if trauma_context and "chest pain" in complaint and has_any_keyword(complaint, CHEST_INJURY_KEYWORDS):
        return RuleResult(2, "Level 2 trauma override triggered by chest pain associated with injury.", "trauma_override")
    if trauma_context and (
        "shortness of breath" in complaint or "difficulty breathing" in complaint
    ) and has_any_keyword(complaint, SOB_INJURY_KEYWORDS):
        return RuleResult(2, "Level 2 trauma override triggered by shortness of breath associated with injury.", "trauma_override")
    if has_any_keyword(complaint, LEVEL2_TRAUMA_KEYWORDS["compartment"]):
        return RuleResult(2, "Level 2 trauma override triggered by compartment syndrome concern.", "trauma_override")
    if has_any_keyword(complaint, LEVEL2_TRAUMA_KEYWORDS["burn"]):
        return RuleResult(2, "Level 2 trauma override triggered by severe burn concern.", "trauma_override")
    if detect_high_risk_mechanism(complaint):
        return RuleResult(2, "Level 2 trauma override triggered by dangerous injury mechanism.", "trauma_override")
    if trauma_context and 85 <= spo2 <= 91:
        return RuleResult(2, "Level 2 trauma override triggered by oxygen saturation between 85% and 91%.", "trauma_override")
    if trauma_context and 80 <= sbp <= 89:
        return RuleResult(2, "Level 2 trauma override triggered by systolic blood pressure between 80 and 89 mmHg.", "trauma_override")
    if trauma_context and hr > 130:
        return RuleResult(2, "Level 2 trauma override triggered by heart rate above 130.", "trauma_override")
    if trauma_context and rr > 30:
        return RuleResult(2, "Level 2 trauma override triggered by respiratory rate above 30.", "trauma_override")

    # Level 3 trauma overrides
    if _contains_non_negated_keyword(complaint, LEVEL3_TRAUMA_KEYWORDS["fracture"]):
        return RuleResult(3, "Level 3 trauma override triggered by suspected fracture.", "trauma_override")
    if _contains_non_negated_keyword(complaint, LEVEL3_TRAUMA_KEYWORDS["deformity"]):
        return RuleResult(3, "Level 3 trauma override triggered by traumatic deformity.", "trauma_override")
    if _contains_non_negated_keyword(complaint, LEVEL3_TRAUMA_KEYWORDS["weight_bearing"]):
        return RuleResult(3, "Level 3 trauma override triggered by inability to bear weight.", "trauma_override")
    if _contains_non_negated_keyword(complaint, LEVEL3_TRAUMA_KEYWORDS["swelling"]):
        return RuleResult(3, "Level 3 trauma override triggered by severe swelling after injury.", "trauma_override")
    if _contains_non_negated_keyword(complaint, LEVEL3_TRAUMA_KEYWORDS["laceration"]):
        return RuleResult(3, "Level 3 trauma override triggered by deep laceration needing repair.", "trauma_override")
    if trauma_context and has_any_keyword(complaint, LEVEL3_TRAUMA_KEYWORDS["procedure"]):
        return RuleResult(3, "Level 3 trauma override triggered by likely imaging or procedure need.", "trauma_override")
    if trauma_context and 6 <= pain <= 10:
        return RuleResult(3, "Level 3 trauma override triggered by significant traumatic pain with stable features.", "trauma_override")

    # Level 4 trauma overrides
    if (
        has_any_keyword(complaint, LEVEL4_TRAUMA_KEYWORDS["minor_injury"])
        and _normal_vitals(hr, sbp, rr, spo2, temp)
        and 0 <= pain <= 5
        and (
            has_any_keyword(complaint, LEVEL4_TRAUMA_KEYWORDS["ambulatory"])
            or not _contains_non_negated_keyword(complaint, ["cannot bear weight", "unable to use", "deformity"])
        )
    ):
        return RuleResult(4, "Level 4 trauma override triggered by minor stable injury pattern.", "trauma_override")

    # Level 5 trauma overrides
    if (
        has_any_keyword(complaint, LEVEL5_TRAUMA_KEYWORDS["trivial"])
        and has_any_keyword(complaint, LEVEL5_TRAUMA_KEYWORDS["intact"])
        and _normal_vitals(hr, sbp, rr, spo2, temp)
        and pain <= 1
    ):
        return RuleResult(5, "Level 5 trauma override triggered by trivial injury presentation.", "trauma_override")

    return None


def _apply_non_trauma_rules(input_data: Dict[str, float]) -> Optional[RuleResult]:
    hr = float(input_data.get("heart_rate", 0) or 0)
    sbp = float(input_data.get("systolic_bp", 0) or 0)
    rr = float(input_data.get("respiratory_rate", 0) or 0)
    spo2 = float(input_data.get("oxygen_saturation", 100) or 100)
    temp = float(input_data.get("temperature", 36.8) or 36.8)
    pain = float(input_data.get("pain_score", 0) or 0)
    complaint = normalize_complaint(str(input_data.get("chief_complaint", "")))

    if spo2 < 90:
        return RuleResult(1, "Level 1 assigned due to oxygen saturation below 90%.", "default_logic")
    if sbp < 90:
        return RuleResult(1, "Level 1 assigned due to systolic blood pressure below 90 mmHg.", "default_logic")
    if rr > 30:
        return RuleResult(1, "Level 1 assigned due to respiratory rate above 30.", "default_logic")
    if hr > 130 and ("chest pain" in complaint or "shortness of breath" in complaint):
        return RuleResult(1, "Level 1 assigned due to severe tachycardia with cardiopulmonary symptoms.", "default_logic")
    if _contains_any(complaint, ["cannot breathe", "unconscious", "not responsive", "cardiac arrest"]):
        return RuleResult(1, "Level 1 assigned due to life-threatening chief complaint.", "default_logic")

    if "chest pain" in complaint:
        return RuleResult(2, "Level 2 assigned due to chest pain.", "default_logic")
    if _contains_any(complaint, ["stroke", "slurred speech", "weakness"]):
        return RuleResult(2, "Level 2 assigned due to neurologic high-risk symptoms.", "default_logic")
    if hr > 110:
        return RuleResult(2, "Level 2 assigned due to heart rate above 110.", "default_logic")
    if temp >= 39:
        return RuleResult(2, "Level 2 assigned due to temperature at or above 39 C.", "default_logic")
    if pain >= 7:
        return RuleResult(2, "Level 2 assigned due to severe pain score (>=7).", "default_logic")

    if 38 <= temp < 39:
        return RuleResult(3, "Level 3 assigned due to fever between 38 C and 39 C.", "default_logic")
    if _contains_any(complaint, ["vomiting", "abdominal pain"]):
        return RuleResult(3, "Level 3 assigned due to gastrointestinal moderate-risk complaint.", "default_logic")
    if _normal_vitals(hr, sbp, rr, spo2, temp) and _contains_any(
        complaint,
        ["multiple resources", "labs and imaging", "needs labs and imaging"],
    ):
        return RuleResult(3, "Level 3 assigned due to expected multi-resource evaluation with stable vitals.", "default_logic")

    if _contains_any(complaint, ["sprain", "cut", "minor injury"]) and _normal_vitals(hr, sbp, rr, spo2, temp) and 2 <= pain <= 4:
        return RuleResult(4, "Level 4 assigned due to minor injury with normal vitals and low pain.", "default_logic")

    if _normal_vitals(hr, sbp, rr, spo2, temp) and _contains_any(
        complaint,
        ["cold", "runny nose", "med refill", "medication refill"],
    ) and pain <= 1:
        return RuleResult(5, "Level 5 assigned due to non-urgent complaint and normal vitals.", "default_logic")

    return None


def apply_clinical_rules(input_data: Dict[str, float]) -> Optional[RuleResult]:
    """Run trauma overrides first, then the existing non-trauma rules."""
    trauma_result = apply_trauma_overrides(input_data)
    if trauma_result is not None:
        return trauma_result
    return _apply_non_trauma_rules(input_data)


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


def validate_trauma_override_cases() -> None:
    """Minimal unit-style trauma validation suite."""
    cases = [
        (
            {
                "heart_rate": 122,
                "respiratory_rate": 18,
                "oxygen_saturation": 82,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 72,
                "pain_score": 4,
                "chief_complaint": "motorcycle crash with trauma and severe respiratory distress",
            },
            1,
            "trauma_override",
        ),
        (
            {
                "heart_rate": 96,
                "respiratory_rate": 18,
                "oxygen_saturation": 98,
                "temperature": 36.9,
                "systolic_bp": 124,
                "diastolic_bp": 78,
                "pain_score": 5,
                "chief_complaint": "open fracture with bone exposed and cold limb after crash",
            },
            2,
            "trauma_override",
        ),
        (
            {
                "heart_rate": 88,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 122,
                "diastolic_bp": 76,
                "pain_score": 7,
                "chief_complaint": "possible broken bone with deformity and unable to bear weight",
            },
            3,
            "trauma_override",
        ),
        (
            {
                "heart_rate": 82,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 118,
                "diastolic_bp": 74,
                "pain_score": 3,
                "chief_complaint": "minor sprain ambulatory functioning limb preserved",
            },
            4,
            "trauma_override",
        ),
        (
            {
                "heart_rate": 76,
                "respiratory_rate": 15,
                "oxygen_saturation": 99,
                "temperature": 36.8,
                "systolic_bp": 116,
                "diastolic_bp": 72,
                "pain_score": 1,
                "chief_complaint": "superficial abrasion healed injury recheck only no deformity no functional loss",
            },
            5,
            "trauma_override",
        ),
        (
            {
                "heart_rate": 84,
                "respiratory_rate": 18,
                "oxygen_saturation": 99,
                "temperature": 36.9,
                "systolic_bp": 124,
                "diastolic_bp": 78,
                "pain_score": 2,
                "chief_complaint": "high speed collision with normal vitals",
            },
            2,
            "trauma_override",
        ),
        (
            {
                "heart_rate": 80,
                "respiratory_rate": 16,
                "oxygen_saturation": 98,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 76,
                "pain_score": 8,
                "chief_complaint": "ankle injury after fall with pain and swelling",
            },
            3,
            "trauma_override",
        ),
    ]

    for payload, expected_level, expected_source in cases:
        result = apply_clinical_rules(payload)
        if result is None:
            raise RuntimeError("Trauma validation failed: expected override result, got None.")
        if result.triage_level != expected_level or result.source != expected_source:
            raise RuntimeError(
                f"Trauma validation failed: expected level={expected_level}, "
                f"source={expected_source}, got level={result.triage_level}, source={result.source}."
            )
