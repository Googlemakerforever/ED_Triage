"""Safety-first deterministic guardrails for ED triage.

This module runs before any model or weighted scoring logic. It validates inputs,
normalizes free text, detects high-risk clinical categories, applies hard
overrides, performs uncertainty escalation, and only then falls through to the
existing lower-acuity default logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ed_triage_ai.triage.normalize_complaint import extract_critical_flags


VALID_RANGES = {
    "age": (0, 120),
    "heart_rate": (20, 260),
    "respiratory_rate": (4, 60),
    "oxygen_saturation": (40, 100),
    "temperature": (30.0, 45.0),
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "pain_score": (0, 10),
}

DEFAULT_VALUES = {
    "heart_rate": 0.0,
    "respiratory_rate": 0.0,
    "oxygen_saturation": 100.0,
    "temperature": 36.8,
    "systolic_bp": 0.0,
    "diastolic_bp": 0.0,
    "pain_score": 0.0,
    "age": 0.0,
}

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bdeny\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bnever had\b",
]

TEMPORAL_MODIFIERS = {
    "resolved": ["resolved", "feels better now", "better now", "now resolved"],
    "transient": ["briefly", "transient", "transiently", "for a moment"],
    "sudden": ["suddenly", "sudden", "abruptly"],
    "post_injury": ["after injury", "after crash", "after fall", "after accident"],
}

ABBREVIATION_MAP = {
    r"\bsob\b": "shortness of breath",
    r"\bloc\b": "loss of consciousness",
    r"\bcp\b": "chest pain",
    r"\bams\b": "altered mental status",
    r"\bsi\b": "suicidal ideation",
    r"\bhi\b": "homicidal ideation",
    r"\bgsw\b": "gunshot wound",
}

PHRASE_NORMALIZATIONS = {
    "passed out": "loss of consciousness",
    "pass out": "loss of consciousness",
    "blacked out": "loss of consciousness",
    "black out": "loss of consciousness",
    "fainted": "loss of consciousness",
    "knocked out": "loss of consciousness",
    "hit head": "head injury",
    "struck head": "head injury",
    "head trauma": "head injury",
    "cant breathe": "cannot breathe",
    "can t breathe": "cannot breathe",
    "trouble breathing": "shortness of breath",
    "hard to breathe": "shortness of breath",
    "slurring speech": "slurred speech",
    "facial droop": "face droop",
    "med refill": "medication refill",
    "rx refill": "medication refill",
    "od": "overdose",
}

CRITICAL_FUZZY_PHRASES = [
    "shortness of breath",
    "chest pain",
    "slurred speech",
    "loss of consciousness",
    "unresponsive",
    "unconscious",
    "stroke",
    "weakness",
    "numbness",
    "seizure",
    "open fracture",
    "pulseless",
    "not breathing",
]

CATEGORY_ALIASES = {
    "airway_compromise": [
        "airway compromise",
        "airway obstructed",
        "cannot maintain airway",
        "throat closing",
        "stridor",
    ],
    "respiratory_distress": [
        "severe respiratory distress",
        "respiratory distress",
        "cannot breathe",
        "shortness of breath",
        "difficulty breathing",
        "gasping",
        "agonal breathing",
        "wheezing badly",
    ],
    "shock_bleeding": [
        "massive bleeding",
        "bleeding out",
        "exsanguinating hemorrhage",
        "hemorrhage",
        "hemorrhaging",
        "severe shock",
        "shock",
        "pale clammy",
    ],
    "neurologic_emergency": [
        "unresponsive",
        "unconscious",
        "altered mental status",
        "not responsive",
        "confusion",
        "new confusion",
        "seizure",
        "active seizure",
    ],
    "head_injury_red_flags": [
        "head injury",
        "concussion",
        "loss of consciousness",
        "vomiting",
        "amnesia",
        "blackout",
        "severe headache",
        "confusion",
    ],
    "cardiac_chest_pain": [
        "chest pain",
        "crushing chest pain",
        "pressure in chest",
        "radiating chest pain",
        "chest tightness",
    ],
    "stroke_deficit": [
        "stroke",
        "slurred speech",
        "weakness",
        "numbness",
        "face droop",
        "facial droop",
        "one sided weakness",
        "one sided numbness",
    ],
    "sepsis_toxic": [
        "sepsis",
        "toxic appearing",
        "toxic appearance",
        "rigors",
        "fever and confusion",
        "fever and hypotension",
        "infection",
        "fever",
        "chills",
    ],
    "abdominal_pelvic_emergency": [
        "severe abdominal pain",
        "abdominal pain",
        "rlq pain",
        "pelvic pain",
        "ectopic",
        "peritonitis",
    ],
    "pregnancy_emergency": [
        "pregnant",
        "pregnancy",
        "pregnancy bleeding",
        "pregnant and bleeding",
        "pregnant and pain",
        "postpartum bleeding",
    ],
    "psychiatric_emergency": [
        "suicidal",
        "suicidal ideation",
        "homicidal",
        "homicidal ideation",
        "overdose intentional",
        "wants to kill self",
        "wants to kill someone",
    ],
    "trauma_mechanism": [
        "high speed collision",
        "high-speed collision",
        "rollover",
        "ejection",
        "pedestrian struck",
        "motorcycle crash",
        "major fall",
        "significant fall",
        "fall from height",
        "crush accident",
        "industrial accident",
        "gunshot wound",
        "stab wound",
        "weapon related injury",
    ],
    "fracture_deformity_open_injury": [
        "open fracture",
        "compound fracture",
        "bone exposed",
        "protruding bone",
        "closed fracture",
        "possible fracture",
        "broken bone",
        "deformity",
        "deep laceration",
        "deep cut",
    ],
    "neurovascular_compromise": [
        "absent distal pulse",
        "no distal pulse",
        "numbness distal",
        "weakness distal",
        "cold limb",
        "cyanotic limb",
        "ischemic limb",
        "neurovascular compromise",
    ],
    "burn_severity": ["severe burn", "circumferential burn", "facial burn", "inhalation burn", "electrical burn"],
    "poisoning_overdose": ["overdose", "ingestion", "poisoning", "took too many pills", "toxic ingestion"],
    "minor_injury": [
        "minor sprain",
        "minor laceration",
        "finger injury",
        "toe injury",
        "bruise",
        "contusion",
        "mild burn",
        "simple extremity injury",
    ],
    "minimal_need": ["medication refill", "paperwork", "recheck only", "work note", "healed injury recheck"],
}

TRAUMA_CATEGORY_SET = {
    "trauma_mechanism",
    "fracture_deformity_open_injury",
    "neurovascular_compromise",
    "burn_severity",
}

MEDIUM_RISK_CATEGORIES = {
    "cardiac_chest_pain",
    "respiratory_distress",
    "stroke_deficit",
    "head_injury_red_flags",
    "sepsis_toxic",
    "pregnancy_emergency",
    "psychiatric_emergency",
    "poisoning_overdose",
    "abdominal_pelvic_emergency",
    "trauma_mechanism",
    "fracture_deformity_open_injury",
    "neurovascular_compromise",
}


@dataclass
class RuleResult:
    triage_level: int
    explanation: str
    source: str
    matched_rules: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def reason(self) -> str:
        return self.explanation


@dataclass
class SafetyContext:
    payload: Dict[str, Any]
    normalized_complaint: str
    matched_categories: Dict[str, List[str]]
    negated_terms: List[str]
    temporal_modifiers: List[str]
    abnormal_vitals: List[str]
    severe_abnormal_vitals: List[str]
    critical_flags: Dict[str, bool]
    semantic_matches: Dict[str, List[str]]


def normalize_complaint(text: str) -> str:
    normalized = str(text or "").lower().strip()
    for pattern, replacement in ABBREVIATION_MAP.items():
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for source, replacement in PHRASE_NORMALIZATIONS.items():
        normalized = re.sub(rf"\b{re.escape(source)}\b", replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _safe_float(input_data: Dict[str, Any], key: str) -> float:
    value = input_data.get(key, DEFAULT_VALUES[key])
    if value in (None, ""):
        return float(DEFAULT_VALUES[key])
    return float(value)


def _validate_inputs(input_data: Dict[str, Any]) -> Optional[RuleResult]:
    errors = []
    for field, (lower, upper) in VALID_RANGES.items():
        if field not in input_data or input_data.get(field) in (None, ""):
            continue
        try:
            value = _safe_float(input_data, field)
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric")
            continue
        if value < lower or value > upper:
            errors.append(f"{field} must be between {lower} and {upper}")

    sex = str(input_data.get("sex", "Other") or "Other")
    if sex not in {"Female", "Male", "Other"}:
        errors.append("sex must be Female, Male, or Other")

    if errors:
        complaint = normalize_complaint(str(input_data.get("chief_complaint", "")))
        return RuleResult(
            triage_level=3,
            explanation="Input validation error. Please review the entered values.",
            source="validation_error",
            matched_rules=["VAL_INVALID_INPUT_RANGE"],
            audit={
                "normalized_complaint": complaint,
                "matched_keyword_categories": [],
                "abnormal_vitals_summary": [],
                "validation_errors": errors,
            },
        )
    return None


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _is_negated_at(text: str, start_idx: int) -> bool:
    # Keep the negation window tight so one negation does not incorrectly
    # suppress a later unrelated symptom in a multi-symptom complaint.
    window = text[max(0, start_idx - 16) : start_idx]
    return any(re.search(pattern, window) for pattern in NEGATION_PATTERNS)


def _find_phrase_matches(text: str, aliases: Sequence[str], *, allow_fuzzy: bool = False) -> List[str]:
    matches: List[str] = []
    tokens = text.split()
    for alias in aliases:
        pattern = rf"\b{re.escape(alias)}\b"
        found = False
        for match in re.finditer(pattern, text):
            if not _is_negated_at(text, match.start()):
                matches.append(alias)
                found = True
                break
        if found or not allow_fuzzy:
            continue
        alias_tokens = alias.split()
        token_count = len(alias_tokens)
        if token_count == 0 or token_count > len(tokens):
            continue
        for idx in range(len(tokens) - token_count + 1):
            candidate = " ".join(tokens[idx : idx + token_count])
            if _ratio(candidate, alias) >= (0.88 if token_count > 1 else 0.84):
                prefix = " ".join(tokens[max(0, idx - 3) : idx])
                if any(re.search(pattern, prefix) for pattern in NEGATION_PATTERNS):
                    continue
                matches.append(alias)
                break
    return sorted(set(matches))


def has_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    return bool(_find_phrase_matches(text, keywords))


def detect_high_risk_mechanism(text: str) -> bool:
    return bool(_find_phrase_matches(text, CATEGORY_ALIASES["trauma_mechanism"]))


def detect_neurovascular_compromise(text: str) -> bool:
    return bool(_find_phrase_matches(text, CATEGORY_ALIASES["neurovascular_compromise"]))


def _collect_temporal_modifiers(text: str) -> List[str]:
    found = []
    for label, aliases in TEMPORAL_MODIFIERS.items():
        if _find_phrase_matches(text, aliases):
            found.append(label)
    return found


def _collect_negated_terms(text: str) -> List[str]:
    negated: List[str] = []
    for aliases in CATEGORY_ALIASES.values():
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            for match in re.finditer(pattern, text):
                if _is_negated_at(text, match.start()):
                    negated.append(alias)
                    break
    return sorted(set(negated))


def _summarize_vitals(input_data: Dict[str, Any]) -> tuple[List[str], List[str]]:
    hr = _safe_float(input_data, "heart_rate")
    rr = _safe_float(input_data, "respiratory_rate")
    spo2 = _safe_float(input_data, "oxygen_saturation")
    temp = _safe_float(input_data, "temperature")
    sbp = _safe_float(input_data, "systolic_bp")

    abnormal = []
    severe = []
    if spo2 < 92:
        abnormal.append(f"oxygen_saturation={spo2:.0f}%")
    if spo2 < 85:
        severe.append(f"critical_hypoxia={spo2:.0f}%")
    if sbp < 90:
        abnormal.append(f"systolic_bp={sbp:.0f} mmHg")
    if sbp < 80:
        severe.append(f"critical_hypotension={sbp:.0f} mmHg")
    if rr > 24:
        abnormal.append(f"respiratory_rate={rr:.0f}/min")
    if rr > 30:
        severe.append(f"critical_tachypnea={rr:.0f}/min")
    if hr > 110:
        abnormal.append(f"heart_rate={hr:.0f} bpm")
    if hr > 130:
        severe.append(f"critical_tachycardia={hr:.0f} bpm")
    if temp >= 38.0:
        abnormal.append(f"temperature={temp:.1f} C")
    return abnormal, severe


def _normal_vitals(input_data: Dict[str, Any]) -> bool:
    return (
        60 <= _safe_float(input_data, "heart_rate") <= 100
        and 12 <= _safe_float(input_data, "respiratory_rate") <= 20
        and _safe_float(input_data, "oxygen_saturation") >= 95
        and 100 <= _safe_float(input_data, "systolic_bp") <= 140
        and 36.0 <= _safe_float(input_data, "temperature") < 38.0
    )


def _age_modifier_categories(input_data: Dict[str, Any], matched_categories: Dict[str, List[str]]) -> List[str]:
    age = _safe_float(input_data, "age")
    modifiers = []
    if age < 2 and any(category in matched_categories for category in {"respiratory_distress", "sepsis_toxic", "neurologic_emergency"}):
        modifiers.append("pediatric_high_risk")
    if age >= 75 and any(category in matched_categories for category in MEDIUM_RISK_CATEGORIES):
        modifiers.append("geriatric_high_risk")
    return modifiers


def _detect_categories(input_data: Dict[str, Any], complaint: str) -> Dict[str, List[str]]:
    matched: Dict[str, List[str]] = {}
    for category, aliases in CATEGORY_ALIASES.items():
        allow_fuzzy = any(alias in CRITICAL_FUZZY_PHRASES for alias in aliases)
        hits = _find_phrase_matches(complaint, aliases, allow_fuzzy=allow_fuzzy)
        if hits:
            matched[category] = hits

    # Extra fuzzy pass for critical misspellings that map to an already-known category.
    for alias in CRITICAL_FUZZY_PHRASES:
        if _find_phrase_matches(complaint, [alias], allow_fuzzy=True) and alias not in complaint:
            for category, aliases in CATEGORY_ALIASES.items():
                if alias in aliases:
                    matched.setdefault(category, []).append(alias)

    if detect_high_risk_mechanism(complaint):
        matched.setdefault("trauma_mechanism", []).append("dangerous mechanism")
    if detect_neurovascular_compromise(complaint):
        matched.setdefault("neurovascular_compromise", []).append("neurovascular compromise")
    if _age_modifier_categories(input_data, matched):
        matched["age_modifiers"] = _age_modifier_categories(input_data, matched)

    extras = str(input_data.get("chief_complaint", "")).lower()
    if any(term in extras for term in ["chemo", "chemotherapy", "transplant", "dialysis", "anticoag", "warfarin", "eliquis", "immunocompromised"]):
        matched["high_risk_modifier"] = ["immunocompromised_or_anticoagulated"]
    return {k: sorted(set(v)) for k, v in matched.items()}


def _build_context(input_data: Dict[str, Any]) -> SafetyContext:
    complaint = normalize_complaint(str(input_data.get("chief_complaint", "")))
    matched_categories = _detect_categories(input_data, complaint)
    abnormal_vitals, severe_abnormal_vitals = _summarize_vitals(input_data)
    critical = extract_critical_flags(complaint)
    return SafetyContext(
        payload=input_data,
        normalized_complaint=complaint,
        matched_categories=matched_categories,
        negated_terms=_collect_negated_terms(complaint),
        temporal_modifiers=_collect_temporal_modifiers(complaint),
        abnormal_vitals=abnormal_vitals,
        severe_abnormal_vitals=severe_abnormal_vitals,
        critical_flags={name: bool(meta["flag"]) for name, meta in critical.items()},
        semantic_matches={name: list(meta["matches"]) for name, meta in critical.items() if meta["matches"]},
    )


def _audit_dict(ctx: SafetyContext) -> Dict[str, Any]:
    return {
        "normalized_complaint": ctx.normalized_complaint,
        "matched_keyword_categories": sorted(ctx.matched_categories.keys()),
        "abnormal_vitals_summary": ctx.abnormal_vitals,
        "negated_terms": ctx.negated_terms,
        "temporal_modifiers": ctx.temporal_modifiers,
        "critical_flags": ctx.critical_flags,
        "semantic_matches": ctx.semantic_matches,
    }


def _result(level: int, reason: str, source: str, rules: Iterable[str], ctx: SafetyContext) -> RuleResult:
    return RuleResult(
        triage_level=level,
        explanation=reason,
        source=source,
        matched_rules=list(rules),
        audit=_audit_dict(ctx),
    )


def _has(ctx: SafetyContext, category: str) -> bool:
    return category in ctx.matched_categories


def _has_any(ctx: SafetyContext, categories: Iterable[str]) -> bool:
    return any(category in ctx.matched_categories for category in categories)


def _requires_immediate_intervention(ctx: SafetyContext) -> bool:
    text = ctx.normalized_complaint
    return any(
        phrase in text
        for phrase in [
            "cp r",
            "cardiac arrest",
            "resuscitation",
            "needs airway now",
            "bagging patient",
        ]
    )


def adult_head_injury_risk_hook(_: SafetyContext) -> Optional[str]:
    """Extension point for future adult head injury risk calculators."""
    return None


def pediatric_head_injury_risk_hook(_: SafetyContext) -> Optional[str]:
    """Extension point for future pediatric head injury risk calculators."""
    return None


def c_spine_risk_hook(_: SafetyContext) -> Optional[str]:
    """Extension point for future c-spine risk calculators."""
    return None


def lower_extremity_imaging_rule_hook(_: SafetyContext) -> Optional[str]:
    """Extension point for Ottawa ankle/knee style rules."""
    return None


def _level_1_overrides(ctx: SafetyContext) -> Optional[RuleResult]:
    complaint = ctx.normalized_complaint
    spo2 = _safe_float(ctx.payload, "oxygen_saturation")
    sbp = _safe_float(ctx.payload, "systolic_bp")

    if ctx.critical_flags.get("semantic_airway_compromise"):
        return _result(1, "Level 1 assigned for semantic airway-compromise detection.", "hard_override", ["L1_SEMANTIC_AIRWAY"], ctx)
    if _find_phrase_matches(complaint, ["not breathing", "apnea", "apneic"]):
        return _result(1, "Level 1 assigned for absent or ineffective breathing.", "hard_override", ["L1_APNEA"], ctx)
    if _find_phrase_matches(complaint, ["pulseless", "no pulse", "without pulse"]):
        return _result(1, "Level 1 assigned for pulseless presentation.", "hard_override", ["L1_PULSELESS"], ctx)
    if _find_phrase_matches(complaint, ["unresponsive", "unconscious", "not responsive"]):
        return _result(1, "Level 1 assigned for unresponsive presentation.", "hard_override", ["L1_UNRESPONSIVE"], ctx)
    if _has(ctx, "airway_compromise"):
        return _result(1, "Level 1 assigned for airway compromise.", "hard_override", ["L1_AIRWAY"], ctx)
    if _has(ctx, "respiratory_distress") and spo2 < 85:
        return _result(1, "Level 1 assigned for critical respiratory compromise.", "hard_override", ["L1_CRITICAL_HYPOXIA"], ctx)
    if _has(ctx, "respiratory_distress") and any(flag.startswith("critical_tachypnea") for flag in ctx.severe_abnormal_vitals):
        return _result(1, "Level 1 assigned for severe respiratory distress with instability.", "hard_override", ["L1_RESP_DISTRESS"], ctx)
    if spo2 < 85:
        return _result(1, "Level 1 assigned for oxygen saturation below 85%.", "hard_override", ["L1_HYPOXIA"], ctx)
    if sbp < 80:
        return _result(1, "Level 1 assigned for systolic blood pressure below 80 mmHg.", "hard_override", ["L1_HYPOTENSION"], ctx)
    if _has(ctx, "shock_bleeding"):
        return _result(1, "Level 1 assigned for severe bleeding or shock concern.", "hard_override", ["L1_SHOCK_OR_BLEEDING"], ctx)
    if _find_phrase_matches(complaint, ["active seizure", "seizing now", "still seizing"]):
        return _result(1, "Level 1 assigned for ongoing seizure activity.", "hard_override", ["L1_ACTIVE_SEIZURE"], ctx)
    if _has(ctx, "neurologic_emergency") and _find_phrase_matches(complaint, ["profoundly confused", "obtunded", "severely altered"]):
        return _result(1, "Level 1 assigned for critical altered mental status.", "hard_override", ["L1_CRITICAL_AMS"], ctx)
    if ctx.critical_flags.get("semantic_severe_trauma") and ctx.severe_abnormal_vitals:
        return _result(1, "Level 1 assigned for semantic severe trauma with instability.", "hard_override", ["L1_SEMANTIC_TRAUMA_INSTABILITY"], ctx)
    if _has_any(ctx, ["trauma_mechanism", "fracture_deformity_open_injury"]) and ctx.severe_abnormal_vitals:
        return _result(1, "Level 1 assigned for major trauma with profound instability.", "hard_override", ["L1_MAJOR_TRAUMA_UNSTABLE"], ctx)
    if _requires_immediate_intervention(ctx):
        return _result(1, "Level 1 assigned for immediate life-saving intervention need.", "hard_override", ["L1_IMMEDIATE_INTERVENTION"], ctx)
    return None


def _level_2_overrides(ctx: SafetyContext) -> Optional[RuleResult]:
    complaint = ctx.normalized_complaint
    spo2 = _safe_float(ctx.payload, "oxygen_saturation")
    sbp = _safe_float(ctx.payload, "systolic_bp")
    hr = _safe_float(ctx.payload, "heart_rate")
    rr = _safe_float(ctx.payload, "respiratory_rate")

    if ctx.critical_flags.get("semantic_head_injury_red_flags"):
        return _result(2, "Level 2 assigned for semantic head-injury red flags.", "hard_override", ["L2_SEMANTIC_HEAD_INJURY"], ctx)
    if ctx.critical_flags.get("semantic_loc"):
        return _result(2, "Level 2 assigned for semantic loss-of-consciousness detection.", "hard_override", ["L2_SEMANTIC_LOC"], ctx)
    if ctx.critical_flags.get("semantic_stroke"):
        return _result(2, "Level 2 assigned for semantic stroke-like complaint.", "hard_override", ["L2_SEMANTIC_STROKE"], ctx)
    if _has(ctx, "cardiac_chest_pain"):
        return _result(2, "Level 2 assigned for potentially dangerous chest pain.", "hard_override", ["L2_CHEST_PAIN"], ctx)
    if _has(ctx, "respiratory_distress"):
        return _result(2, "Level 2 assigned for shortness of breath or respiratory distress.", "hard_override", ["L2_RESPIRATORY"], ctx)
    if _has(ctx, "stroke_deficit"):
        return _result(2, "Level 2 assigned for stroke-like or focal neurologic symptoms.", "hard_override", ["L2_STROKE_SIGNS"], ctx)
    head_injury_context = _find_phrase_matches(complaint, ["head injury", "concussion", "hit head"], allow_fuzzy=True)
    if head_injury_context and _find_phrase_matches(complaint, ["loss of consciousness", "vomiting", "confusion", "amnesia", "blackout", "severe headache"], allow_fuzzy=True):
        return _result(2, "Level 2 assigned for head injury red flags.", "hard_override", ["L2_HEAD_INJURY_RED_FLAGS"], ctx)
    if _has(ctx, "psychiatric_emergency"):
        return _result(2, "Level 2 assigned for psychiatric safety emergency.", "hard_override", ["L2_PSYCH_SAFETY"], ctx)
    if _has(ctx, "pregnancy_emergency") and _has_any(ctx, ["abdominal_pelvic_emergency", "shock_bleeding"]):
        return _result(2, "Level 2 assigned for high-risk pregnancy-related complaint.", "hard_override", ["L2_PREGNANCY_EMERGENCY"], ctx)
    if _has(ctx, "sepsis_toxic") and (_safe_float(ctx.payload, "temperature") >= 39 or sbp < 90 or hr > 110):
        return _result(2, "Level 2 assigned for sepsis concern with concerning physiology.", "hard_override", ["L2_SEPSIS_CONCERN"], ctx)
    if _has(ctx, "poisoning_overdose"):
        return _result(2, "Level 2 assigned for poisoning or overdose concern.", "hard_override", ["L2_OVERDOSE"], ctx)
    if _has(ctx, "neurovascular_compromise"):
        return _result(2, "Level 2 assigned for neurovascular compromise.", "hard_override", ["L2_NEUROVASCULAR"], ctx)
    if _find_phrase_matches(complaint, ["open fracture", "compound fracture", "bone exposed", "protruding bone"]):
        return _result(2, "Level 2 assigned for open fracture concern.", "hard_override", ["L2_OPEN_FRACTURE"], ctx)
    if _has(ctx, "trauma_mechanism"):
        return _result(2, "Level 2 assigned for dangerous injury mechanism.", "hard_override", ["L2_DANGEROUS_MECHANISM"], ctx)
    if _has(ctx, "burn_severity"):
        return _result(2, "Level 2 assigned for severe burn concern.", "hard_override", ["L2_SEVERE_BURN"], ctx)
    if spo2 <= 91:
        return _result(2, "Level 2 assigned for oxygen saturation 91% or lower.", "hard_override", ["L2_HYPOXIA_RANGE"], ctx)
    if 80 <= sbp <= 89:
        return _result(2, "Level 2 assigned for systolic blood pressure 80-89 mmHg.", "hard_override", ["L2_BORDERLINE_HYPOTENSION"], ctx)
    if hr > 130:
        return _result(2, "Level 2 assigned for heart rate above 130 bpm.", "hard_override", ["L2_SEVERE_TACHYCARDIA"], ctx)
    if rr > 30:
        return _result(2, "Level 2 assigned for respiratory rate above 30.", "hard_override", ["L2_SEVERE_TACHYPNEA"], ctx)
    if _safe_float(ctx.payload, "pain_score") >= 7 and _has_any(ctx, ["cardiac_chest_pain", "respiratory_distress", "trauma_mechanism", "abdominal_pelvic_emergency"]):
        return _result(2, "Level 2 assigned for severe pain with dangerous clinical context.", "hard_override", ["L2_PAIN_WITH_RED_FLAGS"], ctx)
    if _has(ctx, "high_risk_modifier") and _has_any(ctx, ["head_injury_red_flags", "cardiac_chest_pain", "stroke_deficit", "sepsis_toxic"]):
        return _result(2, "Level 2 assigned for high-risk modifier with concerning presentation.", "hard_override", ["L2_HIGH_RISK_MODIFIER"], ctx)
    if _has(ctx, "age_modifiers") and _has_any(ctx, ["cardiac_chest_pain", "respiratory_distress", "sepsis_toxic", "neurologic_emergency"]):
        return _result(2, "Level 2 assigned for high-risk age modifier with concerning presentation.", "hard_override", ["L2_AGE_MODIFIER"], ctx)
    return None


def _uncertainty_escalation(ctx: SafetyContext) -> Optional[RuleResult]:
    complaint = ctx.normalized_complaint
    matched_medium_risk = sorted(category for category in ctx.matched_categories if category in MEDIUM_RISK_CATEGORIES)
    concerning_parse = bool(_find_phrase_matches(complaint, ["felt better now", "briefly", "resolved"], allow_fuzzy=False))

    if ctx.critical_flags.get("semantic_loc"):
        return _result(2, "Level 2 assigned because semantic loss-of-consciousness wording is never safely low risk.", "uncertainty_escalation", ["UNC_SEMANTIC_LOC"], ctx)
    if ctx.critical_flags.get("semantic_stroke"):
        return _result(2, "Level 2 assigned because semantic transient neurologic symptoms remain high risk.", "uncertainty_escalation", ["UNC_SEMANTIC_STROKE"], ctx)
    if _find_phrase_matches(complaint, ["loss of consciousness", "blackout", "fainted"], allow_fuzzy=True) and concerning_parse:
        return _result(2, "Level 2 assigned because a transient severe symptom still carries serious risk.", "uncertainty_escalation", ["UNC_TRANSIENT_HIGH_RISK"], ctx)
    if len(matched_medium_risk) >= 2:
        return _result(2, "Level 2 assigned because multiple moderate-risk signals coexist.", "uncertainty_escalation", ["UNC_MULTI_MEDIUM_RISK"], ctx)
    if _has_any(ctx, ["cardiac_chest_pain", "respiratory_distress", "stroke_deficit", "poisoning_overdose"]) and not ctx.abnormal_vitals:
        return _result(2, "Level 2 assigned because high-risk symptoms can deteriorate despite normal vitals.", "uncertainty_escalation", ["UNC_HIGH_RISK_NORMAL_VITALS"], ctx)
    if _has_any(ctx, ["head_injury_red_flags", "trauma_mechanism", "fracture_deformity_open_injury"]) and "resolved" in ctx.temporal_modifiers:
        return _result(2, "Level 2 assigned because improvement does not erase recent trauma red flags.", "uncertainty_escalation", ["UNC_RECENT_RED_FLAG_PERSISTS"], ctx)
    if "?" in str(ctx.payload.get("chief_complaint", "")) and matched_medium_risk:
        return _result(3, "Level 3 assigned because the complaint is incomplete but contains concerning features.", "uncertainty_escalation", ["UNC_INCOMPLETE_PARSE"], ctx)
    return None


def _default_logic(ctx: SafetyContext) -> Optional[RuleResult]:
    complaint = ctx.normalized_complaint
    pain = _safe_float(ctx.payload, "pain_score")

    if _find_phrase_matches(complaint, ["closed fracture", "possible fracture", "broken bone", "deformity", "unable to bear weight"], allow_fuzzy=True):
        return _result(3, "Level 3 assigned for clinically significant extremity injury with stable features.", "default_logic", ["L3_FRACTURE_OR_DEFORMITY"], ctx)
    if _find_phrase_matches(complaint, ["severe abdominal pain", "abdominal pain", "deep laceration", "persistent vomiting"], allow_fuzzy=True):
        return _result(3, "Level 3 assigned for urgent complaint likely needing multiple resources.", "default_logic", ["L3_MULTI_RESOURCE_COMPLAINT"], ctx)
    if _has(ctx, "respiratory_distress") or _has(ctx, "sepsis_toxic"):
        return _result(3, "Level 3 assigned for moderate respiratory or infectious concern without instability.", "default_logic", ["L3_MODERATE_SYSTEMIC"], ctx)
    if pain >= 6 and _has_any(ctx, ["minor_injury", "fracture_deformity_open_injury", "abdominal_pelvic_emergency"]):
        return _result(3, "Level 3 assigned for significant pain in a clinically important presentation.", "default_logic", ["L3_SIGNIFICANT_PAIN"], ctx)

    if _has(ctx, "minor_injury") and _normal_vitals(ctx.payload):
        return _result(4, "Level 4 assigned for minor stable injury likely needing one simple resource.", "default_logic", ["L4_MINOR_INJURY"], ctx)
    if _find_phrase_matches(complaint, ["mild infection", "simple laceration", "mild stable symptoms"], allow_fuzzy=True) and _normal_vitals(ctx.payload):
        return _result(4, "Level 4 assigned for lower-risk problem without red flags.", "default_logic", ["L4_SIMPLE_RESOURCE"], ctx)

    non_low_risk_categories = set(ctx.matched_categories) - {"minimal_need"}
    if _has(ctx, "minimal_need") and _normal_vitals(ctx.payload) and pain <= 1 and not non_low_risk_categories:
        return _result(5, "Level 5 assigned for trivial, low-resource presentation.", "default_logic", ["L5_MINIMAL_NEED"], ctx)
    if _find_phrase_matches(complaint, ["superficial abrasion", "tiny bruise", "very minor bump"], allow_fuzzy=True) and _normal_vitals(ctx.payload) and pain <= 1:
        return _result(5, "Level 5 assigned for very minor stable presentation.", "default_logic", ["L5_TRIVIAL_INJURY"], ctx)
    return None


def evaluate_safety_guardrails(input_data: Dict[str, Any]) -> Optional[RuleResult]:
    validation_error = _validate_inputs(input_data)
    if validation_error is not None:
        return validation_error

    ctx = _build_context(input_data)

    level_1 = _level_1_overrides(ctx)
    if level_1 is not None:
        return level_1

    level_2 = _level_2_overrides(ctx)
    if level_2 is not None:
        return level_2

    uncertainty = _uncertainty_escalation(ctx)
    if uncertainty is not None:
        return uncertainty

    return _default_logic(ctx)


def apply_clinical_rules(input_data: Dict[str, Any]) -> Optional[RuleResult]:
    return evaluate_safety_guardrails(input_data)


def apply_trauma_overrides(input_data: Dict[str, Any]) -> Optional[RuleResult]:
    result = evaluate_safety_guardrails(input_data)
    if result is None:
        return None
    categories = set(result.audit.get("matched_keyword_categories", []))
    if result.source in {"hard_override", "uncertainty_escalation", "default_logic"} and categories.intersection(TRAUMA_CATEGORY_SET | {"minor_injury"}):
        return result
    return None


def validate_required_test_case() -> None:
    payload = {
        "oxygen_saturation": 88,
        "systolic_bp": 85,
        "heart_rate": 120,
        "respiratory_rate": 24,
        "temperature": 37.2,
        "pain_score": 6,
        "chief_complaint": "chest pain",
    }
    result = evaluate_safety_guardrails(payload)
    if result is None or result.triage_level != 2:
        raise RuntimeError("Safety validation failed: high-risk chest pain with unstable vitals must not be under-triaged.")


def validate_trauma_override_cases() -> None:
    cases = [
        ({"heart_rate": 122, "respiratory_rate": 18, "oxygen_saturation": 82, "temperature": 36.8, "systolic_bp": 118, "diastolic_bp": 72, "pain_score": 4, "chief_complaint": "motorcycle crash with severe respiratory distress"}, 1),
        ({"heart_rate": 96, "respiratory_rate": 18, "oxygen_saturation": 98, "temperature": 36.9, "systolic_bp": 124, "diastolic_bp": 78, "pain_score": 5, "chief_complaint": "open fracture with bone exposed and cold limb after crash"}, 2),
        ({"heart_rate": 88, "respiratory_rate": 16, "oxygen_saturation": 98, "temperature": 36.8, "systolic_bp": 122, "diastolic_bp": 76, "pain_score": 7, "chief_complaint": "possible broken bone with deformity and unable to bear weight"}, 3),
        ({"heart_rate": 82, "respiratory_rate": 16, "oxygen_saturation": 98, "temperature": 36.8, "systolic_bp": 118, "diastolic_bp": 74, "pain_score": 3, "chief_complaint": "minor sprain ambulatory"}, 4),
    ]
    for payload, expected in cases:
        result = apply_trauma_overrides(payload)
        if result is None or result.triage_level != expected:
            raise RuntimeError(f"Trauma validation failed for expected level {expected}.")
