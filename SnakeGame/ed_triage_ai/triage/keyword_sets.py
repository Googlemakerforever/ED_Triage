"""Centralized thresholds and phrase sets for the hybrid triage engine."""

from __future__ import annotations

REQUIRED_FIELDS = [
    "age",
    "sex",
    "heart_rate",
    "respiratory_rate",
    "oxygen_saturation",
    "temperature",
    "systolic_bp",
    "diastolic_bp",
    "pain_score",
    "chief_complaint",
]

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

THRESHOLDS = {
    "critical_spo2": 85,
    "low_spo2": 91,
    "critical_sbp": 80,
    "low_sbp": 89,
    "critical_hr": 130,
    "critical_rr": 30,
    "fever": 38.0,
    "high_fever": 39.0,
}

NEGATION_CUES = ["no", "not", "denies", "deny", "without", "negative for"]
TEMPORAL_MAP = {
    "sudden": ["suddenly", "sudden", "abruptly"],
    "transient": ["briefly", "brief", "transient", "for a moment"],
    "resolved": ["now resolved", "resolved", "feels better now", "better now"],
    "post_injury": ["after injury", "after crash", "after fall", "after accident"],
    "earlier_today": ["earlier today", "this morning", "this afternoon", "tonight"],
}

ABBREVIATIONS = {
    r"\bsob\b": "shortness of breath",
    r"\bloc\b": "loss of consciousness",
    r"\bcp\b": "chest pain",
    r"\bsi\b": "suicidal ideation",
    r"\bhi\b": "homicidal ideation",
    r"\bams\b": "altered mental status",
    r"\bgsw\b": "gunshot wound",
}

PHRASE_NORMALIZATIONS = {
    "passed out": "loss of consciousness",
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
    "med refill": "medication refill",
    "rx refill": "medication refill",
}

FEATURE_ALIASES = {
    "chest_pain": ["chest pain", "pressure in chest", "crushing chest pain", "radiating chest pain"],
    "shortness_of_breath": ["shortness of breath", "cannot breathe", "difficulty breathing", "respiratory distress"],
    "stroke_like_symptoms": ["stroke", "slurred speech", "weakness", "numbness", "face droop", "facial droop"],
    "head_injury": ["head injury", "concussion"],
    "loss_of_consciousness": ["loss of consciousness", "blackout", "syncope"],
    "vomiting": ["vomiting", "vomit", "emesis"],
    "confusion": ["confusion", "confused"],
    "amnesia": ["amnesia", "memory loss"],
    "possible_fracture": ["possible fracture", "closed fracture", "broken bone", "suspected fracture"],
    "open_injury": ["open fracture", "compound fracture", "bone exposed", "protruding bone", "open wound"],
    "deformity": ["deformity", "angulated", "crooked"],
    "neurovascular_compromise": ["absent distal pulse", "no distal pulse", "cold limb", "cyanotic limb", "ischemic limb", "numbness distal", "weakness distal"],
    "dangerous_mechanism": ["high speed collision", "high-speed collision", "rollover", "ejection", "pedestrian struck", "motorcycle crash", "fall from height", "fell down stairs", "down stairs", "stair fall", "industrial accident", "gunshot wound", "stab wound"],
    "severe_bleeding": ["massive bleeding", "bleeding out", "exsanguinating hemorrhage", "heavy bleeding"],
    "severe_abdominal_pain": ["severe abdominal pain", "abdominal pain", "pelvic pain"],
    "pregnancy_related": ["pregnant", "pregnancy", "postpartum", "vaginal bleeding pregnant"],
    "suicidal_risk": ["suicidal", "suicidal ideation", "wants to kill self"],
    "homicidal_risk": ["homicidal", "homicidal ideation", "wants to kill someone"],
    "overdose_or_ingestion": ["overdose", "ingestion", "poisoning", "took too many pills"],
    "sepsis_concern": ["sepsis", "fever and confusion", "toxic appearance", "rigors", "infection"],
    "altered_mental_status": ["altered mental status", "unresponsive", "unconscious", "not responsive", "obtunded"],
    "airway_compromise": ["airway compromise", "throat closing", "stridor"],
    "active_seizure": ["active seizure", "seizing now", "still seizing"],
    "severe_allergic_reaction": ["anaphylaxis", "severe allergic reaction", "allergic reaction throat closing"],
    "minor_injury": ["minor sprain", "minor laceration", "finger injury", "toe injury", "bruise", "contusion", "superficial abrasion"],
    "low_resource_case": ["medication refill", "paperwork", "recheck only", "work note"],
}

FUZZY_CRITICAL_FEATURES = {
    "chest_pain",
    "shortness_of_breath",
    "stroke_like_symptoms",
    "loss_of_consciousness",
    "open_injury",
    "neurovascular_compromise",
    "altered_mental_status",
}
