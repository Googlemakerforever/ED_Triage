"""Complaint normalization, negation, and temporal parsing."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence, Tuple

from ed_triage_ai.triage.keyword_sets import (
    ABBREVIATIONS,
    NEGATION_CUES,
    PHRASE_NORMALIZATIONS,
    SEMANTIC_PHRASE_FAMILIES,
    TEMPORAL_MAP,
)


def normalize_complaint(text: str) -> str:
    normalized = str(text or "").lower().strip()
    for pattern, replacement in ABBREVIATIONS.items():
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for raw, replacement in PHRASE_NORMALIZATIONS.items():
        normalized = re.sub(rf"\b{re.escape(raw)}\b", replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _negation_window(text: str, start_idx: int) -> str:
    return text[max(0, start_idx - 18) : start_idx]


def is_negated(text: str, start_idx: int) -> bool:
    window = _negation_window(text, start_idx)
    return any(re.search(rf"\b{re.escape(cue)}\b", window) for cue in NEGATION_CUES)


def detect_negations(text: str, aliases: Iterable[str]) -> List[str]:
    negated: List[str] = []
    for alias in aliases:
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text):
            if is_negated(text, match.start()):
                negated.append(alias)
                break
    return sorted(set(negated))


def detect_temporal_modifiers(text: str) -> List[str]:
    found: List[str] = []
    for label, aliases in TEMPORAL_MAP.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                found.append(label)
                break
    return found


def fuzzy_phrase_match(text: str, alias: str) -> bool:
    alias_tokens = alias.split()
    tokens = text.split()
    if not alias_tokens or len(tokens) < len(alias_tokens):
        return False
    for idx in range(len(tokens) - len(alias_tokens) + 1):
        candidate = " ".join(tokens[idx : idx + len(alias_tokens)])
        if SequenceMatcher(None, candidate, alias).ratio() >= (0.88 if len(alias_tokens) > 1 else 0.84):
            if not is_negated(text, text.find(candidate)):
                return True
    return False


def find_phrase_matches(text: str, aliases: Sequence[str], *, allow_fuzzy: bool = False) -> List[str]:
    matches: List[str] = []
    for alias in aliases:
        found = False
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text):
            if not is_negated(text, match.start()):
                matches.append(alias)
                found = True
                break
        if not found and allow_fuzzy and fuzzy_phrase_match(text, alias):
            matches.append(alias)
    return sorted(set(matches))


def flatten_aliases(alias_map: Dict[str, Sequence[str]]) -> List[str]:
    flattened: List[str] = []
    for aliases in alias_map.values():
        flattened.extend(list(aliases))
    return flattened


def _semantic_hits(text: str, family: str, *, allow_fuzzy: bool = True) -> List[str]:
    return find_phrase_matches(text, SEMANTIC_PHRASE_FAMILIES[family], allow_fuzzy=allow_fuzzy)


def detect_semantic_loc(text: str) -> Tuple[bool, List[str]]:
    core_hits = _semantic_hits(text, "loc_core")
    support_hits = _semantic_hits(text, "loc_support")
    if core_hits:
        return True, sorted(set(core_hits + support_hits))
    # Secondary deterministic net: collapse wording plus duration/recovery wording.
    collapse_hits = find_phrase_matches(text, ["collapsed", "collapse", "went down"], allow_fuzzy=True)
    if collapse_hits and support_hits:
        return True, sorted(set(collapse_hits + support_hits))
    return False, []


def detect_semantic_stroke(text: str) -> Tuple[bool, List[str]]:
    core_hits = _semantic_hits(text, "stroke_core")
    support_hits = _semantic_hits(text, "stroke_support")
    if core_hits:
        return True, sorted(set(core_hits + support_hits))
    if len(support_hits) >= 2:
        return True, sorted(set(support_hits))
    if support_hits and find_phrase_matches(text, ["speech difficulty", "slurred speech"], allow_fuzzy=True):
        return True, sorted(set(support_hits + ["speech difficulty"]))
    return False, []


def detect_semantic_head_injury_red_flags(text: str) -> Tuple[bool, List[str]]:
    context_hits = _semantic_hits(text, "head_injury_context")
    support_hits = _semantic_hits(text, "head_red_flag_support")
    loc_flag, loc_hits = detect_semantic_loc(text)
    if context_hits and (loc_flag or support_hits):
        return True, sorted(set(context_hits + support_hits + loc_hits))
    return False, []


def detect_semantic_airway_compromise(text: str) -> Tuple[bool, List[str]]:
    hits = _semantic_hits(text, "airway_core")
    return bool(hits), hits


def detect_semantic_severe_trauma(text: str) -> Tuple[bool, List[str]]:
    hits = _semantic_hits(text, "severe_trauma_core")
    return bool(hits), hits


def extract_critical_flags(text: str) -> Dict[str, Dict[str, object]]:
    loc_flag, loc_hits = detect_semantic_loc(text)
    stroke_flag, stroke_hits = detect_semantic_stroke(text)
    head_flag, head_hits = detect_semantic_head_injury_red_flags(text)
    airway_flag, airway_hits = detect_semantic_airway_compromise(text)
    trauma_flag, trauma_hits = detect_semantic_severe_trauma(text)
    return {
        "semantic_loc": {"flag": loc_flag, "matches": loc_hits},
        "semantic_stroke": {"flag": stroke_flag, "matches": stroke_hits},
        "semantic_head_injury_red_flags": {"flag": head_flag, "matches": head_hits},
        "semantic_airway_compromise": {"flag": airway_flag, "matches": airway_hits},
        "semantic_severe_trauma": {"flag": trauma_flag, "matches": trauma_hits},
    }
