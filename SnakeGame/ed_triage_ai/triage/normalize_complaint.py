"""Complaint normalization, negation, and temporal parsing."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence

from ed_triage_ai.triage.keyword_sets import ABBREVIATIONS, NEGATION_CUES, PHRASE_NORMALIZATIONS, TEMPORAL_MAP


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
