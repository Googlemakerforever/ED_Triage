"""Derived feature computation for the hybrid engine."""

from __future__ import annotations

from typing import Any, Dict

from ed_triage_ai.triage.keyword_sets import THRESHOLDS


def compute_derived_vitals(payload: Dict[str, Any], extracted_features: Dict[str, Any]) -> Dict[str, Any]:
    hr = float(payload.get("heart_rate", 0) or 0)
    sbp = float(payload.get("systolic_bp", 0) or 0)
    dbp = float(payload.get("diastolic_bp", 0) or 0)
    rr = float(payload.get("respiratory_rate", 0) or 0)
    spo2 = float(payload.get("oxygen_saturation", 100) or 100)
    temp = float(payload.get("temperature", 36.8) or 36.8)
    age = float(payload.get("age", 0) or 0)

    shock_index = hr / sbp if sbp else None
    red_flag_count = sum(1 for value in extracted_features.get("features", {}).values() if value)

    return {
        "shock_index": round(shock_index, 3) if shock_index is not None else None,
        "pulse_pressure": round(sbp - dbp, 1),
        "fever_flag": temp >= THRESHOLDS["fever"],
        "hypoxia_bucket": "critical" if spo2 < THRESHOLDS["critical_spo2"] else "low" if spo2 <= THRESHOLDS["low_spo2"] else "normal",
        "tachycardia_flag": hr > 110,
        "tachypnea_flag": rr > 24,
        "hypotension_flag": sbp < 90,
        "age_bucket": "pediatric" if age < 18 else "geriatric" if age >= 75 else "adult",
        "multi_red_flag_count": red_flag_count,
        "missingness_flags": {
            key: payload.get(key) in (None, "")
            for key in ("heart_rate", "respiratory_rate", "oxygen_saturation", "temperature", "systolic_bp", "diastolic_bp", "pain_score")
        },
    }
