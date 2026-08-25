"""ML prediction for gray-zone triage cases after safety screening."""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from ed_triage_ai.data.preprocess import enrich_features
from ed_triage_ai.utils.config import ARTIFACT_DIR


BINARY_MODEL_PATH = ARTIFACT_DIR / "nhamcs_binary_xgb.joblib"
BINARY_METADATA_PATH = ARTIFACT_DIR / "nhamcs_binary_xgb_metadata.json"
_LOW_RESOURCE_COMPLAINT_MARKERS = (
    "medication refill",
    "prescription refill",
    "rx refill",
    "paperwork",
    "work note",
    "recheck only",
)


@lru_cache(maxsize=1)
def get_binary_model_bundle() -> Tuple[Any, Dict[str, Any], list[str], float | None]:
    try:
        metadata = json.loads(BINARY_METADATA_PATH.read_text(encoding="utf-8"))
        model = joblib.load(BINARY_MODEL_PATH)
        feature_list = list(metadata["feature_list"])
        threshold = float(metadata["threshold_used"])
        print(f"Loaded NHAMCS binary model from {BINARY_MODEL_PATH}")
        return model, metadata, feature_list, threshold
    except Exception as exc:
        print(f"Warning: NHAMCS binary model unavailable, using fallback mode: {exc}")
        return None, {}, [], None


def _align_features_for_model(model: Any, enriched: pd.DataFrame) -> pd.DataFrame:
    out = enriched.copy()
    preprocessor = model.named_steps["preprocessor"]
    for _, _, cols in preprocessor.transformers_:
        if cols == "drop" or cols == "passthrough":
            continue
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            if col not in out.columns:
                if col in {"fever_flag", "hypoxia_flag", "hypotension_flag"} or str(col).startswith("kw_"):
                    out[col] = 0.0
                elif col == "chief_complaint":
                    out[col] = ""
                elif col == "sex":
                    out[col] = "Other"
                else:
                    out[col] = np.nan
    return out


def calibrate_prediction(probabilities: Dict[str, float]) -> Dict[str, float]:
    total = sum(probabilities.values()) or 1.0
    return {label: round(value / total, 4) for label, value in probabilities.items()}


def _is_missing(value: Any) -> bool:
    return value in (None, "") or pd.isna(value)


def _to_float_or_nan(value: Any) -> float:
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_runtime_features_for_binary_model(patient: Dict[str, Any], normalized_complaint: str) -> Dict[str, Any]:
    heart_rate = _to_float_or_nan(patient.get("heart_rate"))
    respiratory_rate = _to_float_or_nan(patient.get("respiratory_rate"))
    oxygen_saturation = _to_float_or_nan(patient.get("oxygen_saturation"))
    temperature = _to_float_or_nan(patient.get("temperature"))
    systolic_bp = _to_float_or_nan(patient.get("systolic_bp"))
    diastolic_bp = _to_float_or_nan(patient.get("diastolic_bp"))
    pain_score = _to_float_or_nan(patient.get("pain_score"))
    age = _to_float_or_nan(patient.get("age"))

    shock_index = heart_rate / systolic_bp if not np.isnan(heart_rate) and not np.isnan(systolic_bp) and systolic_bp != 0 else float("nan")
    pulse_pressure = systolic_bp - diastolic_bp if not np.isnan(systolic_bp) and not np.isnan(diastolic_bp) else float("nan")
    fever_flag = int(not np.isnan(temperature) and temperature >= 38.0)
    hypoxia_flag = int(not np.isnan(oxygen_saturation) and oxygen_saturation < 92.0)
    tachycardia_flag = int(not np.isnan(heart_rate) and heart_rate > 100.0)
    tachypnea_flag = int(not np.isnan(respiratory_rate) and respiratory_rate > 20.0)
    hypotension_flag = int(not np.isnan(systolic_bp) and systolic_bp < 90.0)
    extreme_hr_flag = int((not np.isnan(heart_rate) and heart_rate < 40.0) or (not np.isnan(heart_rate) and heart_rate > 130.0))
    extreme_bp_flag = int(
        (not np.isnan(systolic_bp) and systolic_bp < 80.0)
        or (not np.isnan(systolic_bp) and systolic_bp > 200.0)
        or (not np.isnan(diastolic_bp) and diastolic_bp > 120.0)
    )
    abnormal_vitals_count = int(
        fever_flag
        + hypoxia_flag
        + tachycardia_flag
        + tachypnea_flag
        + hypotension_flag
        + extreme_hr_flag
        + extreme_bp_flag
    )
    severe_vitals_flag = int(
        extreme_hr_flag
        or extreme_bp_flag
        or (not np.isnan(temperature) and temperature >= 39.0)
        or (not np.isnan(respiratory_rate) and respiratory_rate >= 30.0)
    )
    age_risk_flag = int(not np.isnan(age) and age > 65.0)

    return {
        "age": age,
        "sex": patient.get("sex", "unknown") or "unknown",
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "oxygen_saturation": oxygen_saturation,
        "temperature": temperature,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "pain_score": pain_score,
        "chief_complaint": normalized_complaint,
        "source_year": float("nan"),
        "heart_rate_missing": int(_is_missing(patient.get("heart_rate"))),
        "respiratory_rate_missing": int(_is_missing(patient.get("respiratory_rate"))),
        "oxygen_saturation_missing": int(_is_missing(patient.get("oxygen_saturation"))),
        "temperature_missing": int(_is_missing(patient.get("temperature"))),
        "systolic_bp_missing": int(_is_missing(patient.get("systolic_bp"))),
        "diastolic_bp_missing": int(_is_missing(patient.get("diastolic_bp"))),
        "pain_score_missing": int(_is_missing(patient.get("pain_score"))),
        "shock_index": shock_index,
        "pulse_pressure": pulse_pressure,
        "fever_flag": fever_flag,
        "hypoxia_flag": hypoxia_flag,
        "tachycardia_flag": tachycardia_flag,
        "tachypnea_flag": tachypnea_flag,
        "hypotension_flag": hypotension_flag,
        "abnormal_vitals_count": abnormal_vitals_count,
        "severe_vitals_flag": severe_vitals_flag,
        "age_risk_flag": age_risk_flag,
        "extreme_hr_flag": extreme_hr_flag,
        "extreme_bp_flag": extreme_bp_flag,
    }


def score_high_acuity_binary(features: Dict[str, Any]) -> Dict[str, Any]:
    model, _metadata, feature_list, threshold = get_binary_model_bundle()
    if model is None or threshold is None or not feature_list:
        print("Using NHAMCS binary fallback result.")
        return {
            "high_acuity_score": 0.0,
            "high_acuity_pred": 0,
            "threshold": None,
            "model_available": False,
        }

    input_frame = pd.DataFrame([features])
    input_frame = input_frame.reindex(columns=feature_list)

    prob = float(model.predict_proba(input_frame)[0, 1])
    complaint = str(features.get("chief_complaint", "") or "").lower()
    is_low_resource_case = any(marker in complaint for marker in _LOW_RESOURCE_COMPLAINT_MARKERS)
    vitals_stable = (
        int(features.get("fever_flag", 0) or 0) == 0
        and int(features.get("hypoxia_flag", 0) or 0) == 0
        and int(features.get("hypotension_flag", 0) or 0) == 0
        and int(features.get("tachycardia_flag", 0) or 0) == 0
        and int(features.get("tachypnea_flag", 0) or 0) == 0
        and int(features.get("severe_vitals_flag", 0) or 0) == 0
        and float(features.get("pain_score", 0.0) or 0.0) <= 1.0
    )
    if is_low_resource_case and vitals_stable:
        prob = min(prob, 0.05)
    pred = int(prob >= threshold)

    return {
        "high_acuity_score": prob,
        "high_acuity_pred": pred,
        "threshold": threshold,
        "model_available": True,
    }


def predict_acuity_ml(model: Any, patient: Dict[str, Any], normalized_complaint: str) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
    row = pd.DataFrame([{**patient, "chief_complaint": normalized_complaint}])
    enriched = _align_features_for_model(model, enrich_features(row))
    pred_level = int(model.predict(enriched)[0])
    probabilities = model.predict_proba(enriched)[0]
    classes = model.named_steps["clf"].classes_
    prob_map = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    runtime_features = build_runtime_features_for_binary_model(patient, normalized_complaint)
    binary_result = score_high_acuity_binary(runtime_features)
    return pred_level, calibrate_prediction(prob_map), binary_result
