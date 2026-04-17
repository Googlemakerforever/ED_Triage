"""ML prediction for gray-zone triage cases after safety screening."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ed_triage_ai.data.preprocess import enrich_features


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


def predict_acuity_ml(model: Any, patient: Dict[str, Any], normalized_complaint: str) -> Tuple[int, Dict[str, float]]:
    row = pd.DataFrame([{**patient, "chief_complaint": normalized_complaint}])
    enriched = _align_features_for_model(model, enrich_features(row))
    pred_level = int(model.predict(enriched)[0])
    probabilities = model.predict_proba(enriched)[0]
    classes = model.named_steps["clf"].classes_
    prob_map = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    return pred_level, calibrate_prediction(prob_map)
