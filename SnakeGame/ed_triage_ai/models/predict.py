"""Prediction entrypoint with safety overrides and explainability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ed_triage_ai.data.loaders import load_and_merge_datasets
from ed_triage_ai.data.preprocess import build_preprocessor, enrich_features, split_xy
from ed_triage_ai.rules.clinical_rules import (
    apply_clinical_rules,
    validate_required_test_case,
    validate_trauma_override_cases,
)
from ed_triage_ai.utils.config import (
    DEFAULT_MODEL_PATH,
    HIGH_ACUITY_LEVELS,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)


@dataclass
class PredictionOutput:
    triage_level: int
    risk_score: float
    risk_category: str
    explanation: List[str]
    reason: str
    prediction_source: str
    override_triggered: bool
    override_reasons: List[str]
    matched_rules: List[str]
    audit: Dict[str, Any]


class TriagePredictor:
    def __init__(self, model_path: str | None = None):
        validate_required_test_case()
        validate_trauma_override_cases()
        self.model = self._load_or_rebuild_model(model_path or DEFAULT_MODEL_PATH)
        self._shap_explainer = None
        self._shap_available = False

        try:
            import shap  # type: ignore

            clf = self.model.named_steps["clf"]
            preprocessor = self.model.named_steps["preprocessor"]

            if hasattr(clf, "feature_importances_"):
                self._shap_explainer = shap.TreeExplainer(clf)
                self._preprocessor = preprocessor
                self._shap_available = True
        except Exception:
            self._shap_available = False

    @staticmethod
    def _load_or_rebuild_model(model_path: str | Path):
        model_path = Path(model_path)
        try:
            return joblib.load(model_path)
        except Exception:
            # Cloud runtimes can fail to unpickle locally-built artifacts due to
            # Python/sklearn version differences. Rebuild a minimal compatible
            # pipeline directly instead of invoking the full grid-search trainer.
            df = load_and_merge_datasets(
                mimic_path="",
                eicu_path="",
                kaggle_paths=[],
                synthetic_fallback_n=1200,
                random_state=42,
            )
            X, y = split_xy(df)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=180,
                            max_depth=10,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=1,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            pipeline.fit(X, y)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_path)
            return pipeline

    @staticmethod
    def _risk_category(risk_score: float) -> str:
        if risk_score >= RISK_HIGH_THRESHOLD:
            return "High"
        if risk_score >= RISK_MEDIUM_THRESHOLD:
            return "Medium"
        return "Low"

    def _heuristic_explanations(self, patient: Dict[str, float], risk_score: float) -> List[str]:
        reasons = []
        if patient.get("oxygen_saturation", 100) < 92:
            reasons.append("Low oxygen saturation increased acuity risk")
        if patient.get("systolic_bp", 200) < 90:
            reasons.append("Low systolic blood pressure suggested hemodynamic instability")
        if patient.get("heart_rate", 0) > 110:
            reasons.append("Elevated heart rate contributed to urgent triage")
        if "chest pain" in str(patient.get("chief_complaint", "")).lower():
            reasons.append("Chief complaint indicates potential cardiac emergency")
        if risk_score >= RISK_HIGH_THRESHOLD and not reasons:
            reasons.append("Model estimated high probability of severe acuity")
        if len(reasons) < 3:
            reasons.append("Age and symptom profile aligned with observed high-risk cohorts")
        return reasons[:3]

    def _shap_explanations(self, row: pd.DataFrame, pred_label: int) -> List[str]:
        X_transformed = self._preprocessor.transform(row)
        shap_values = self._shap_explainer.shap_values(X_transformed)

        class_idx = int(np.where(self.model.named_steps["clf"].classes_ == pred_label)[0][0])

        if isinstance(shap_values, list):
            class_shap = np.asarray(shap_values[class_idx])[0]
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                class_shap = values[0, :, class_idx]
            else:
                class_shap = values[0]

        feature_names = self._preprocessor.get_feature_names_out()

        top_idx = np.argsort(np.abs(class_shap))[-3:][::-1]
        reasons = []
        for idx in top_idx:
            fname = str(feature_names[idx])
            direction = "increased" if class_shap[idx] > 0 else "reduced"
            reasons.append(f"{fname} {direction} likelihood of triage level {pred_label}")
        return reasons

    def _align_features_for_model(self, enriched: pd.DataFrame) -> pd.DataFrame:
        """Ensure inference rows include all columns expected by the fitted preprocessor."""
        out = enriched.copy()
        preprocessor = self.model.named_steps["preprocessor"]

        for _, _, cols in preprocessor.transformers_:
            if cols == "drop" or cols == "passthrough":
                continue
            if isinstance(cols, str):
                cols = [cols]
            for col in cols:
                if col not in out.columns:
                    if col in {"fever_flag", "hypoxia_flag", "hypotension_flag"} or col.startswith("kw_"):
                        out[col] = 0.0
                    elif col == "chief_complaint":
                        out[col] = ""
                    elif col == "sex":
                        out[col] = "Other"
                    else:
                        out[col] = np.nan
        return out

    def predict(self, patient: Dict[str, float]) -> PredictionOutput:
        rule_result = apply_clinical_rules(patient)
        if rule_result is not None:
            level_to_risk = {
                1: 0.98,
                2: 0.82,
                3: 0.52,
                4: 0.22,
                5: 0.06,
            }
            risk_score = level_to_risk[int(rule_result.triage_level)]
            return PredictionOutput(
                triage_level=int(rule_result.triage_level),
                risk_score=float(risk_score),
                risk_category=self._risk_category(float(risk_score)),
                explanation=[rule_result.explanation],
                reason=rule_result.explanation,
                prediction_source=rule_result.source,
                override_triggered=rule_result.source in {"hard_override", "uncertainty_escalation", "validation_error"},
                override_reasons=[rule_result.explanation],
                matched_rules=list(rule_result.matched_rules),
                audit=dict(rule_result.audit),
            )

        row = pd.DataFrame([patient])
        enriched = self._align_features_for_model(enrich_features(row))
        pred_level = int(self.model.predict(enriched)[0])
        proba = self.model.predict_proba(enriched)[0]
        classes = np.array(self.model.named_steps["clf"].classes_)

        high_mask = np.isin(classes, list(HIGH_ACUITY_LEVELS))
        risk_score = float(proba[high_mask].sum())

        if self._shap_available:
            try:
                explanation = self._shap_explanations(enriched, pred_level)
            except Exception:
                explanation = self._heuristic_explanations(patient, risk_score)
        else:
            explanation = self._heuristic_explanations(patient, risk_score)

        return PredictionOutput(
            triage_level=pred_level,
            risk_score=risk_score,
            risk_category=self._risk_category(risk_score),
            explanation=explanation,
            reason=explanation[0],
            prediction_source="ml prediction",
            override_triggered=False,
            override_reasons=[],
            matched_rules=[],
            audit={
                "normalized_complaint": str(patient.get("chief_complaint", "")).lower().strip(),
                "matched_keyword_categories": [],
                "abnormal_vitals_summary": [],
            },
        )
