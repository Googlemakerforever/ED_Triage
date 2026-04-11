"""Train ED triage models and persist the best pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from ed_triage_ai.data.loaders import load_and_merge_datasets
from ed_triage_ai.data.preprocess import build_preprocessor, split_xy
from ed_triage_ai.rules.clinical_rules import apply_clinical_rules
from ed_triage_ai.models.evaluate import (
    compute_metrics,
    global_feature_importance,
    high_acuity_risk,
    save_calibration_plot,
    save_model_comparison,
)
from ed_triage_ai.utils.config import ARTIFACT_DIR, DEFAULT_DATASET_PATH, TARGET
from ed_triage_ai.utils.io import save_json


def _get_xgb_classifier() -> Tuple[str, object]:
    if sys.platform == "darwin":
        libomp_path = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
        if not libomp_path.exists():
            model = ExtraTreesClassifier(
                n_estimators=450,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=1,
                class_weight="balanced",
            )
            return "xgboost_fallback", model

    try:
        from xgboost import XGBClassifier  # type: ignore

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )
        return "xgboost", model
    except Exception:
        model = ExtraTreesClassifier(
            n_estimators=450,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        )
        return "xgboost_fallback", model


def _model_grid() -> Dict[str, Tuple[object, Dict[str, list]]]:
    xgb_name, xgb_model = _get_xgb_classifier()

    grids: Dict[str, Tuple[object, Dict[str, list]]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=1500, class_weight="balanced"),
            {"clf__C": [0.5, 1.0, 2.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced"),
            {
                "clf__n_estimators": [250, 450],
                "clf__max_depth": [8, None],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
        xgb_name: (
            xgb_model,
            (
                {
                    "clf__n_estimators": [220, 420],
                    "clf__max_depth": [4, 6],
                    "clf__learning_rate": [0.05, 0.1],
                    "clf__subsample": [0.8, 1.0],
                }
                if xgb_name == "xgboost"
                else {
                    "clf__n_estimators": [300, 450],
                    "clf__max_depth": [12, None],
                    "clf__min_samples_leaf": [1, 2],
                }
            ),
        ),
    }

    return grids


def _oversample_critical_cases(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Oversample Levels 1 and 2 to improve critical-case sensitivity."""
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values

    counts = train_df[TARGET].value_counts()
    if counts.empty:
        return X_train, y_train

    target_count = int(counts.max())
    rng = np.random.default_rng(random_state)
    parts = [train_df]

    for level in [1, 2]:
        level_df = train_df[train_df[TARGET] == level]
        if level_df.empty:
            continue
        deficit = target_count - len(level_df)
        if deficit > 0:
            idx = rng.integers(0, len(level_df), size=deficit)
            sampled = level_df.iloc[idx].copy()
            parts.append(sampled)

    balanced = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    X_bal = balanced.drop(columns=[TARGET])
    y_bal = balanced[TARGET].astype(int)
    return X_bal, y_bal


def train(args: argparse.Namespace) -> None:
    # Legacy single-path input remains supported for compatibility.
    if args.data_path and not args.mimic_path and not args.eicu_path and not args.kaggle_paths:
        df = pd.read_csv(args.data_path)
    else:
        df = load_and_merge_datasets(
            mimic_path=args.mimic_path,
            eicu_path=args.eicu_path,
            kaggle_paths=args.kaggle_paths,
            synthetic_fallback_n=args.n_samples,
            random_state=args.seed,
        )
        DEFAULT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DEFAULT_DATASET_PATH, index=False)

    X, y = split_xy(df, target=TARGET)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train, y_train = _oversample_critical_cases(X_train, y_train, random_state=args.seed)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    model_defs = _model_grid()

    model_results = []
    best_by_name = {}

    for model_name, (clf, param_grid) in model_defs.items():
        print(f"[train] Fitting {model_name} ...", flush=True)
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("clf", clf),
            ]
        )
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            n_jobs=1,
            scoring="f1_weighted",
            verbose=0,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        pred = best_model.predict(X_test)
        proba = best_model.predict_proba(X_test)

        metrics = compute_metrics(y_test, pred, proba, classes=[1, 2, 3, 4, 5])
        metrics["model"] = model_name
        metrics["best_params"] = search.best_params_
        print(
            f"[train] {model_name}: f1={metrics['f1_weighted']:.4f}, recall_l1={metrics['recall_level1']:.4f}",
            flush=True,
        )

        model_results.append(metrics)
        best_by_name[model_name] = best_model

    results_df = pd.DataFrame(model_results).sort_values("f1_weighted", ascending=False)

    primary_name = "xgboost" if "xgboost" in best_by_name else results_df.iloc[0]["model"]
    best_pipeline = best_by_name[primary_name]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "best_model.joblib"
    results_path = ARTIFACT_DIR / "model_report.json"
    predictions_path = ARTIFACT_DIR / "test_predictions.csv"
    comparison_path = ARTIFACT_DIR / "model_comparison.png"
    calibration_path = ARTIFACT_DIR / "calibration_curve.png"
    feature_importance_path = ARTIFACT_DIR / "global_feature_importance.csv"

    joblib.dump(best_pipeline, model_path)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)
    classes = np.array(best_pipeline.named_steps["clf"].classes_)
    risk_score = high_acuity_risk(y_proba, classes)
    best_metrics = compute_metrics(y_test, y_pred, y_proba, classes=[1, 2, 3, 4, 5])
    hybrid_pred = []
    for i in range(len(X_test)):
        row = X_test.iloc[i].to_dict()
        rule_result = apply_clinical_rules(row)
        hybrid_pred.append(rule_result.triage_level if rule_result is not None else int(y_pred[i]))
    hybrid_pred = np.asarray(hybrid_pred)
    hybrid_metrics = {
        "accuracy": float(accuracy_score(y_test, hybrid_pred)),
        "f1_weighted": float(f1_score(y_test, hybrid_pred, average="weighted")),
        "recall_level1": float(recall_score((y_test == 1).astype(int), (hybrid_pred == 1).astype(int), zero_division=0)),
    }

    pred_out = X_test.copy()
    pred_out["true_triage_level"] = y_test.values
    pred_out["pred_triage_level"] = y_pred
    pred_out["risk_score"] = risk_score
    pred_out.to_csv(predictions_path, index=False)

    if not args.skip_plots:
        save_model_comparison(results_df[["model", "accuracy", "f1_weighted", "auroc_ovr_weighted"]], str(comparison_path))
        save_calibration_plot(y_test, risk_score, str(calibration_path))
    feature_importance_saved = False
    try:
        fi_df = global_feature_importance(best_pipeline, top_n=30)
        fi_df.to_csv(feature_importance_path, index=False)
        feature_importance_saved = True
    except Exception:
        pass

    summary = {
        "primary_model": primary_name,
        "dataset_summary": {
            "total_rows": int(len(df)),
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "train_distribution_after_oversampling": {
                str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
            },
            "sources": (
                {str(k): int(v) for k, v in df["data_source"].value_counts().items()}
                if "data_source" in df.columns
                else {}
            ),
        },
        "best_model_metrics": best_metrics,
        "hybrid_metrics": hybrid_metrics,
        "requirements_check": {
            "level1_recall_target": 0.95,
            "level1_recall_actual": float(hybrid_metrics.get("recall_level1", float("nan"))),
            "level1_recall_passed": bool(hybrid_metrics.get("recall_level1", 0.0) >= 0.95),
        },
        "models": model_results,
        "artifact_paths": {
            "best_model": str(model_path),
            "predictions": str(predictions_path),
            "comparison_plot": str(comparison_path) if not args.skip_plots else "",
            "calibration_plot": str(calibration_path) if not args.skip_plots else "",
            "global_feature_importance": str(feature_importance_path) if feature_importance_saved else "",
        },
    }
    save_json(results_path, summary)

    print(results_df[["model", "accuracy", "f1_weighted", "auroc_ovr_weighted"]].to_string(index=False))
    print(
        f"Best-model Level 1 recall: {best_metrics.get('recall_level1', float('nan')):.4f} "
        f"| Hybrid pipeline Level 1 recall: {hybrid_metrics.get('recall_level1', float('nan')):.4f} "
        "(target >= 0.95)"
    )
    print(f"\nPrimary model: {primary_name}")
    print(f"Saved model: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ED triage models")
    parser.add_argument("--data-path", type=str, default="", help="Optional CSV path with triage data")
    parser.add_argument("--mimic-path", type=str, default="", help="Path to standardized MIMIC-IV-ED extract (CSV/Parquet)")
    parser.add_argument("--eicu-path", type=str, default="", help="Path to standardized eICU extract (CSV/Parquet)")
    parser.add_argument(
        "--kaggle-paths",
        type=str,
        nargs="*",
        default=[],
        help="Optional one or more Kaggle emergency dataset files",
    )
    parser.add_argument("--n-samples", type=int, default=12000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot artifact generation (faster, no matplotlib).")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
