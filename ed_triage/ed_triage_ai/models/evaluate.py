"""Model evaluation helpers."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from ed_triage_ai.utils.config import HIGH_ACUITY_LEVELS


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, classes: List[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "recall_level1": float(recall_score((y_true == 1).astype(int), (y_pred == 1).astype(int), zero_division=0)),
    }

    try:
        metrics["auroc_ovr_weighted"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        )
    except ValueError:
        metrics["auroc_ovr_weighted"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def high_acuity_risk(y_proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    high_mask = np.isin(classes, list(HIGH_ACUITY_LEVELS))
    return y_proba[:, high_mask].sum(axis=1)


def save_calibration_plot(
    y_true: pd.Series,
    risk_score: np.ndarray,
    output_path: str,
) -> None:
    y_binary = y_true.isin(HIGH_ACUITY_LEVELS).astype(int)
    prob_true, prob_pred = calibration_curve(y_binary, risk_score, n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Predicted high-acuity probability")
    plt.ylabel("Observed high-acuity frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_model_comparison(results_df: pd.DataFrame, output_path: str) -> None:
    plot_df = results_df.copy()
    plot_df = plot_df.sort_values("f1_weighted", ascending=False)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(plot_df))
    width = 0.25
    plt.bar(x - width, plot_df["accuracy"], width=width, label="Accuracy")
    plt.bar(x, plot_df["f1_weighted"], width=width, label="F1 weighted")
    plt.bar(x + width, plot_df["auroc_ovr_weighted"], width=width, label="AUROC")
    plt.xticks(x, plot_df["model"], rotation=15)
    plt.ylim(0, 1.0)
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _safe_feature_names(preprocessor) -> np.ndarray:
    try:
        return np.asarray(preprocessor.get_feature_names_out())
    except Exception:
        names: List[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue

            if isinstance(cols, str):
                cols = [cols]

            if name == "txt":
                tfidf = transformer.named_steps["tfidf"]
                txt_names = tfidf.get_feature_names_out()
                names.extend([f"{name}__{t}" for t in txt_names])
                continue

            if hasattr(transformer, "get_feature_names_out"):
                try:
                    out_names = transformer.get_feature_names_out(cols)
                    names.extend(list(out_names))
                    continue
                except Exception:
                    pass

            names.extend([f"{name}__{c}" for c in cols])

        return np.asarray(names)


def global_feature_importance(pipeline, top_n: int = 25) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]
    feature_names = _safe_feature_names(preprocessor)

    if hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_).ravel()
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef).ravel()
    else:
        raise ValueError("Model does not expose feature importance or coefficients.")

    out = pd.DataFrame({"feature": feature_names, "importance": values})
    out = out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return out
