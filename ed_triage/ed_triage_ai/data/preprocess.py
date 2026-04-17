"""Feature engineering and preprocessing pipeline definitions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ed_triage_ai.nlp.symptom_parser import KEYWORD_PATTERNS, append_keyword_features

NUMERIC_FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "oxygen_saturation",
    "temperature",
    "pain_score",
    "shock_index",
]

BINARY_FEATURES = ["fever_flag", "hypoxia_flag", "hypotension_flag", *KEYWORD_PATTERNS.keys()]
CATEGORICAL_FEATURES = ["sex"]
TEXT_FEATURE = "chief_complaint"


def _clean_text_column(values):
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(np.asarray(values).ravel())
    return series.fillna("").astype(str)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sbp = out["systolic_bp"].replace(0, np.nan)
    out["shock_index"] = out["heart_rate"] / sbp
    out["shock_index"] = out["shock_index"].replace([np.inf, -np.inf], np.nan)

    out["fever_flag"] = (out["temperature"] > 38.0).astype(float)
    out["hypoxia_flag"] = (out["oxygen_saturation"] < 92).astype(float)
    out["hypotension_flag"] = (out["systolic_bp"] < 90).astype(float)
    return out


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    return append_keyword_features(add_derived_features(df), text_col=TEXT_FEATURE)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    text_pipe = Pipeline(
        steps=[
            (
                "clean_text",
                FunctionTransformer(
                    _clean_text_column,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=300,
                    lowercase=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("txt", text_pipe, [TEXT_FEATURE]),
        ]
    )


def get_model_features() -> List[str]:
    return NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]


def split_xy(df: pd.DataFrame, target: str = "triage_level") -> Tuple[pd.DataFrame, pd.Series]:
    feat = enrich_features(df)
    X = feat[get_model_features()].copy()
    y = feat[target].astype(int)
    return X, y
