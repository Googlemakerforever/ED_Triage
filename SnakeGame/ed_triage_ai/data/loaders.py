"""Dataset loaders and schema standardization for multi-source ED training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ed_triage_ai.data.generate_data import generate_synthetic_ed_data

REQUIRED_COLUMNS = [
    "age",
    "sex",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "oxygen_saturation",
    "temperature",
    "pain_score",
    "chief_complaint",
    "triage_level",
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "age": ["age", "anchor_age", "patient_age"],
    "sex": ["sex", "gender", "patient_sex"],
    "heart_rate": ["heart_rate", "heartrate", "hr", "triage_heartrate", "vital_hr"],
    "systolic_bp": ["systolic_bp", "sbp", "triage_sbp", "vital_sbp", "blood_pressure_systolic"],
    "diastolic_bp": ["diastolic_bp", "dbp", "triage_dbp", "vital_dbp", "blood_pressure_diastolic"],
    "respiratory_rate": ["respiratory_rate", "rr", "resp_rate", "triage_resprate", "vital_rr"],
    "oxygen_saturation": ["oxygen_saturation", "spo2", "o2_sat", "triage_o2sat", "vital_spo2"],
    "temperature": ["temperature", "temp", "temperature_c", "triage_temperature", "vital_temp"],
    "pain_score": ["pain_score", "pain", "pain_scale", "triage_pain"],
    "chief_complaint": ["chief_complaint", "complaint", "chiefcomplaint", "arrival_reason", "chiefcomplainttext"],
    "triage_level": ["triage_level", "acuity", "esi", "esi_level", "triage_acuity"],
}

NUMERIC_COLUMNS = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "oxygen_saturation",
    "temperature",
    "pain_score",
    "triage_level",
]

DEFAULT_IMPUTE = {
    "age": 50,
    "heart_rate": 88,
    "systolic_bp": 118,
    "diastolic_bp": 72,
    "respiratory_rate": 18,
    "oxygen_saturation": 97,
    "temperature": 37.0,
    "pain_score": 3,
}


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type for {path}. Use CSV or Parquet.")


def _find_column(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES[canonical_name]:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _normalize_sex(value: object) -> str:
    v = str(value).strip().lower()
    if v in {"m", "male", "man"}:
        return "Male"
    if v in {"f", "female", "woman"}:
        return "Female"
    if v in {"other", "non-binary", "nonbinary"}:
        return "Other"
    return "Other"


def _normalize_temperature_c(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median = s.median(skipna=True)
    if pd.notna(median) and median > 45:
        s = (s - 32.0) * 5.0 / 9.0
    return s


def _normalize_oxygen_sat(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median = s.median(skipna=True)
    if pd.notna(median) and 0 <= median <= 1.0:
        s = s * 100.0
    return s


def _normalize_pain(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median = s.median(skipna=True)
    if pd.notna(median) and median > 10 and median <= 100:
        s = s / 10.0
    return s


def standardize_schema(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in REQUIRED_COLUMNS:
        src_col = _find_column(df, col)
        out[col] = df[src_col] if src_col else np.nan

    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["sex"] = out["sex"].map(_normalize_sex)

    for col in ["heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "triage_level"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["oxygen_saturation"] = _normalize_oxygen_sat(out["oxygen_saturation"])
    out["temperature"] = _normalize_temperature_c(out["temperature"])
    out["pain_score"] = _normalize_pain(out["pain_score"])
    out["chief_complaint"] = out["chief_complaint"].fillna("unknown complaint").astype(str)

    out = clean_standardized_df(out)
    out["data_source"] = dataset_name
    return out


def clean_standardized_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out[out["triage_level"].isin([1, 2, 3, 4, 5])]

    ranges = {
        "age": (0, 110),
        "heart_rate": (20, 240),
        "systolic_bp": (50, 260),
        "diastolic_bp": (20, 160),
        "respiratory_rate": (4, 60),
        "oxygen_saturation": (50, 100),
        "temperature": (33, 43),
        "pain_score": (0, 10),
    }

    for col, (low, high) in ranges.items():
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[(out[col] < low) | (out[col] > high), col] = np.nan

    for col, default in DEFAULT_IMPUTE.items():
        med = out[col].median(skipna=True)
        out[col] = out[col].fillna(med if pd.notna(med) else default)

    out["pain_score"] = np.clip(out["pain_score"], 0, 10).round(0)
    out["triage_level"] = out["triage_level"].astype(int)

    out = out.dropna(subset=["chief_complaint"])
    out["chief_complaint"] = out["chief_complaint"].astype(str).str.strip().replace("", "unknown complaint")

    return out.reset_index(drop=True)


def load_mimic_data(path: str) -> pd.DataFrame:
    return standardize_schema(_read_table(path), dataset_name="mimic_iv_ed")


def load_eicu_data(path: str) -> pd.DataFrame:
    return standardize_schema(_read_table(path), dataset_name="eicu")


def load_kaggle_data(path: str) -> pd.DataFrame:
    return standardize_schema(_read_table(path), dataset_name="kaggle")


def _balance_source_representation(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    if "data_source" not in df.columns or df.empty:
        return df

    counts = df["data_source"].value_counts()
    if len(counts) <= 1:
        return df

    total = len(df)
    max_allowed = int(0.7 * total)
    dominant = counts.idxmax()

    if counts.max() <= max_allowed:
        return df

    keep_idx = df[df["data_source"] != dominant].index
    dominant_df = df[df["data_source"] == dominant].sample(n=max_allowed, random_state=random_state)
    out = pd.concat([df.loc[keep_idx], dominant_df], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_and_merge_datasets(
    mimic_path: str = "",
    eicu_path: str = "",
    kaggle_paths: Optional[Iterable[str]] = None,
    synthetic_fallback_n: int = 12000,
    random_state: int = 42,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    if mimic_path:
        frames.append(load_mimic_data(mimic_path))
    if eicu_path:
        frames.append(load_eicu_data(eicu_path))
    if kaggle_paths:
        for path in kaggle_paths:
            if path:
                frames.append(load_kaggle_data(path))

    if not frames:
        fallback = generate_synthetic_ed_data(n_samples=synthetic_fallback_n, random_state=random_state)
        fallback = standardize_schema(fallback, dataset_name="synthetic_fallback")
        return fallback.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=REQUIRED_COLUMNS)
    combined_df = _balance_source_representation(combined_df, random_state=random_state)
    combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return combined_df
