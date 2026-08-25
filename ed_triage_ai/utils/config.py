"""Project-wide constants for ED triage AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "ed_triage_ai" / "artifacts"
DATA_DIR = PROJECT_ROOT / "ed_triage_ai" / "data"

RAW_FEATURES = [
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
]

DERIVED_FEATURES = [
    "shock_index",
    "fever_flag",
    "hypoxia_flag",
]

TARGET = "triage_level"

HIGH_ACUITY_LEVELS = {1, 2}
RISK_HIGH_THRESHOLD = 0.65
RISK_MEDIUM_THRESHOLD = 0.35

DEFAULT_MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
DEFAULT_REPORT_PATH = ARTIFACT_DIR / "model_report.json"
DEFAULT_DATASET_PATH = DATA_DIR / "synthetic_ed_triage.csv"
