# ED Triage AI System

Production-style Emergency Department triage assistant combining machine learning, NLP symptom parsing, and a clinical safety rules layer.

## Features

- Multi-dataset training pipeline with schema harmonization:
  - MIMIC-IV-ED (primary, when available)
  - eICU Collaborative Research Database
  - Optional Kaggle emergency/triage tables
  - Synthetic fallback with realistic correlations if protected datasets are not locally available
- Multimodal modeling using:
  - Demographics + vital signs
  - Derived clinical indicators (shock index, fever/hypoxia/hypotension flags)
  - Free-text chief complaint via TF-IDF
- Model comparison:
  - Logistic Regression baseline
  - Random Forest
  - XGBoost primary model (with fallback if unavailable)
- 5-fold cross-validation + hyperparameter tuning (GridSearchCV)
- Oversampling for critical acuity classes (Levels 1 and 2) during training
- Safety override rules for critical conditions
- Explainability:
  - Global feature importance (from model metrics/artifacts)
  - Local explanation via SHAP when available, heuristic fallback otherwise
- Streamlit UI for interactive triage prediction
- Bonus outputs:
  - ICU admission and hospitalization labels in generated data
  - Calibration curve and model comparison plot artifacts

## Project Structure

```text
ed_triage_ai/
  data/
    generate_data.py
    loaders.py
    preprocess.py
  models/
    train.py
    evaluate.py
    predict.py
  nlp/
    symptom_parser.py
  rules/
    clinical_rules.py
  app/
    app.py
  utils/
  artifacts/               # generated outputs
```

## Setup

```bash
cd /Users/vedangholay/Visual_studioJava/PingPong/SnakeGame
python3 -m venv .venv
source .venv/bin/activate
pip install -r ed_triage_ai/requirements.txt
```

## Datasets

### PhysioNet sources (recommended)

1. Request and obtain credentialed access to:
   - MIMIC-IV-ED
   - eICU Collaborative Research Database
2. Export structured tables (CSV or Parquet) containing columns that can map to:
   `age, sex, heart_rate, systolic_bp, diastolic_bp, respiratory_rate, oxygen_saturation, temperature, pain_score, chief_complaint, triage_level`
3. Use the training command with paths:

```bash
python -m ed_triage_ai.models.train \
  --mimic-path /path/to/mimic_ed_extract.csv \
  --eicu-path /path/to/eicu_extract.csv \
  --kaggle-paths /path/to/kaggle_triage.csv \
  --cv-folds 5
```

If PhysioNet datasets are unavailable, the pipeline automatically falls back to synthetic data:

```bash
python -m ed_triage_ai.models.train --n-samples 12000 --cv-folds 5
```

## Train

```bash
python -m ed_triage_ai.models.train --n-samples 12000 --cv-folds 5
```

Fast local run without plot generation:

```bash
python -m ed_triage_ai.models.train --n-samples 800 --cv-folds 2 --skip-plots
```

Outputs are saved to `ed_triage_ai/artifacts/`:

- `best_model.joblib`
- `model_report.json`
- `test_predictions.csv`
- `model_comparison.png`
- `calibration_curve.png`
- `global_feature_importance.csv`

Synthetic dataset is saved to `ed_triage_ai/data/synthetic_ed_triage.csv`.

`model_report.json` includes:
- dataset source composition
- class distribution before/after oversampling
- holdout model metrics including `recall_level1`
- hybrid pipeline metrics (rules first, then ML)
- requirement check for Level 1 recall target (`>= 0.95`) on the deployed hybrid pipeline

## Run App

```bash
streamlit run ed_triage_ai/app/app.py
```

To enable AI summaries locally, use an environment variable or Streamlit secrets:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
streamlit run ed_triage_ai/app/app.py
```

Or create `.streamlit/secrets.toml` from `.streamlit/secrets.example.toml`.

## Deploy

Best hosting option for the current codebase: Streamlit Community Cloud.

Why:
- the UI is already a Streamlit app
- it supports secret management cleanly
- it avoids rewriting the frontend for a serverless host

Deploy steps:
1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Use app entrypoint: `ed_triage_ai/app/app.py`
4. In the app secrets panel, add:

```toml
OPENROUTER_API_KEY = "your-rotated-openrouter-key"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

5. Deploy.

The root [`requirements.txt`](/Users/vedangholay/Visual_studioJava/PingPong/SnakeGame/requirements.txt) forwards to the app dependency file so Streamlit Cloud installs the correct packages automatically.

If you need a non-Streamlit host later, Render is the next-best option because it can run the current app as a long-lived web process. Vercel is not a good fit for this architecture.

## Prediction Output

Each inference returns:

- `triage_level` (ESI 1-5)
- `risk_score` (probability of high acuity: ESI 1-2)
- `risk_category` (`High`/`Medium`/`Low`)
- `explanation` (top 3 reasons)
- `override_triggered` + `override_reasons`

## Clinical Safety Rules (examples)

- `oxygen_saturation < 90` -> escalate to high acuity
- `systolic_bp < 90` -> critical escalation
- `chest pain + HR > 110` -> escalate
- `altered mental status` or `stroke` symptoms -> highest urgency

## Disclaimer

For educational/research prototyping only. Not a substitute for clinician judgment or validated medical device software.
