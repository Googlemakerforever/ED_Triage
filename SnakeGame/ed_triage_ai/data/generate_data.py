"""Generate realistic synthetic ED triage data when MIMIC-IV-ED is unavailable."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ed_triage_ai.utils.config import DEFAULT_DATASET_PATH


@dataclass(frozen=True)
class SeverityProfile:
    hr_mu: float
    hr_sd: float
    sbp_mu: float
    sbp_sd: float
    rr_mu: float
    rr_sd: float
    spo2_mu: float
    spo2_sd: float
    temp_mu: float
    temp_sd: float
    pain_mu: float
    pain_sd: float


PROFILES = {
    1: SeverityProfile(128, 16, 84, 10, 31, 5, 86, 5, 38.7, 1.0, 8.7, 1.2),
    2: SeverityProfile(111, 14, 98, 12, 24, 4, 92, 4, 38.1, 0.9, 7.2, 1.6),
    3: SeverityProfile(96, 12, 116, 13, 20, 3, 95, 3, 37.6, 0.8, 5.8, 2.0),
    4: SeverityProfile(84, 10, 126, 14, 17, 3, 97, 2, 37.1, 0.7, 4.0, 1.8),
    5: SeverityProfile(76, 9, 132, 14, 15, 2, 98, 2, 36.9, 0.6, 2.6, 1.5),
}

COMPLAINTS = {
    1: [
        "altered mental status and hypotension",
        "severe shortness of breath with chest pain",
        "active hemorrhage after trauma",
        "possible stroke with slurred speech",
    ],
    2: [
        "chest pain radiating to left arm",
        "shortness of breath and tachycardia",
        "syncope and dizziness",
        "high fever with confusion",
    ],
    3: [
        "abdominal pain with vomiting",
        "moderate asthma exacerbation",
        "fever and cough for 3 days",
        "headache and nausea",
    ],
    4: [
        "ankle sprain from minor fall",
        "mild allergic reaction",
        "sore throat and low fever",
        "small laceration without bleeding",
    ],
    5: [
        "medication refill request",
        "mild rash no respiratory symptoms",
        "chronic back pain stable",
        "administrative medical clearance",
    ],
}


TRIAGE_DISTRIBUTION = np.array([0.02, 0.14, 0.36, 0.30, 0.18])


def _clip(v: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(v, low), high)


def generate_synthetic_ed_data(n_samples: int = 12000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    triage = rng.choice(np.arange(1, 6), size=n_samples, p=TRIAGE_DISTRIBUTION)
    age = _clip(rng.normal(49, 20, size=n_samples), 0, 97).round()
    sex = rng.choice(["Female", "Male", "Other"], p=[0.49, 0.49, 0.02], size=n_samples)

    hr, sbp, dbp, rr, spo2, temp, pain = [], [], [], [], [], [], []
    complaints = []

    for level in triage:
        level_int = int(level)
        profile = PROFILES[level_int]

        # Add overlap between adjacent acuity bands to mimic real-world ambiguity.
        if rng.uniform() < 0.2:
            neighbor = np.clip(level_int + rng.choice([-1, 1]), 1, 5)
            neighbor_profile = PROFILES[int(neighbor)]
            blend = rng.uniform(0.25, 0.55)
            profile = SeverityProfile(
                hr_mu=profile.hr_mu * (1 - blend) + neighbor_profile.hr_mu * blend,
                hr_sd=profile.hr_sd * (1 - blend) + neighbor_profile.hr_sd * blend,
                sbp_mu=profile.sbp_mu * (1 - blend) + neighbor_profile.sbp_mu * blend,
                sbp_sd=profile.sbp_sd * (1 - blend) + neighbor_profile.sbp_sd * blend,
                rr_mu=profile.rr_mu * (1 - blend) + neighbor_profile.rr_mu * blend,
                rr_sd=profile.rr_sd * (1 - blend) + neighbor_profile.rr_sd * blend,
                spo2_mu=profile.spo2_mu * (1 - blend) + neighbor_profile.spo2_mu * blend,
                spo2_sd=profile.spo2_sd * (1 - blend) + neighbor_profile.spo2_sd * blend,
                temp_mu=profile.temp_mu * (1 - blend) + neighbor_profile.temp_mu * blend,
                temp_sd=profile.temp_sd * (1 - blend) + neighbor_profile.temp_sd * blend,
                pain_mu=profile.pain_mu * (1 - blend) + neighbor_profile.pain_mu * blend,
                pain_sd=profile.pain_sd * (1 - blend) + neighbor_profile.pain_sd * blend,
            )

        complaint_pool = COMPLAINTS[level_int][:]
        if rng.uniform() < 0.22:
            # Include non-specific or cross-acuity complaints.
            complaint_pool += COMPLAINTS[int(np.clip(level_int + rng.choice([-1, 1]), 1, 5))]
        complaint = rng.choice(complaint_pool)

        heart_rate = rng.normal(profile.hr_mu, profile.hr_sd + 2.5)
        systolic = rng.normal(profile.sbp_mu, profile.sbp_sd + 2.0)
        respiratory = rng.normal(profile.rr_mu, profile.rr_sd + 1.0)
        oxygen = rng.normal(profile.spo2_mu, profile.spo2_sd + 0.8)
        temperature = rng.normal(profile.temp_mu, profile.temp_sd + 0.25)
        pain_score = rng.normal(profile.pain_mu, profile.pain_sd + 0.5)

        complaint_lower = complaint.lower()
        if "chest pain" in complaint_lower:
            heart_rate += rng.normal(10, 4)
        if "shortness of breath" in complaint_lower:
            respiratory += rng.normal(4, 2)
            oxygen -= rng.normal(3, 1.2)
        if "stroke" in complaint_lower or "altered mental" in complaint_lower:
            systolic -= rng.normal(8, 4)

        diastolic = 0.62 * systolic + rng.normal(0, 6)

        hr.append(heart_rate)
        sbp.append(systolic)
        dbp.append(diastolic)
        rr.append(respiratory)
        spo2.append(oxygen)
        temp.append(temperature)
        pain.append(pain_score)
        complaints.append(complaint)

    df = pd.DataFrame(
        {
            "age": age.astype(int),
            "sex": sex,
            "heart_rate": _clip(np.array(hr), 35, 220).round(1),
            "systolic_bp": _clip(np.array(sbp), 55, 230).round(1),
            "diastolic_bp": _clip(np.array(dbp), 30, 140).round(1),
            "respiratory_rate": _clip(np.array(rr), 6, 55).round(1),
            "oxygen_saturation": _clip(np.array(spo2), 65, 100).round(1),
            "temperature": _clip(np.array(temp), 34.0, 41.8).round(1),
            "pain_score": _clip(np.array(pain), 0, 10).round(0).astype(int),
            "chief_complaint": complaints,
            "triage_level": triage.astype(int),
        }
    )

    # Small triage label noise to represent inter-rater variability.
    noise_mask = rng.uniform(0, 1, size=n_samples) < 0.06
    shift = rng.choice([-1, 1], size=n_samples)
    df.loc[noise_mask, "triage_level"] = np.clip(
        df.loc[noise_mask, "triage_level"] + shift[noise_mask],
        1,
        5,
    )

    shock_index = df["heart_rate"] / df["systolic_bp"]
    acute_score = (
        (df["triage_level"] <= 2).astype(float) * 0.5
        + (df["oxygen_saturation"] < 92).astype(float) * 0.2
        + (shock_index > 0.9).astype(float) * 0.2
        + (df["age"] > 75).astype(float) * 0.1
    )

    icu_prob = np.clip(0.02 + acute_score * 0.65, 0.01, 0.95)
    hosp_prob = np.clip(0.08 + acute_score * 0.55 + (df["triage_level"] <= 3).astype(float) * 0.2, 0.05, 0.97)

    df["icu_admission"] = (rng.uniform(0, 1, size=n_samples) < icu_prob).astype(int)
    df["hospitalization"] = (rng.uniform(0, 1, size=n_samples) < hosp_prob).astype(int)

    for col, miss_rate in {
        "heart_rate": 0.03,
        "systolic_bp": 0.04,
        "diastolic_bp": 0.05,
        "respiratory_rate": 0.05,
        "oxygen_saturation": 0.03,
        "temperature": 0.04,
        "pain_score": 0.06,
    }.items():
        mask = rng.uniform(0, 1, size=n_samples) < miss_rate
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    df = generate_synthetic_ed_data()
    DEFAULT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    print(f"Saved synthetic dataset to {DEFAULT_DATASET_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
