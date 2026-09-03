"""Build Dataset-1 feature vectors from manual weekly check-in forms."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_ORDER = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "stress_level_ord",
    "gender_Male",
    "gender_Other",
    "reward_app_share",
]

FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "daily_screen_time_hours": "Daily screen time",
    "social_media_hours": "Social media use",
    "gaming_hours": "Gaming time",
    "work_study_hours": "Study / work time",
    "sleep_hours": "Sleep duration",
    "notifications_per_day": "Notifications per day",
    "app_opens_per_day": "App opens per day",
    "stress_level_ord": "Stress level",
    "gender_Male": "Gender (Male)",
    "gender_Other": "Gender (Other)",
    "reward_app_share": "Reward-app share of screen time",
}

STRESS_TO_ORD = {"Low": 0, "Medium": 1, "High": 2}


def build_feature_row(
    *,
    age: int,
    gender: str,
    daily_screen_time_hours: float,
    social_media_hours: float,
    gaming_hours: float,
    work_study_hours: float,
    sleep_hours: float,
    notifications_per_day: int,
    app_opens_per_day: int,
    stress_level: str,
) -> pd.DataFrame:
    """Convert user form inputs into a one-row DataFrame matching model training."""
    gender_norm = gender.strip().title()
    gender_male = 1 if gender_norm == "Male" else 0
    gender_other = 1 if gender_norm == "Other" else 0

    stress_ord = STRESS_TO_ORD.get(stress_level.strip().title())
    if stress_ord is None:
        raise ValueError(f"Unknown stress level: {stress_level}")

    eps = 1e-6
    reward_app_share = float(
        np.clip(
            (social_media_hours + gaming_hours) / (daily_screen_time_hours + eps),
            0.0,
            3.0,
        )
    )

    row: dict[str, Any] = {
        "age": int(age),
        "daily_screen_time_hours": float(daily_screen_time_hours),
        "social_media_hours": float(social_media_hours),
        "gaming_hours": float(gaming_hours),
        "work_study_hours": float(work_study_hours),
        "sleep_hours": float(sleep_hours),
        "notifications_per_day": int(notifications_per_day),
        "app_opens_per_day": int(app_opens_per_day),
        "stress_level_ord": int(stress_ord),
        "gender_Male": int(gender_male),
        "gender_Other": int(gender_other),
        "reward_app_share": reward_app_share,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def form_payload_from_inputs(**kwargs: Any) -> dict[str, Any]:
    """Store the raw form values (for check-in history), not only model features."""
    return {
        "age": kwargs["age"],
        "gender": kwargs["gender"],
        "daily_screen_time_hours": kwargs["daily_screen_time_hours"],
        "social_media_hours": kwargs["social_media_hours"],
        "gaming_hours": kwargs["gaming_hours"],
        "work_study_hours": kwargs["work_study_hours"],
        "sleep_hours": kwargs["sleep_hours"],
        "notifications_per_day": kwargs["notifications_per_day"],
        "app_opens_per_day": kwargs["app_opens_per_day"],
        "stress_level": kwargs["stress_level"],
    }
