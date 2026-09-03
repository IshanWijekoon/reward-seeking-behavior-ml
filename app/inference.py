"""Load Dataset-1 XGBoost pipeline and produce risk + SHAP drivers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from app.features import FEATURE_DISPLAY_NAMES, FEATURE_ORDER

LABEL_ORDER = ["Low", "Moderate", "High"]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_model_path() -> Path:
    return project_root() / "results" / "dataset1_best_model.joblib"


def load_pipeline(model_path: Path | None = None):
    path = model_path or default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def predict_risk(pipeline, X: pd.DataFrame) -> dict[str, Any]:
    """Return predicted label, class index, and class probabilities."""
    X = X[FEATURE_ORDER]
    pred_idx = int(pipeline.predict(X.values)[0])
    proba = pipeline.predict_proba(X.values)[0]
    return {
        "risk_level": LABEL_ORDER[pred_idx],
        "risk_index": pred_idx,
        "probabilities": {
            LABEL_ORDER[i]: float(proba[i]) for i in range(len(LABEL_ORDER))
        },
    }


def top_shap_drivers(
    pipeline,
    X: pd.DataFrame,
    *,
    predicted_class: int,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Rank features by |SHAP| for the predicted class using TreeExplainer
    on the inner XGBoost estimator (SMOTE is fit-only; unused at predict).
    """
    X = X[FEATURE_ORDER]
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)

    # Handle list-per-class or (n, features, classes) layouts
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[predicted_class])[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            values = arr[0, :, predicted_class]
        elif arr.ndim == 2:
            values = arr[0]
        else:
            raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

    order = np.argsort(np.abs(values))[::-1][:top_k]
    drivers: list[dict[str, Any]] = []
    for idx in order:
        feat = FEATURE_ORDER[int(idx)]
        drivers.append(
            {
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "shap_value": float(values[int(idx)]),
                "abs_shap": float(abs(values[int(idx)])),
                "feature_value": float(X.iloc[0, int(idx)]),
                "direction": "increases risk" if values[int(idx)] > 0 else "decreases risk",
            }
        )
    return drivers


def analyze_checkin(pipeline, X: pd.DataFrame) -> dict[str, Any]:
    """Full inference bundle for one weekly check-in row."""
    prediction = predict_risk(pipeline, X)
    drivers = top_shap_drivers(
        pipeline, X, predicted_class=prediction["risk_index"], top_k=3
    )
    return {**prediction, "drivers": drivers}
