"""
CardioVision - Prediction Pipeline
=====================================
Implements FR-15 (Prediction Pipeline) and FR-12 (Explainability).

Pipeline: Input -> Validation -> Imputation -> Scaling -> Feature Selection
          -> Prediction -> Probability -> Threshold -> Risk Category -> Explanation
"""

import numpy as np
import pandas as pd

import config
from utils import load_json, load_pickle, recommendation_for, risk_category

_CACHE = {}


def _load_artifacts():
    """Lazily load and cache model + preprocessing artifacts."""
    if not _CACHE:
        _CACHE["model"] = load_pickle(config.MODEL_PATH)
        _CACHE["imputer"] = load_pickle(config.IMPUTER_PATH)
        _CACHE["scaler"] = load_pickle(config.SCALER_PATH)
        _CACHE["selector"] = load_pickle(config.SELECTOR_PATH)
        _CACHE["selected_features"] = load_json(config.FEATURE_LIST_PATH)
        _CACHE["threshold"] = load_json(config.THRESHOLD_PATH)["threshold"]
    return _CACHE


def validate_input(input_dict):
    """Ensure all required raw features are present and numeric."""
    missing = [f for f in config.ALL_FEATURES if f not in input_dict]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    row = {}
    for f in config.ALL_FEATURES:
        try:
            row[f] = float(input_dict[f])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{f}' must be numeric, got: {input_dict[f]!r}")
    return row


def predict_single(input_dict, explain=True):
    """
    Run the full inference pipeline for a single patient record.

    Parameters
    ----------
    input_dict : dict mapping raw feature name -> value
    explain : bool, whether to attach a layman explanation of top drivers

    Returns
    -------
    dict with prediction, probability, risk category, and explanation
    """
    artifacts = _load_artifacts()

    # 1. Validation
    row = validate_input(input_dict)
    X_raw = pd.DataFrame([row])[config.ALL_FEATURES]

    # 2. Imputation
    X_imp = pd.DataFrame(
        artifacts["imputer"].transform(X_raw), columns=config.ALL_FEATURES
    )

    # 3. Scaling
    X_scaled = pd.DataFrame(
        artifacts["scaler"].transform(X_imp), columns=config.ALL_FEATURES
    )

    # 4. Feature Selection
    selected = artifacts["selected_features"]
    X_selected = X_scaled[selected]

    # 5. Prediction / Probability
    probability = float(artifacts["model"].predict_proba(X_selected)[:, 1][0])

    # 6. Threshold -> class
    threshold = artifacts["threshold"]
    prediction = int(probability >= threshold)

    # 7. Risk Category
    category = risk_category(probability, threshold)

    result = {
        "prediction": prediction,
        "prediction_label": "Heart Disease Likely" if prediction == 1 else "Heart Disease Unlikely",
        "probability": round(probability, 4),
        "probability_pct": f"{probability * 100:.1f}%",
        "threshold_used": round(threshold, 4),
        "risk_category": category,
        "recommendation": recommendation_for(category),
    }

    # 8. Explanation (FR-12)
    if explain:
        result["top_factors"] = explain_prediction(artifacts["model"], X_selected, selected)

    return result


def explain_prediction(model, X_selected, selected_features, top_n=3):
    """Rank the top contributing features for this single prediction using
    the model's global feature importances weighted by the input's
    (scaled) feature magnitude — a lightweight, dependency-free explainer."""
    importances = model.feature_importances_
    values = X_selected.iloc[0].values
    contribution = np.abs(importances * values)

    order = np.argsort(contribution)[::-1][:top_n]
    factors = []
    for idx in order:
        feat = selected_features[idx]
        meta = config.FEATURE_EXPLANATIONS.get(feat, {})
        factors.append({
            "feature": feat,
            "display_name": meta.get("display_name", feat),
            "explanation": meta.get("explanation", ""),
        })
    return factors


def predict_batch(df):
    """Run predictions for a DataFrame of raw patient records."""
    results = [predict_single(row.to_dict(), explain=False) for _, row in df.iterrows()]
    return pd.DataFrame(results)


if __name__ == "__main__":
    sample = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
    result = predict_single(sample)
    import json
    print(json.dumps(result, indent=2))
