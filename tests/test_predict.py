"""Tests for predict.py (FR-15 Prediction Pipeline) and model loading.

These tests require trained artifacts to exist in models/ (run `python
train.py` first). They are skipped automatically if artifacts are missing,
so CI can run `train.py` before `pytest`.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

pytestmark = pytest.mark.skipif(
    not os.path.exists(config.MODEL_PATH),
    reason="Trained model artifacts not found; run `python train.py` first.",
)

SAMPLE_RECORD = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


def test_model_loading():
    from utils import load_pickle
    model = load_pickle(config.MODEL_PATH)
    assert hasattr(model, "predict_proba")


def test_validate_input_rejects_missing_field():
    import predict
    bad_record = SAMPLE_RECORD.copy()
    del bad_record["age"]
    with pytest.raises(ValueError):
        predict.validate_input(bad_record)


def test_validate_input_rejects_non_numeric():
    import predict
    bad_record = SAMPLE_RECORD.copy()
    bad_record["age"] = "not-a-number"
    with pytest.raises(ValueError):
        predict.validate_input(bad_record)


def test_predict_single_output_schema():
    import predict
    result = predict.predict_single(SAMPLE_RECORD)
    for key in ("prediction", "probability", "risk_category", "recommendation", "top_factors"):
        assert key in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["top_factors"]) > 0


def test_predict_batch():
    import pandas as pd
    import predict
    df = pd.DataFrame([SAMPLE_RECORD, SAMPLE_RECORD])
    out = predict.predict_batch(df)
    assert len(out) == 2
    assert "prediction" in out.columns
