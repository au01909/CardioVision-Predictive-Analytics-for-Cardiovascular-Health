"""Tests for FR-14 (Model Persistence) and dashboard module sanity (FR-16)."""

import ast
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


def test_all_artifacts_persisted():
    for path in [config.MODEL_PATH, config.IMPUTER_PATH, config.SCALER_PATH,
                 config.SELECTOR_PATH, config.THRESHOLD_PATH, config.FEATURE_LIST_PATH]:
        assert os.path.exists(path), f"Missing artifact: {path}"


def test_threshold_json_valid():
    from utils import load_json
    data = load_json(config.THRESHOLD_PATH)
    assert 0.0 <= data["threshold"] <= 1.0


def test_metrics_present_and_reasonable():
    from utils import load_json
    if not os.path.exists(config.METRICS_PATH):
        pytest.skip("metrics.json not generated yet")
    metrics = load_json(config.METRICS_PATH)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_app_py_is_syntactically_valid():
    """Dashboard test (FR-16): app.py must at least parse without errors."""
    app_path = os.path.join(config.BASE_DIR, "app.py")
    with open(app_path, "r") as f:
        source = f.read()
    ast.parse(source)  # raises SyntaxError if invalid
