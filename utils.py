"""
CardioVision - Shared Utilities
=================================
Small reusable helpers used across train.py, evaluate.py, predict.py and app.py.
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    return joblib.load(path)


def get_git_commit_hash():
    """Return the short git commit hash, or 'unversioned' if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unversioned"


def get_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def risk_category(probability, threshold):
    """Translate a probability + decision threshold into a human risk band."""
    if probability >= max(threshold, 0.66):
        return "High Risk"
    elif probability >= threshold:
        return "Moderate-High Risk"
    elif probability >= threshold * 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"


def recommendation_for(category):
    mapping = {
        "High Risk": "Please consult a cardiologist promptly for a comprehensive cardiac evaluation.",
        "Moderate-High Risk": "Consider scheduling a check-up with a physician and monitoring key risk factors closely.",
        "Moderate Risk": "Maintain a heart-healthy lifestyle and consider a routine check-up.",
        "Low Risk": "No immediate concern indicated. Continue regular health maintenance.",
    }
    return mapping.get(category, "Consult a healthcare professional for personalized advice.")
