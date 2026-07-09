"""
CardioVision - Evaluation
============================
Implements FR-11 (Evaluation metrics + plots).
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

import config
from utils import save_json

warnings.filterwarnings("ignore")


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }
    return metrics


def plot_roc_curve(y_true, y_prob, path=config.ROC_CURVE_PATH):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2E86AB", lw=2, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, path=config.PR_CURVE_PATH):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#A23B72", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path=config.CONFUSION_MATRIX_PATH):
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_calibration_curve(y_true, y_prob, path=config.CALIBRATION_CURVE_PATH):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", color="#F18F01", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def evaluate_model(model, X_test, y_test, threshold, log_to_mlflow=False):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)

    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_calibration_curve(y_test, y_prob)

    report = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"],
                                    output_dict=True, zero_division=0)
    save_json(report, config.CLASSIFICATION_REPORT_PATH)
    save_json(metrics, config.METRICS_PATH)

    if log_to_mlflow:
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(config.ROC_CURVE_PATH)
        mlflow.log_artifact(config.PR_CURVE_PATH)
        mlflow.log_artifact(config.CONFUSION_MATRIX_PATH)
        mlflow.log_artifact(config.CALIBRATION_CURVE_PATH)
        mlflow.log_artifact(config.CLASSIFICATION_REPORT_PATH)
        mlflow.log_artifact(config.METRICS_PATH)

    return metrics


if __name__ == "__main__":
    from utils import load_json, load_pickle
    import preprocessing

    model = load_pickle(config.MODEL_PATH)
    threshold_info = load_json(config.THRESHOLD_PATH)
    threshold = threshold_info["threshold"]

    df = preprocessing.load_dataset(verbose=False)
    splits, _ = preprocessing.run_preprocessing_pipeline(df, run_eda_flag=False)
    metrics = evaluate_model(model, splits["X_test"], splits["y_test"], threshold)

    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")
