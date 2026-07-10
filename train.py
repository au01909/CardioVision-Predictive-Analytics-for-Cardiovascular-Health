"""
CardioVision - Training Script
=================================
Implements FR-9 (Model Training), FR-10 (Threshold Optimization),
FR-13 (MLflow Tracking), and FR-14 (Model Persistence).
"""

import time

import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

import config
import preprocessing
from utils import get_git_commit_hash, get_python_version, save_json, save_pickle


# ---------------------------------------------------------------------------
# FR-9 Model Training (XGBoost + RandomizedSearchCV, selected by ROC AUC)
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    base_model = XGBClassifier(
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                        random_state=config.RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=config.XGB_PARAM_DISTRIBUTIONS,
        n_iter=config.N_SEARCH_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    search.fit(X_train, y_train)
    training_time = time.time() - start

    print(f"Best CV ROC AUC: {search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}")
    print(f"Training Time: {training_time:.2f}s")

    return search.best_estimator_, search.best_params_, search.best_score_, training_time


# ---------------------------------------------------------------------------
# FR-10 Threshold Optimization (Youden Index = TPR - FPR)
# ---------------------------------------------------------------------------
def optimize_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, probs)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_threshold = float(thresholds[best_idx])

    print(f"Optimal Threshold (Youden Index): {best_threshold:.4f} "
          f"(TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")

    return best_threshold


# ---------------------------------------------------------------------------
# Main training pipeline with full MLflow logging
# ---------------------------------------------------------------------------
def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_cardiovision"):
        df = preprocessing.load_dataset()
        preprocessing.run_eda(df)

        splits, artifacts = preprocessing.run_preprocessing_pipeline(df, run_eda_flag=False)

        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        # ---- Train ----
        model, best_params, cv_auc, training_time = train_model(X_train, y_train)

        # ---- Threshold ----
        best_threshold = optimize_threshold(model, X_val, y_val)
        save_json({"threshold": best_threshold, "method": "youden_index"},
                config.THRESHOLD_PATH)

        # ---- Persist artifacts (FR-14) ----
        save_pickle(model, config.MODEL_PATH)
        save_pickle(artifacts["imputer"], config.IMPUTER_PATH)
        save_pickle(artifacts["scaler"], config.SCALER_PATH)
        save_pickle(artifacts["selector"], config.SELECTOR_PATH)
        save_json(artifacts["selected_features"], config.FEATURE_LIST_PATH)

        # ---- MLflow logging (FR-13) ----
        mlflow.log_params(best_params)
        mlflow.log_param("k_best_features", config.K_BEST_FEATURES)
        mlflow.log_param("smote_applied", artifacts["smote_applied"])
        mlflow.log_param("cv_folds", config.CV_FOLDS)
        mlflow.log_param("n_search_iter", config.N_SEARCH_ITER)

        mlflow.log_metric("cv_roc_auc", cv_auc)
        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("training_time_seconds", training_time)

        mlflow.log_param("git_commit_hash", get_git_commit_hash())
        mlflow.log_param("python_version", get_python_version())
        mlflow.log_param("dataset_version", "uci-heart-disease-v1")

        mlflow.log_artifact(config.FEATURE_SCORES_PATH)
        mlflow.log_artifact(config.FEATURE_IMPORTANCE_PATH)
        mlflow.log_artifact(config.THRESHOLD_PATH)

        mlflow.xgboost.log_model(model, name="model")

        # ---- Evaluate on the held-out test set ----
        import evaluate
        test_metrics = evaluate.evaluate_model(
            model, X_test, y_test, best_threshold, log_to_mlflow=True
        )

        print("\n" + "=" * 60)
        print("FINAL TEST METRICS")
        print("=" * 60)
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"{k:25s}: {v:.4f}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")

    return model, test_metrics


if __name__ == "__main__":
    main()
