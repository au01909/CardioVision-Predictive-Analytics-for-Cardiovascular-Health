"""
CardioVision - Central Configuration
=====================================
All paths, constants, and hyperparameter search spaces live here so that
train.py, evaluate.py, predict.py, and app.py stay in sync.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

DATA_PATH = os.path.join(DATA_DIR, "heart.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
SELECTOR_PATH = os.path.join(MODELS_DIR, "selector.pkl")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "selected_features.json")

ROC_CURVE_PATH = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
PR_CURVE_PATH = os.path.join(ARTIFACTS_DIR, "pr_curve.png")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
CALIBRATION_CURVE_PATH = os.path.join(ARTIFACTS_DIR, "calibration_curve.png")
FEATURE_SCORES_PATH = os.path.join(ARTIFACTS_DIR, "feature_scores.csv")
MISSING_HEATMAP_PATH = os.path.join(ARTIFACTS_DIR, "missing_value_heatmap.png")
CORRELATION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "correlation_matrix.png")
TARGET_DIST_PATH = os.path.join(ARTIFACTS_DIR, "target_distribution.png")
FEATURE_HIST_PATH = os.path.join(ARTIFACTS_DIR, "feature_histograms.png")
BOXPLOT_PATH = os.path.join(ARTIFACTS_DIR, "boxplots.png")
CLASSIFICATION_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "classification_report.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
TARGET_COL = "target"

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Number of top features to keep with SelectKBest
K_BEST_FEATURES = 10

# ---------------------------------------------------------------------------
# Splitting / reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20        # held out from the full dataset
VAL_SIZE = 0.20         # held out from the remaining "train+val" pool (=> 16% of full)

# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------
IMBALANCE_RATIO_THRESHOLD = 1.5  # majority/minority ratio above which SMOTE kicks in

# ---------------------------------------------------------------------------
# Hyperparameter search space (RandomizedSearchCV)
# ---------------------------------------------------------------------------
XGB_PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    "max_depth": [2, 3, 4, 5, 6, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3, 0.5, 1],
    "min_child_weight": [1, 2, 3, 4, 5],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "n_estimators": [100, 150, 200, 300, 400, 500],
}
N_SEARCH_ITER = 60
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "CardioVision"
# SQLite-backed local tracking store (MLflow's filesystem store is in
# maintenance mode as of MLflow 3.x, so a local DB file is used instead).
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

# ---------------------------------------------------------------------------
# Feature explanations (technical / medical / layman) - used by FR-5 & FR-12
# ---------------------------------------------------------------------------
FEATURE_EXPLANATIONS = {
    "age": {
        "display_name": "Age",
        "medical_name": "Patient Age (years)",
        "explanation": "The patient's age in years. Cardiovascular risk generally rises with age.",
    },
    "sex": {
        "display_name": "Sex",
        "medical_name": "Biological Sex",
        "explanation": "Biological sex of the patient (1 = male, 0 = female). Men tend to have higher risk at younger ages.",
    },
    "cp": {
        "display_name": "Chest Pain Type",
        "medical_name": "Chest Pain Classification",
        "explanation": "Type of chest pain experienced: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic.",
    },
    "trestbps": {
        "display_name": "Resting Blood Pressure",
        "medical_name": "Resting Systolic Blood Pressure (mm Hg)",
        "explanation": "Blood pressure measured while at rest. High values indicate hypertension, a major risk factor.",
    },
    "chol": {
        "display_name": "Cholesterol",
        "medical_name": "Serum Cholesterol (mg/dl)",
        "explanation": "Amount of cholesterol in the blood. Elevated levels can clog arteries over time.",
    },
    "fbs": {
        "display_name": "Fasting Blood Sugar",
        "medical_name": "Fasting Blood Sugar > 120 mg/dl",
        "explanation": "Whether fasting blood sugar exceeds 120 mg/dl (1 = true, 0 = false), a diabetes indicator.",
    },
    "restecg": {
        "display_name": "Resting ECG",
        "medical_name": "Resting Electrocardiographic Results",
        "explanation": "Results of an ECG taken at rest: 0 = normal, 1 = ST-T abnormality, 2 = probable/definite left ventricular hypertrophy.",
    },
    "thalach": {
        "display_name": "Maximum Heart Rate",
        "medical_name": "Maximum Heart Rate Achieved (bpm)",
        "explanation": "The highest heart rate reached during a stress test. Lower peak rates can signal poor cardiac fitness.",
    },
    "exang": {
        "display_name": "Exercise Induced Angina",
        "medical_name": "Exercise-Induced Angina",
        "explanation": "Whether physical exertion triggers chest pain (1 = yes, 0 = no), suggesting restricted blood flow.",
    },
    "oldpeak": {
        "display_name": "ST Depression",
        "medical_name": "ST Depression Induced by Exercise Relative to Rest",
        "explanation": "A measurement from the ECG showing how much the heart's electrical signal dips during exercise; larger dips suggest ischemia.",
    },
    "slope": {
        "display_name": "ST Segment Slope",
        "medical_name": "Slope of the Peak Exercise ST Segment",
        "explanation": "The slope pattern of the ST segment during peak exercise: 0 = upsloping, 1 = flat, 2 = downsloping.",
    },
    "ca": {
        "display_name": "Major Vessels Colored",
        "medical_name": "Number of Major Vessels (0-3) Colored by Fluoroscopy",
        "explanation": "Count of major blood vessels visible via fluoroscopy; more visible vessels can indicate more significant blockage detection.",
    },
    "thal": {
        "display_name": "Thalassemia",
        "medical_name": "Thalassemia Blood Disorder Test Result",
        "explanation": "Result of a thallium stress test: 1 = normal, 2 = fixed defect, 3 = reversible defect.",
    },
}
