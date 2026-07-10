"""
CardioVision - Streamlit Dashboard
=====================================
Implements FR-16 and Section 6 (Home, About Dataset, Model Performance,
Risk Prediction, MLflow Experiments, Documentation).
"""

import os
import joblib
import json

import pandas as pd
import streamlit as st

import config
from utils import load_json

st.set_page_config(
    page_title="CardioVision",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Pre-load all artifacts into memory safely for cloud deployment
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    """Loads model and preprocessing objects from the models/ folder."""
    artifacts = {
        "model": joblib.load(config.MODEL_PATH),
        "imputer": joblib.load(config.IMPUTER_PATH),
        "scaler": joblib.load(config.SCALER_PATH),
        "selector": joblib.load(config.SELECTOR_PATH),
    }
    with open(config.THRESHOLD_PATH, "r") as f:
        artifacts["threshold"] = json.load(f)
    with open(config.FEATURE_LIST_PATH, "r") as f:
        artifacts["features"] = json.load(f)
    return artifacts

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("❤️ CardioVision")
st.sidebar.caption("Cardiovascular Disease Risk Prediction & Evaluation System")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "About Dataset", "Model Performance", "Risk Prediction",
     "MLflow Experiments", "Documentation"],
)


def artifacts_ready():
    return os.path.exists(config.MODEL_PATH) and os.path.exists(config.THRESHOLD_PATH)


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("❤️ CardioVision")
    st.subheader("Cardiovascular Disease Risk Prediction & Evaluation System")

    st.markdown(
        """
CardioVision is an end-to-end machine learning application that predicts an
individual's risk of cardiovascular disease from clinical measurements,
built using the UCI Heart Disease dataset and modern ML engineering
practices — reproducible pipelines, experiment tracking, explainable
predictions, and a deployable dashboard.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔬 Pipeline")
        st.markdown(
            "Imputation → Scaling → Feature Selection → XGBoost → "
            "Threshold Optimization → Explanation"
        )
    with col2:
        st.markdown("### 📊 Tracking")
        st.markdown("Every run logs parameters, metrics, plots, and the model to MLflow.")
    with col3:
        st.markdown("### 🖥️ Deployment")
        st.markdown("Interactive Streamlit dashboard with real-time, explainable predictions.")

    st.divider()
    st.markdown("### Project Architecture")
    st.code(
        """
Dataset (heart.csv)
      │
      ▼
Preprocessing  (impute → scale → select-k-best → SMOTE if imbalanced)
      │
      ▼
Model Training (XGBoost + RandomizedSearchCV, selected by ROC AUC)
      │
      ▼
Threshold Optimization (Youden Index)
      │
      ▼
Evaluation (Accuracy, ROC AUC, F1, Calibration, ...)
      │
      ▼
MLflow Tracking  +  Model Persistence (models/*.pkl)
      │
      ▼
Streamlit Dashboard  (this app)
        """,
        language="text",
    )

    if artifacts_ready():
        metrics = load_json(config.METRICS_PATH) if os.path.exists(config.METRICS_PATH) else {}
        st.divider()
        st.markdown("### Current Model Snapshot")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
        m2.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
        m3.metric("F1 Score", f"{metrics.get('f1_score', 0) * 100:.1f}%")
        m4.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0) * 100:.1f}%")
    else:
        st.warning("No trained model found yet. Run `python train.py` first.")

# ---------------------------------------------------------------------------
# ABOUT DATASET
# ---------------------------------------------------------------------------
elif page == "About Dataset":
    st.title("📁 About the Dataset")
    st.markdown(
        "**Source:** UCI Machine Learning Repository — Heart Disease Dataset "
        "(Cleveland database, 14 attributes, 303 patient records)."
    )

    if os.path.exists(config.DATA_PATH):
        df = pd.read_csv(config.DATA_PATH)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        st.markdown("### Feature Dictionary")
        rows = []
        for feat, meta in config.FEATURE_EXPLANATIONS.items():
            rows.append({
                "Dataset Column": feat,
                "Display Name": meta["display_name"],
                "Medical Name": meta["medical_name"],
                "Layman Explanation": meta["explanation"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Sample Records")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Exploratory Data Analysis")
        tabs = st.tabs(["Target Distribution", "Correlation Matrix", "Feature Histograms",
                         "Boxplots (Outliers)", "Missing Values"])
        paths = [config.TARGET_DIST_PATH, config.CORRELATION_MATRIX_PATH,
                 config.FEATURE_HIST_PATH, config.BOXPLOT_PATH, config.MISSING_HEATMAP_PATH]
        for tab, path in zip(tabs, paths):
            with tab:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.info("Run `python train.py` or `python preprocessing.py` to generate this plot.")
    else:
        st.error(f"Dataset not found at {config.DATA_PATH}")

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("📈 Model Performance")

    if not artifacts_ready():
        st.warning("No trained model found. Run `python train.py` first.")
    else:
        metrics = load_json(config.METRICS_PATH) if os.path.exists(config.METRICS_PATH) else {}
        threshold_info = load_json(config.THRESHOLD_PATH)

        st.markdown("### Test Set Metrics")
        cols = st.columns(5)
        keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for c, k in zip(cols, keys):
            val = metrics.get(k, 0)
            c.metric(k.replace("_", " ").title(), f"{val * 100:.1f}%" if val <= 1 else f"{val:.3f}")

        cols2 = st.columns(5)
        keys2 = ["balanced_accuracy", "sensitivity", "specificity",
                  "false_positive_rate", "false_negative_rate"]
        for c, k in zip(cols2, keys2):
            val = metrics.get(k, 0)
            c.metric(k.replace("_", " ").title(), f"{val * 100:.1f}%")

        st.info(f"Decision threshold in use: **{threshold_info['threshold']:.4f}** "
                f"(optimized via {threshold_info['method']})")

        st.markdown("### Diagnostic Plots")
        tabs = st.tabs(["ROC Curve", "Precision-Recall", "Confusion Matrix",
                         "Calibration Curve", "Feature Importance"])
        paths = [config.ROC_CURVE_PATH, config.PR_CURVE_PATH, config.CONFUSION_MATRIX_PATH,
                 config.CALIBRATION_CURVE_PATH, config.FEATURE_IMPORTANCE_PATH]
        for tab, path in zip(tabs, paths):
            with tab:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)

        if os.path.exists(config.CLASSIFICATION_REPORT_PATH):
            st.markdown("### Classification Report")
            report = load_json(config.CLASSIFICATION_REPORT_PATH)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# ---------------------------------------------------------------------------
# RISK PREDICTION
# ---------------------------------------------------------------------------
elif page == "Risk Prediction":
    st.title("🩺 Cardiovascular Risk Prediction")

    if not artifacts_ready():
        st.warning("No trained model found. Run `python train.py` first.")
    else:
        st.markdown("Enter patient information below to estimate cardiovascular disease risk.")

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=1, max_value=120, value=54)
                sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)],
                                    format_func=lambda x: x[0])[1]
                cp = st.selectbox(
                    "Chest Pain Type",
                    options=[("Typical Angina", 0), ("Atypical Angina", 1),
                             ("Non-Anginal Pain", 2), ("Asymptomatic", 3)],
                    format_func=lambda x: x[0])[1]
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                    options=[("No", 0), ("Yes", 1)],
                                    format_func=lambda x: x[0])[1]
            with c2:
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)",
                                            min_value=60, max_value=250, value=130)
                chol = st.number_input("Cholesterol (mg/dl)",
                                        min_value=100, max_value=600, value=246)
                restecg = st.selectbox(
                    "Resting ECG",
                    options=[("Normal", 0), ("ST-T Abnormality", 1),
                             ("Left Ventricular Hypertrophy", 2)],
                    format_func=lambda x: x[0])[1]
                thalach = st.number_input("Maximum Heart Rate Achieved",
                                           min_value=60, max_value=220, value=150)
            with c3:
                exang = st.selectbox("Exercise Induced Angina",
                                      options=[("No", 0), ("Yes", 1)],
                                      format_func=lambda x: x[0])[1]
                oldpeak = st.number_input("ST Depression (oldpeak)",
                                           min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.selectbox(
                    "ST Segment Slope",
                    options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                    format_func=lambda x: x[0])[1]
                ca = st.selectbox("Major Vessels Colored (0-3)", options=[0, 1, 2, 3])
                thal = st.selectbox(
                    "Thalassemia",
                    options=[("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)],
                    format_func=lambda x: x[0])[1]

            submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

        if submitted:
            import predict as predict_module
            record = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            }
            result = predict_module.predict_single(record)

            st.divider()
            risk_colors = {
                "High Risk": "🔴", "Moderate-High Risk": "🟠",
                "Moderate Risk": "🟡", "Low Risk": "🟢",
            }
            emoji = risk_colors.get(result["risk_category"], "⚪")

            r1, r2, r3 = st.columns(3)
            r1.metric("Prediction", result["prediction_label"])
            r2.metric("Probability", result["probability_pct"])
            r3.metric("Risk Category", f"{emoji} {result['risk_category']}")

            st.markdown(f"**Recommendation:** {result['recommendation']}")

            st.markdown("### Main Contributing Factors")
            for factor in result["top_factors"]:
                st.markdown(f"- **{factor['display_name']}** — {factor['explanation']}")

            st.caption(
                "⚠️ This tool is for educational and research purposes only and is "
                "not a substitute for professional medical diagnosis."
            )

# ---------------------------------------------------------------------------
# MLFLOW EXPERIMENTS
# ---------------------------------------------------------------------------
elif page == "MLflow Experiments":
    st.title("🧪 MLflow Experiment Tracking")

    st.markdown(
        "All training runs are tracked locally with MLflow (parameters, metrics, plots, "
        "the serialized model, dataset version, and git commit hash)."
    )
    st.code(f"mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}", language="bash")
    st.caption("Run the command above on your local machine, then open http://localhost:5000 to browse runs.")
    
    st.info("🚀 **Note for Streamlit Cloud Users:** MLflow tracking is designed for local development and is not connected to this live cloud dashboard to keep the app lightweight and fast. The dashboard loads the pre-trained model artifacts directly.")

# ---------------------------------------------------------------------------
# DOCUMENTATION
# ---------------------------------------------------------------------------
elif page == "Documentation":
    st.title("📚 Documentation")
    readme_path = os.path.join(config.BASE_DIR, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            st.markdown(f.read())
    else:
        st.info("README.md not found.")