# ❤️ CardioVision — Cardiovascular Disease Risk Prediction & Evaluation System

An end-to-end machine learning application that predicts an individual's risk of
cardiovascular disease using the **UCI Heart Disease dataset**, built with modern
ML engineering practices: reproducible pipelines, experiment tracking, explainable
predictions, automated evaluation, and a deployable dashboard.

---

## 1. Project Architecture

```
Dataset (heart.csv, UCI Heart Disease / Cleveland database)
      │
      ▼
Preprocessing   (median imputation → standard scaling → SelectKBest → SMOTE if imbalanced)
      │
      ▼
Model Training  (XGBoost + RandomizedSearchCV over 60 candidate configs, 5-fold CV, selected by ROC AUC)
      │
      ▼
Threshold Optimization  (Youden Index: maximize TPR − FPR on the validation split)
      │
      ▼
Evaluation  (Accuracy, ROC AUC, F1, Sensitivity/Specificity, Calibration, ...)
      │
      ▼
MLflow Tracking  +  Model Persistence (models/*.pkl, models/threshold.json)
      │
      ▼
Streamlit Dashboard  (real-time, explainable predictions)
```

Every transformer (imputer, scaler, selector) is **fit only on the training
split** — validation and test data never leak into preprocessing statistics.

---

## 2. Results

Trained on the 303-row UCI Heart Disease (Cleveland) dataset with an
80/20 train+val / test split, and an inner 80/20 train/val split
(64% train / 16% val / 20% test overall):

| Metric              | Test Set Result |
| -------------------- | ---------------- |
| Accuracy              | 80.3%            |
| Precision             | 74.4%            |
| Recall (Sensitivity)  | 97.0%            |
| F1 Score              | 84.2%            |
| ROC AUC               | 0.908            |
| Balanced Accuracy     | 78.8%            |
| Specificity           | 60.7%            |
| Optimal Threshold     | 0.418 (Youden Index) |

*Reproduce these numbers with `python train.py` — figures are written to
`artifacts/metrics.json`.*

> **Note on the PRD's >95% targets:** the PRD's success-criteria table set an
> aspirational bar of >95% across accuracy/precision/recall/F1 and >0.97 ROC
> AUC. On this real, publicly available 303-row Cleveland dataset — with a
> leakage-free pipeline (no test-set peeking, transformers fit only on
> train) — those numbers are not realistically achievable; published
> benchmarks on this exact dataset typically fall in the 80–90% accuracy /
> 0.88–0.93 ROC AUC range, which is what CardioVision achieves here. Numbers
> above that on this dataset are a strong signal of information leakage or
> test-set overlap with training data, which this project deliberately
> avoids. The resume bullet in Section 15 of the PRD should be read as a
> template to adapt to your own actual results rather than a literal claim.

---

## 3. Dataset

**Source:** UCI Machine Learning Repository — [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Cleveland database), 303 patient records, 13 predictive features + target.

| Feature  | Description              |
| -------- | ------------------------- |
| age      | Patient age                |
| sex      | Biological sex              |
| cp       | Chest pain type              |
| trestbps | Resting blood pressure        |
| chol     | Serum cholesterol               |
| fbs      | Fasting blood sugar > 120 mg/dl  |
| restecg  | Resting ECG results                |
| thalach  | Maximum heart rate achieved          |
| exang    | Exercise-induced angina                |
| oldpeak  | ST depression induced by exercise        |
| slope    | Slope of peak exercise ST segment          |
| ca       | Number of major vessels colored by fluoroscopy |
| thal     | Thalassemia test result                          |
| target   | Heart disease presence (0/1)                       |

You can also fetch it programmatically via `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets
```

A ready-to-use CSV is bundled at `data/heart.csv`.

---

## 4. Installation

```bash
git clone https://github.com/<your-username>/CardioVision.git
cd CardioVision
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires **Python 3.11+**.

---

## 5. Usage

### Train the model (preprocessing → training → threshold → evaluation → MLflow logging)

```bash
python train.py
```

This regenerates everything under `models/` and `artifacts/`, and logs a full
run to the local MLflow store (`mlflow.db`).

### Evaluate an already-trained model

```bash
python evaluate.py
```

### Run predictions from the command line

```bash
python predict.py
```

### Launch the interactive dashboard

```bash
streamlit run app.py
```

### Browse MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://localhost:5000.

### Run the test suite

```bash
pytest tests/ -v
```

---

## 6. Streamlit Dashboard

| Page | Purpose |
| ---- | ------- |
| **Home** | Project overview, architecture diagram, live model snapshot |
| **About Dataset** | Feature dictionary (technical / medical / layman names), sample records, EDA plots |
| **Model Performance** | Full metrics suite, ROC/PR/confusion/calibration plots, classification report |
| **Risk Prediction** | Interactive form → real-time prediction, probability, risk category, top contributing factors |
| **MLflow Experiments** | Browse logged runs, parameters, and metrics directly in-app |
| **Documentation** | Renders this README |

---

## 7. Machine Learning Pipeline Detail

1. **Dataset Loading** — schema/dtype validation, summary stats (`preprocessing.load_dataset`)
2. **EDA** — missing-value heatmap, correlation matrix, target distribution, histograms, boxplots
3. **Missing Value Handling** — median `SimpleImputer`, fit on train only, persisted as `imputer.pkl`
4. **Data Splitting** — stratified 64% train / 16% val / 20% test, `random_state=42`
5. **Feature Scaling** — `StandardScaler`, fit on train only, persisted as `scaler.pkl`
6. **Feature Selection** — `SelectKBest(f_classif, k=10)`, ranks all 13 features, persisted as `selector.pkl`
7. **Class Imbalance** — SMOTE applied to the training split only if the majority/minority ratio exceeds 1.5
8. **Model Training** — `XGBClassifier` tuned via `RandomizedSearchCV` (60 candidates, 5-fold CV, scored on ROC AUC)
9. **Threshold Optimization** — Youden Index (`argmax(TPR − FPR)`) on the validation split
10. **Evaluation** — accuracy, precision, recall, F1, ROC AUC, balanced accuracy, sensitivity, specificity, FPR, FNR, ROC/PR/calibration curves, confusion matrix
11. **Explainability** — per-prediction top-factor ranking using model feature importances weighted by input magnitude, mapped to layman explanations
12. **MLflow Tracking** — parameters, metrics, artifacts, model, git commit hash, Python version, dataset version logged on every run
13. **Model Persistence** — `model.pkl`, `imputer.pkl`, `scaler.pkl`, `selector.pkl`, `threshold.json`
14. **Inference Pipeline** — `predict.py`: validate → impute → scale → select → predict → threshold → risk category → explanation

---

## 8. Folder Structure

```
CardioVision/
├── app.py                 # Streamlit dashboard
├── train.py                # Training + MLflow logging entrypoint
├── preprocessing.py         # Loading, EDA, imputation, splitting, scaling, selection, SMOTE
├── evaluate.py               # Metrics + diagnostic plots
├── predict.py                  # Reusable inference pipeline
├── config.py                    # Central paths, constants, hyperparameter grids
├── utils.py                       # Shared helpers (I/O, git hash, risk banding)
├── models/                          # Persisted model + preprocessing artifacts
├── artifacts/                        # EDA & evaluation plots, feature scores, metrics
├── data/heart.csv                     # UCI Heart Disease dataset
├── notebooks/EDA.ipynb                  # Exploratory analysis notebook
├── tests/                                 # Unit tests (pytest)
├── .github/workflows/ci.yml                # CI/CD pipeline
├── requirements.txt
└── README.md
```

---

## 9. Deployment

- **Application:** deploy `app.py` on [Streamlit Community Cloud](https://streamlit.io/cloud)
  (or any Streamlit-compatible host) by pointing it at this repository.
  GitHub Pages cannot run a Python/Streamlit app — it only serves static files.
- **Documentation site:** the CI pipeline publishes static docs (README +
  evaluation artifacts) to GitHub Pages on every push to `main`.

---

## 10. CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

```
Push → Install deps → Lint → Unit tests (pre-training) → Train model
     → Evaluate → Upload artifacts → Full test suite → Build & deploy docs
```

---

## 11. Testing

`tests/` covers:
- **Preprocessing** — schema validation, split proportions/stratification, leakage-free imputation & scaling, feature-selection count, SMOTE behavior
- **Prediction pipeline** — input validation, output schema, single & batch inference
- **Model loading & persistence** — all artifacts present and well-formed
- **Dashboard** — `app.py` syntax/import sanity check

Run with:
```bash
python train.py     # generate artifacts the tests depend on
pytest tests/ -v
```

---

## 12. Future Scope

- Add SHAP-based explanations for richer per-prediction attributions
- Model registry + staged promotion (Staging → Production) via MLflow Model Registry
- Add calibration-aware probability recalibration (Platt scaling / isotonic regression)
- Expand to multi-class severity prediction (UCI's original 0–4 `num` target)
- Containerize with Docker for one-command deployment

---

## 13. License

This project uses the UCI Heart Disease dataset, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Project code is
released under the MIT License — see `LICENSE`.

---

## 14. Disclaimer

CardioVision is a research/educational project. It is **not** a certified
medical device and must not be used for actual clinical diagnosis or
treatment decisions. Always consult a qualified healthcare professional.
