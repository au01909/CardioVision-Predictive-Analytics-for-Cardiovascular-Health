"""
CardioVision - Preprocessing
==============================
Implements FR-1 (Dataset Loading), FR-2 (EDA), FR-3 (Missing Value Handling),
FR-4 (Feature Engineering / Selection), FR-6 (Data Splitting), FR-7 (Scaling),
and FR-8 (Class Imbalance / SMOTE).

Design principle: NO information leakage. Every fitted transformer
(imputer, scaler, selector) is fit ONLY on the training split.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# FR-1 Dataset Loading
# ---------------------------------------------------------------------------
def load_dataset(path=config.DATA_PATH, verbose=True):
    """Load the CSV, validate schema/dtypes, and print a summary."""
    df = pd.read_csv(path)

    expected_cols = set(config.ALL_FEATURES + [config.TARGET_COL])
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    # Coerce everything to numeric (dataset is numerically encoded)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if verbose:
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Rows            : {df.shape[0]}")
        print(f"Columns         : {df.shape[1]}")
        print(f"Missing Values  : {int(df.isna().sum().sum())}")
        print("\nData Types:")
        print(df.dtypes)
        print("\nTarget Distribution:")
        print(df[config.TARGET_COL].value_counts())
        print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# FR-2 Exploratory Data Analysis
# ---------------------------------------------------------------------------
def run_eda(df, out_dir=config.ARTIFACTS_DIR):
    """Generate the full suite of EDA plots required by FR-2."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Missing value heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.savefig(config.MISSING_HEATMAP_PATH, dpi=120)
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(config.CORRELATION_MATRIX_PATH, dpi=120)
    plt.close()

    # Target distribution
    plt.figure(figsize=(6, 5))
    sns.countplot(x=config.TARGET_COL, data=df, hue=config.TARGET_COL,
                   palette="Set2", legend=False)
    plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
    plt.tight_layout()
    plt.savefig(config.TARGET_DIST_PATH, dpi=120)
    plt.close()

    # Feature histograms
    num_cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flatten(), num_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(config.FEATURE_HIST_PATH, dpi=120)
    plt.close()

    # Boxplots (outlier detection)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.flatten(), num_cols):
        sns.boxplot(y=df[col], ax=ax, color="salmon")
        ax.set_title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.savefig(config.BOXPLOT_PATH, dpi=120)
    plt.close()

    print(f"EDA artifacts saved to {out_dir}")


# ---------------------------------------------------------------------------
# FR-6 Data Splitting: Train 64% / Val 16% / Test 20%
# ---------------------------------------------------------------------------
def split_data(df):
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET_COL].copy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_trainval,
    )
    print(f"Train: {X_train.shape[0]} rows | Val: {X_val.shape[0]} rows | "
          f"Test: {X_test.shape[0]} rows")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# FR-3 Missing Value Handling (fit ONLY on training data)
# ---------------------------------------------------------------------------
def impute_missing(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_imp, X_val_imp, X_test_imp, imputer


# ---------------------------------------------------------------------------
# FR-7 Data Scaling (fit ONLY on training data)
# ---------------------------------------------------------------------------
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_sc = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_sc, X_val_sc, X_test_sc, scaler


# ---------------------------------------------------------------------------
# FR-4 Feature Engineering / Selection (fit ONLY on training data)
# ---------------------------------------------------------------------------
def select_features(X_train, X_val, X_test, y_train, k=config.K_BEST_FEATURES):
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    mask = selector.get_support()
    selected_cols = X_train.columns[mask].tolist()

    scores_df = pd.DataFrame({
        "feature": X_train.columns,
        "f_score": selector.scores_,
        "p_value": selector.pvalues_,
        "selected": mask,
    }).sort_values("f_score", ascending=False)
    scores_df.to_csv(config.FEATURE_SCORES_PATH, index=False)

    # Feature importance-style bar plot from F-scores
    plt.figure(figsize=(9, 6))
    plot_df = scores_df.sort_values("f_score", ascending=True)
    colors = ["#2E86AB" if s else "#CBD5E1" for s in plot_df["selected"]]
    plt.barh(plot_df["feature"], plot_df["f_score"], color=colors)
    plt.xlabel("ANOVA F-score")
    plt.title("Feature Ranking (SelectKBest, f_classif)")
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_PATH, dpi=120)
    plt.close()

    X_train_sel = X_train[selected_cols]
    X_val_sel = X_val[selected_cols]
    X_test_sel = X_test[selected_cols]

    print(f"Selected {len(selected_cols)} features: {selected_cols}")
    return X_train_sel, X_val_sel, X_test_sel, selector, selected_cols


# ---------------------------------------------------------------------------
# FR-8 Class Imbalance (SMOTE on TRAIN only)
# ---------------------------------------------------------------------------
def handle_imbalance(X_train, y_train, threshold=config.IMBALANCE_RATIO_THRESHOLD):
    counts = y_train.value_counts()
    ratio = counts.max() / counts.min()
    print(f"Train class ratio (majority/minority): {ratio:.2f}")

    if ratio > threshold:
        print("Imbalance threshold exceeded -> applying SMOTE on training data only.")
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res, True
    else:
        print("Class balance acceptable -> SMOTE not applied.")
        return X_train, y_train, False


# ---------------------------------------------------------------------------
# Full pipeline orchestration (used by train.py)
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(df=None, run_eda_flag=True):
    if df is None:
        df = load_dataset()

    if run_eda_flag:
        run_eda(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train, X_val, X_test, imputer = impute_missing(X_train, X_val, X_test)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    X_train, X_val, X_test, selector, selected_cols = select_features(
        X_train, X_val, X_test, y_train
    )
    X_train, y_train, smote_applied = handle_imbalance(X_train, y_train)

    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector,
        "selected_features": selected_cols,
        "smote_applied": smote_applied,
    }
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
    return splits, artifacts


if __name__ == "__main__":
    df = load_dataset()
    run_eda(df)
    splits, artifacts = run_preprocessing_pipeline(df, run_eda_flag=False)
    print("Preprocessing complete.")
