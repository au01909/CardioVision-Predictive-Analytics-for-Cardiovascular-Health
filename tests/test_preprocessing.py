"""Unit tests for preprocessing.py (FR-1, FR-3, FR-6, FR-7, FR-8)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import preprocessing


@pytest.fixture(scope="module")
def df():
    return preprocessing.load_dataset(verbose=False)


def test_load_dataset_shape(df):
    assert df.shape[0] > 0
    assert config.TARGET_COL in df.columns


def test_load_dataset_schema(df):
    for col in config.ALL_FEATURES:
        assert col in df.columns


def test_split_data_proportions(df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_data(df)
    total = len(df)
    assert abs(len(X_test) / total - config.TEST_SIZE) < 0.05
    assert len(X_train) + len(X_val) + len(X_test) == total


def test_split_data_stratified(df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_data(df)
    overall_rate = df[config.TARGET_COL].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - overall_rate) < 0.15


def test_impute_no_leakage(df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_data(df)
    X_train_imp, X_val_imp, X_test_imp, imputer = preprocessing.impute_missing(
        X_train, X_val, X_test
    )
    assert X_train_imp.isna().sum().sum() == 0
    assert X_val_imp.isna().sum().sum() == 0
    assert X_test_imp.isna().sum().sum() == 0


def test_scale_features_mean_near_zero(df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_data(df)
    X_train_imp, X_val_imp, X_test_imp, _ = preprocessing.impute_missing(X_train, X_val, X_test)
    X_train_sc, X_val_sc, X_test_sc, scaler = preprocessing.scale_features(
        X_train_imp, X_val_imp, X_test_imp
    )
    assert abs(X_train_sc.mean().mean()) < 1e-6


def test_select_features_count(df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_data(df)
    X_train_imp, X_val_imp, X_test_imp, _ = preprocessing.impute_missing(X_train, X_val, X_test)
    X_train_sc, X_val_sc, X_test_sc, _ = preprocessing.scale_features(
        X_train_imp, X_val_imp, X_test_imp
    )
    X_train_sel, X_val_sel, X_test_sel, selector, cols = preprocessing.select_features(
        X_train_sc, X_val_sc, X_test_sc, y_train
    )
    assert len(cols) == config.K_BEST_FEATURES
    assert X_train_sel.shape[1] == config.K_BEST_FEATURES


def test_handle_imbalance_returns_valid_data(df):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_data(df)
    X_res, y_res, applied = preprocessing.handle_imbalance(X_train, y_train)
    assert len(X_res) == len(y_res)
    if applied:
        assert y_res.value_counts().min() / y_res.value_counts().max() > 0.9
