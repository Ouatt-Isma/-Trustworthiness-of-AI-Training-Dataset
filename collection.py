"""
Collection/measurement-stage quality metrics for dataset reliability assessment.

Metrics quantify measurement validity, missingness disparity, privacy exposure,
and sensitive-attribute leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def out_of_range_rate(df: pd.DataFrame, col: str,
                      valid_min: float, valid_max: float) -> float:
    """
    Out-of-Range Rate (ORR): fraction of values outside the valid bounds
    [valid_min, valid_max].
    """
    if col not in df.columns:
        return np.nan
    x = df[col].dropna()
    if x.empty:
        return np.nan
    return float(((x < valid_min) | (x > valid_max)).mean())


def missing_data_disparity(df: pd.DataFrame, group_col: str,
                            cols: list = None) -> float:
    """
    Missing Data Disparity (MDD): max − min missingness rate across groups,
    indicating differential data quality by subgroup.
    """
    if cols is not None:
        keep = [c for c in cols if c in df.columns and c != group_col]
        df = df[[group_col] + keep]
    rates = [
        gdf.drop(columns=[group_col]).isna().any(axis=1).mean()
        for _, gdf in df.groupby(group_col)
    ]
    return float(max(rates) - min(rates)) if rates else np.nan


def pii_density(df: pd.DataFrame, pii_columns: list) -> float:
    """
    PII Density (PIID): ratio of identified PII columns to total columns.
    """
    if df is None or df.shape[1] == 0:
        return np.nan
    return len(pii_columns) / df.shape[1]


def sensitive_attribute_leakage(df: pd.DataFrame, sensitive_col: str,
                                 feature_cols: list) -> float:
    """
    Sensitive Attribute Leakage Score (SALS): in-sample accuracy of a
    logistic regression predicting the sensitive attribute from features,
    measuring how much demographic information is implicitly encoded.
    """
    if sensitive_col not in df.columns:
        return np.nan
    X, y = df[feature_cols].values, df[sensitive_col].values
    if len(np.unique(y)) < 2:
        return np.nan
    model = LogisticRegression(max_iter=10_000, solver="lbfgs")
    model.fit(X, y)
    return float(accuracy_score(y, model.predict(X)))


def reidentification_risk(df: pd.DataFrame, quasi_identifiers: list) -> float:
    """
    Re-identification Risk Score (RRS): mean inverse anonymity set size
    E[1/|group|] over quasi-identifier combinations, following the
    k-anonymity framework.
    """
    if not quasi_identifiers:
        return np.nan
    group_sizes = df.groupby(quasi_identifiers).size()
    if group_sizes.empty:
        return np.nan
    return float(np.mean(1.0 / group_sizes))
