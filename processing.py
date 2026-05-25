"""
Processing-stage quality metrics for dataset reliability assessment.

Metrics quantify data integrity, feature–label leakage, and the impact
of cleaning operations.
"""

import numpy as np
import pandas as pd


def duplicate_instance_overlap_rate(df: pd.DataFrame) -> float:
    """
    Duplicate Instance Overlap Rate (DIOR): fraction of rows that are
    exact duplicates of at least one other row.
    """
    if df is None or len(df) == 0:
        return np.nan
    return float(df.duplicated().mean())


def feature_label_correlation(df: pd.DataFrame, feature_cols: list,
                               label_col: str) -> float:
    """
    Feature–Label Correlation Audit (FLCA): maximum absolute Pearson
    correlation between any numeric feature and the label, indicating
    potential shortcut learning or label leakage.
    """
    if df is None or df.empty:
        return np.nan
    corrs = []
    for col in feature_cols:
        if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
            continue
        valid = df[[col, label_col]].dropna()
        if len(valid) < 2:
            continue
        c = valid[col].corr(valid[label_col])
        if not np.isnan(c):
            corrs.append(abs(c))
    return float(max(corrs)) if corrs else np.nan


def pre_post_cleaning_change_ratio(n_pre: int, n_post: int) -> float:
    """
    Pre/Post Cleaning Change Ratio (PCCR): fraction of instances removed
    during cleaning, i.e., 1 − n_post / n_pre.
    """
    if n_pre <= 0:
        return np.nan
    return 1.0 - n_post / n_pre


def feature_loss_rate(n_features_original: int, n_features_kept: int) -> float:
    """
    Feature Loss Rate (FLR): fraction of features dropped during processing.
    """
    if n_features_original <= 0:
        return np.nan
    return 1.0 - n_features_kept / n_features_original
