"""
Annotation-stage quality metrics for dataset reliability assessment.

Metrics quantify inter-annotator agreement, label bias, label noise,
annotation uncertainty, and temporal drift.
"""

import numpy as np
from scipy.stats import entropy


def inter_annotator_agreement(y1, y2) -> float:
    """
    Inter-Annotator Agreement (IAA): proportion of items on which two
    annotators agree (simple percentage agreement).
    """
    y1, y2 = np.asarray(y1), np.asarray(y2)
    if len(y1) == 0 or len(y1) != len(y2):
        return np.nan
    return float(np.mean(y1 == y2))


def group_label_bias_index(df, label_col: str, group_col: str) -> float:
    """
    Group Label Bias Index (GLBI): KL divergence between the label
    distributions of exactly two demographic groups, measuring systematic
    labelling differences.
    """
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("GLBI requires exactly two groups.")
    g1, g2 = groups
    p1 = df[df[group_col] == g1][label_col].value_counts(normalize=True)
    p2 = df[df[group_col] == g2][label_col].value_counts(normalize=True)
    p1, p2 = p1.align(p2, fill_value=0)
    return float(entropy(p1.values, p2.values))


def label_consistency_score(y1, y2) -> float:
    """
    Label Consistency Score (LCS): fraction of labels that agree across
    two annotation sources.
    """
    y1, y2 = np.asarray(y1), np.asarray(y2)
    if len(y1) == 0 or len(y1) != len(y2):
        return np.nan
    return float(np.mean(y1 == y2))


def outlier_label_rate(y_true, y_pred) -> float:
    """
    Outlier Label Rate (OLR): fraction of annotations that disagree with
    a reference (ground truth) label.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(y_true != y_pred))


def uncertainty_score_from_annotators(prob_matrix: np.ndarray,
                                       eps: float = 1e-12) -> float:
    """
    Uncertainty Score from Annotators (USA): mean normalised Shannon entropy
    of per-item label distributions derived from annotator votes.

    Each row of prob_matrix is the (confidence-weighted) probability
    distribution over labels for one item.
    """
    prob_matrix = np.asarray(prob_matrix)
    if prob_matrix.size == 0:
        return np.nan
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = prob_matrix / row_sums
    H = entropy(P + eps, axis=1)
    H_max = np.log(P.shape[1])
    return 0.0 if H_max == 0 else float(np.mean(H / H_max))


def annotation_drift_over_time(df, label_col: str, time_col: str) -> float:
    """
    Annotation Drift Over Time (ADT): KL divergence between the label
    distributions before and after the temporal median, detecting systematic
    shifts in annotation behaviour.
    """
    df = df.dropna(subset=[label_col, time_col])
    if df.empty:
        return np.nan
    t_med = df[time_col].median()
    p1 = df[df[time_col] <= t_med][label_col].value_counts(normalize=True)
    p2 = df[df[time_col] > t_med][label_col].value_counts(normalize=True)
    if p1.empty or p2.empty:
        return np.nan
    p1, p2 = p1.align(p2, fill_value=0)
    return float(entropy(p1.values, p2.values))
