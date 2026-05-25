"""
Sampling-stage quality metrics for dataset reliability assessment.

Metrics quantify representativeness, temporal coverage, and completeness
of the data-collection process.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def distribution_gap(df: pd.DataFrame, group_col: str,
                     population_dist: dict) -> float:
    """
    Distribution Gap (DG): KL divergence between the dataset group
    distribution and a reference population distribution.
    """
    data_dist = df[group_col].value_counts(normalize=True)
    pop_dist = pd.Series(population_dist)
    data_dist, pop_dist = data_dist.align(pop_dist, fill_value=0)
    return entropy(data_dist.values + 1e-12, pop_dist.values + 1e-12)


def minority_coverage_rate(df: pd.DataFrame, group_col: str,
                           minority: str, population_dist: dict) -> float:
    """
    Minority Coverage Rate (MCR): ratio of the minority group's dataset
    frequency to its reference population frequency.
    """
    p_pop = population_dist.get(minority, 0)
    if p_pop == 0:
        return np.nan
    return (df[group_col] == minority).mean() / p_pop


def group_representation_ratio(df: pd.DataFrame, group_col: str,
                                group: str, population_dist: dict) -> float:
    """
    Group Representation Ratio (GRR): P_dataset(g) / P_population(g).
    """
    p_pop = population_dist.get(group, 0)
    if p_pop == 0:
        return np.nan
    return (df[group_col] == group).mean() / p_pop


def long_tail_coverage_index(df: pd.DataFrame, category_col: str,
                              long_tail_categories: set) -> float:
    """
    Long-Tail Coverage Index (LTCI): fraction of rare reference categories
    that appear at least once in the dataset.
    """
    if not long_tail_categories:
        return np.nan
    observed = set(df[category_col].dropna().unique())
    return len(observed & long_tail_categories) / len(long_tail_categories)


def data_freshness_index(df: pd.DataFrame, time_col: str,
                          t_now, t_max: pd.Timedelta) -> float:
    """
    Data Freshness Index (DFI): mean fractional freshness of records,
    where freshness decays linearly from 1 (at t_now) to 0 (at t_now − t_max).
    """
    times = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if times.empty:
        return np.nan
    ages = (t_now - times).dt.total_seconds()
    return float(np.mean(1.0 - np.clip(ages / t_max.total_seconds(), 0.0, 1.0)))


def missing_at_source_audit(df: pd.DataFrame, essential_cols: list) -> float:
    """
    Missing-at-Source Audit (MSA): fraction of records missing at least
    one essential field.
    """
    if not essential_cols:
        return np.nan
    return float(df[essential_cols].isna().any(axis=1).mean())
