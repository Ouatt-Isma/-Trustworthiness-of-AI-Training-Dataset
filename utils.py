"""
Quality-to-evidence mapping utilities for BPQ-based opinion computation.

All quality scores are in [0, 1]; evidence (r, s) follows from the BPQ
operator; opinions (b, d, u) are derived via Subjective Logic.
"""

import numpy as np


def lower_is_better(x: float, T: float) -> float:
    """
    Linear quality score for a metric where lower values are better.

        q = 1       if x ≤ 0
        q = 1 − x/T if 0 < x < T
        q = 0       if x ≥ T
    """
    if x is None or np.isnan(x):
        return None
    if x <= 0:
        return 1.0
    if x >= T:
        return 0.0
    return 1.0 - x / T


def higher_is_better(x: float, A: float, L: float) -> float:
    """
    Linear quality score for a metric where higher values are better.

        q = 0             if x ≤ A
        q = (x−A)/(L−A)  if A < x < L
        q = 1             if x ≥ L
    """
    if x is None or np.isnan(x):
        return None
    if x <= A:
        return 0.0
    if x >= L:
        return 1.0
    return (x - A) / (L - A)


def metric_to_evidence(q: float, K: float = 5) -> tuple[float, float]:
    """
    Convert a quality score q ∈ [0, 1] to BPQ evidence (r, s):
        r = q·K,  s = (1−q)·K.
    """
    if q is None or np.isnan(q):
        return 0.0, 0.0
    q = float(np.clip(q, 0.0, 1.0))
    return q * K, (1.0 - q) * K


def aggregate_evidence(evidence_list: list[tuple]) -> tuple[float, float]:
    """Sum a list of (r, s) pairs into a single (r_total, s_total)."""
    if not evidence_list:
        return 0.0, 0.0
    return sum(r for r, _ in evidence_list), sum(s for _, s in evidence_list)


def bpq(r: float, s: float, W: float = 2) -> tuple[float, float, float]:
    """
    Binomial Probability Quantification (BPQ).

    Maps evidence (r, s) and prior weight W to a Subjective Logic opinion:
        b = r / (r + s + W)
        d = s / (r + s + W)
        u = W / (r + s + W)
    """
    r, s, W = float(r), float(s), float(W)
    denom = r + s + W
    if denom == 0:
        return 0.0, 0.0, 1.0
    return r / denom, s / denom, W / denom
