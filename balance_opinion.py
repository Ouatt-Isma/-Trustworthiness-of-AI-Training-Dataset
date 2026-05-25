"""
VC-dimension-based class-balance opinion computation.

Opinion model
-------------
Given a dataset of N samples and a model with VC dimension d = W × L:

    N_req = (d / ε₀) · log(1/δ₀)           PAC learning sample bound
    u     = clip((log₁₀ N_req − log₁₀ N) / 10, 0, 1)   uncertainty mass

    n_b = |{c : |P(c) − 1/K| ≤ ε}|         classes within the tolerance zone
    b   = (1 − u) · n_b / K                 belief mass
    d   = (1 − u) · (K − n_b) / K           disbelief mass

where K is the number of classes and ε is the half-width of the tolerance zone.
"""

import math

import matplotlib.pyplot as plt
import numpy as np


def vc_dimension(n_params: int, n_layers: int) -> float:
    """Estimate VC dimension as d = W × L (parameter count × layer count)."""
    return float(n_params * n_layers)


def required_sample_size(vc_dim: float, epsilon: float = 0.05,
                          delta: float = 0.5) -> float:
    """PAC learning sample bound: N_req = (d / ε) · log(1/δ)."""
    return (vc_dim / epsilon) * math.log(1.0 / delta)


def dataset_size_uncertainty(model, n_samples: int) -> float:
    """
    Uncertainty mass from the gap between dataset size and PAC requirement:
        u = clip((log₁₀ N_req − log₁₀ N) / 10, 0, 1).
    """
    vc_dim = vc_dimension(model.count_params(), len(model.layers))
    n_req = required_sample_size(vc_dim)
    return float(np.clip((np.log10(n_req) - np.log10(n_samples)) / 10, 0.0, 1.0))


def balance_belief_scores(class_probs: list[float],
                           eps: float) -> tuple[int, int]:
    """Count (n_balanced, n_imbalanced) classes w.r.t. the uniform reference 1/K."""
    p_uniform = 1.0 / len(class_probs)
    n_balanced = sum(1 for p in class_probs if abs(p - p_uniform) <= eps)
    return n_balanced, len(class_probs) - n_balanced


def compute_balance_opinion(model, n_samples: int, class_probs: list[float],
                             eps: float = 0.02) -> tuple[float, float, float]:
    """
    Compute the class-balance opinion (b, d, u) for a dataset.

    Parameters
    ----------
    model       : Keras model (n_params × n_layers defines VC dimension)
    n_samples   : dataset size N
    class_probs : per-class frequencies summing to 1
    eps         : half-width of the balance tolerance zone around 1/K

    Returns
    -------
    (belief, disbelief, uncertainty)
    """
    n_balanced, n_imbalanced = balance_belief_scores(class_probs, eps)
    u = dataset_size_uncertainty(model, n_samples)
    total = n_balanced + n_imbalanced
    return (float((1.0 - u) * n_balanced / total),
            float((1.0 - u) * n_imbalanced / total),
            float(u))


def plot_epsilon_sweep(model, n_samples: int, class_probs: list[float],
                        title: str, eps_max: float = None,
                        stability_zone: tuple[float, float] = None,
                        output_path: str = None) -> list[tuple]:
    """
    Plot belief, disbelief, and uncertainty as ε varies from 0 to eps_max,
    with an optional shaded stability zone.

    Returns the list of (b, d, u) opinions for each ε value.
    """
    n_classes = len(class_probs)
    if eps_max is None:
        eps_max = 2.0 / n_classes

    eps_values = np.linspace(0.0, eps_max, 100)
    opinions = [compute_balance_opinion(model, n_samples, class_probs, eps=e)
                for e in eps_values]

    plt.figure(figsize=(9, 6))
    plt.plot(eps_values, [op[0] for op in opinions], label="belief",      linewidth=2)
    plt.plot(eps_values, [op[1] for op in opinions], label="disbelief",   linewidth=2)
    plt.plot(eps_values, [op[2] for op in opinions], label="uncertainty", linewidth=2)
    plt.axvline(x=1.0 / n_classes, color="r", linestyle="--", linewidth=2,
                label="Expected probability (1/K)")

    if stability_zone is not None:
        ymin, ymax = plt.gca().get_ylim()
        plt.fill_between(list(stability_zone), ymin, ymax,
                         color="orange", alpha=0.3, label="Stability zone")

    plt.xlabel("ε")
    plt.ylabel("Opinion mass")
    plt.title(title)
    plt.legend()
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2))
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.show()

    return opinions
