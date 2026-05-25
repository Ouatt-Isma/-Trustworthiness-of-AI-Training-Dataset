"""
Annotation-level Subjective Logic opinion analysis for CIFAR-10H.

Converts per-image annotation count arrays into BinomialOpinions and
measures Pearson correlation between each opinion mass (belief, disbelief,
uncertainty) and annotator disagreement across sub-sample sizes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 14})


class AnnotationOpinion:
    """
    Binomial opinion derived from annotation vote counts.

    For an annotation vector with prior weight W:
        b = r / (r + s + W),   d = s / (r + s + W),   u = W / (r + s + W)

    where r = votes for the positive (majority or specified) label,
          s = remaining votes.
    """

    def __init__(self, t: float, d: float, u: float, a: float = 0.5):
        assert round(t + d + u, 10) == 1, f"t={t}, d={d}, u={u} must sum to 1"
        self.t = t
        self.d = d
        self.a = a
        self.u = 1.0 - self.t - self.d

    def __repr__(self):
        return f"AnnotationOpinion(t={self.t:.3f}, d={self.d:.3f}, u={self.u:.3f})"


def annotation_counts_to_opinions(
    data: np.ndarray,
    labels=None,
    W: int = 2,
) -> tuple:
    """
    Convert a (N × C) annotation count matrix into BinomialOpinions.

    Parameters
    ----------
    data   : ndarray of shape (N, C) — raw annotation vote counts.
    labels : list of length N.  If given, the specified label is used as the
             positive class; otherwise the majority label is used.
    W      : prior weight.

    Returns
    -------
    opinions        : list[AnnotationOpinion]
    inferred_labels : list[int]  (majority label per item; empty when labels given)
    """
    opinions: list[AnnotationOpinion] = []
    inferred_labels: list[int] = []

    for i in range(len(data)):
        n_total = int(np.sum(data[i]))
        if labels is None:
            n_pos = int(np.max(data[i]))
            inferred_labels.append(int(np.argmax(data[i])))
        else:
            n_pos = int(data[i][labels[i]])

        n_neg = n_total - n_pos
        denom = n_pos + n_neg + W
        opinions.append(AnnotationOpinion(n_pos / denom, n_neg / denom, W / denom))

    return opinions, inferred_labels


def cap_annotation_counts(arr: np.ndarray, max_count: int = 10) -> np.ndarray:
    """
    Reduce per-item annotation totals to at most `max_count` votes.

    Excess votes are removed greedily from the most-voted label.
    """
    arr_new = arr.copy()
    for i in range(arr_new.shape[0]):
        row = arr_new[i]
        excess = int(np.sum(row)) - max_count
        if excess <= 0:
            continue
        for idx in np.argsort(-row):
            reducible = min(int(row[idx]), excess)
            row[idx] -= reducible
            excess -= reducible
            if excess <= 0:
                break
        arr_new[i] = row
    return arr_new


def _compute_box_stats(data: np.ndarray):
    """Return (lower_whisker, Q1, median, Q3, upper_whisker)."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = float(np.min(data[data >= q1 - 1.5 * iqr]))
    upper = float(np.max(data[data <= q3 + 1.5 * iqr]))
    return lower, float(q1), float(np.percentile(data, 50)), float(q3), upper


def plot_opinion_boxplot(
    t_vals: np.ndarray,
    d_vals: np.ndarray,
    u_vals: np.ndarray,
    title: str,
):
    """Boxplot of belief / disbelief / uncertainty masses. Saves as <title>.pdf."""
    stats_t = _compute_box_stats(t_vals)
    stats_d = _compute_box_stats(d_vals)

    fig, ax = plt.subplots()
    labels = ['Trust Mass', 'Distrust Mass', 'Uncertainty Mass']
    box = ax.boxplot(
        [t_vals, d_vals, u_vals],
        vert=True, patch_artist=True, labels=labels,
    )

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    extra_ticks = [stats_t[0], stats_t[4], stats_d[0], stats_d[4]]
    ax.set_yticks(sorted(set(ax.get_yticks()).union(extra_ticks)))

    for color, lbl in zip(colors, labels):
        ax.plot([], [], color=color, label=lbl)

    ax.set_title(title)
    ax.set_ylabel("Opinion mass")
    plt.tight_layout()

    pdf_path = f"{title}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    plt.close(fig)


def subsample_annotation_counts(data: np.ndarray, n: int) -> np.ndarray:
    """
    Subsample each row of a count matrix to exactly `n` annotations.

    Rows with ≤ n total votes are kept unchanged. The majority-label proportion
    is preserved via proportional sampling of the remaining votes.
    """
    result = []
    for row in data:
        total = int(np.sum(row))
        if total <= n:
            result.append(row.copy())
            continue

        maj_idx = int(np.argmax(row))
        maj_keep = max(1, int(np.round((row[maj_idx] / total) * n)))
        remaining = n - maj_keep

        other_idx = np.delete(np.arange(len(row)), maj_idx)
        new_row = np.zeros_like(row, dtype=int)
        new_row[maj_idx] = maj_keep

        if remaining > 0 and np.sum(row[other_idx]) > 0:
            proportions = row[other_idx] / np.sum(row[other_idx])
            new_row[other_idx] = np.random.multinomial(remaining, proportions)

        result.append(new_row)
    return np.array(result)


def _annotator_disagreement(row: np.ndarray) -> float:
    """Disagreement = 1 − majority-vote share.  Returns 0 for empty rows."""
    total = np.sum(row)
    return float(1.0 - np.max(row) / total) if total > 0 else 0.0


def run_annotator_size_experiment(
    data: np.ndarray,
    n_annotations: int,
    W: int = 2,
) -> dict:
    """
    Subsample to `n_annotations` per item and compute Pearson correlations
    between each opinion mass and annotator disagreement.

    Returns
    -------
    dict with keys "t", "d", "u" — each a (correlation, mean) tuple.
    """
    sub = subsample_annotation_counts(data, n_annotations)
    opinions, _ = annotation_counts_to_opinions(sub, W=W)

    t_vals = np.array([op.t for op in opinions])
    d_vals = np.array([op.d for op in opinions])
    u_vals = np.array([op.u for op in opinions])
    disagreements = np.array([_annotator_disagreement(row) for row in sub])

    return {
        "t": (float(np.corrcoef(t_vals, disagreements)[0, 1]), float(np.mean(t_vals))),
        "d": (float(np.corrcoef(d_vals, disagreements)[0, 1]), float(np.mean(d_vals))),
        "u": (float(np.corrcoef(u_vals, disagreements)[0, 1]), float(np.mean(u_vals))),
    }


def run_cifar10h_label_analysis(counts_path: str = "cifar10h-counts.npy"):
    """
    Full CIFAR-10H annotation-level opinion analysis.

    Computes opinions from raw and capped (≤ 10) annotation count matrices,
    plots belief/disbelief/uncertainty boxplots, and reports Pearson correlations
    with annotator disagreement for n ∈ {5, 10, 20, 50}.  Outputs
    results_table.{tex,csv}.
    """
    data = np.load(counts_path)

    opinions_raw, labels = annotation_counts_to_opinions(data)
    print(f"Items: {len(opinions_raw)}")
    plot_opinion_boxplot(
        np.array([op.t for op in opinions_raw]),
        np.array([op.d for op in opinions_raw]),
        np.array([op.u for op in opinions_raw]),
        title="CIFAR-10H",
    )

    data_capped = cap_annotation_counts(data, max_count=10)
    opinions_capped, _ = annotation_counts_to_opinions(data_capped, labels)
    print(f"Items (capped at 10): {len(opinions_capped)}")
    plot_opinion_boxplot(
        np.array([op.t for op in opinions_capped]),
        np.array([op.d for op in opinions_capped]),
        np.array([op.u for op in opinions_capped]),
        title="CIFAR-10H Cropped to 10 annotators",
    )

    rows = []
    for n in [5, 10, 20, 50]:
        stats = run_annotator_size_experiment(data, n)
        rows.append([
            n,
            round(stats["t"][0], 2), round(stats["t"][1], 2),
            round(stats["d"][0], 2), round(stats["d"][1], 2),
            round(stats["u"][0], 2), round(stats["u"][1], 2),
        ])

    df = pd.DataFrame(rows, columns=[
        "Annotations per item",
        "Corr(t)", "Avg(t)",
        "Corr(d)", "Avg(d)",
        "Corr(u)", "Avg(u)",
    ])
    print("\nCorrelation with annotator disagreement:")
    print(df.to_string(index=False))

    df.to_latex("results_table.tex", index=False, float_format="%.2f")
    df.to_csv("results_table.csv", index=False)
    print("\nSaved: results_table.tex, results_table.csv")


if __name__ == "__main__":
    run_cifar10h_label_analysis()
