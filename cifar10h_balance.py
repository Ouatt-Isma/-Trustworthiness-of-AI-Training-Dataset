"""
VC-dimension-based annotation-label balance opinion analysis for CIFAR-10H.

Complements cifar_10h.py (annotation + processing trace) with a single-metric
balance opinion and an epsilon-sweep stability plot over the 10 CIFAR-10 classes.
"""

import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 14})

try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Input
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

from balance_opinion import compute_balance_opinion, plot_epsilon_sweep

N_CIFAR10_CLASSES = 10


def build_annotation_classifier(input_dim: int, n_classes: int) -> Sequential:
    """
    Two-hidden-layer MLP for annotation-label classification.

    The architecture (parameter count × layer count) determines the VC dimension
    used for uncertainty estimation; the model is not trained by default.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_label_distribution(y: np.ndarray, title: str):
    """Count-plot of annotation label frequencies."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Chosen label (CIFAR-10 class)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_class_probability_distribution(class_probs: list, title: str):
    """Histogram + KDE of per-class annotation probabilities with uniform reference 1/K."""
    plt.figure(figsize=(8, 5))
    sns.histplot(class_probs, bins=len(class_probs), stat="density", alpha=0.6)
    sns.kdeplot(class_probs, linewidth=2)
    plt.axvline(x=1.0 / len(class_probs), color="red", label="Uniform (1/K)")
    plt.title(title)
    plt.xlabel("Class probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_cifar10h_balance_analysis(
    data_path: str = "data/cifar10h-raw.csv",
    label_col: str = "chosen_label",
    output_dir: str = "saved",
):
    """
    Compute the annotation-label balance opinion for CIFAR-10H and produce
    an epsilon-sweep stability plot.

    Parameters
    ----------
    data_path : path to cifar10h-raw.csv
    label_col : annotation label column (default: "chosen_label")
    output_dir: directory for saved figures
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)

    n_classes = len(np.unique(y))
    classes   = list(range(n_classes))

    numeric_cols = (
        df.select_dtypes(include=[np.number])
          .drop(columns=[label_col], errors="ignore")
          .columns.tolist()
    )
    input_dim = max(len(numeric_cols), 1)

    plot_label_distribution(y, f"CIFAR-10H – {label_col} Distribution")

    class_probs = [Counter(y)[i] / len(y) for i in classes]
    plot_class_probability_distribution(
        class_probs, f"CIFAR-10H – {label_col} Class Probabilities",
    )

    model = build_annotation_classifier(input_dim, n_classes)

    eps_default = 0.01
    op = compute_balance_opinion(model, len(df), class_probs, eps=eps_default)
    print(f"\nOpinion at eps={eps_default}:  b={op[0]:.4f},  d={op[1]:.4f},  u={op[2]:.4f}")

    p_uniform = 1.0 / n_classes
    plot_epsilon_sweep(
        model, len(df), class_probs,
        title=f"Opinion Evolution vs ε — CIFAR-10H ({label_col})",
        eps_max=2.0 / n_classes,
        stability_zone=(p_uniform * 0.9, p_uniform * 1.1),
        output_path=os.path.join(output_dir, "cifar10h_epsilon_sweep.pdf"),
    )


if __name__ == "__main__":
    run_cifar10h_balance_analysis()
