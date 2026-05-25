"""
Class-balance opinion analysis for the GTSRB dataset (Stallkamp et al., 2012).

Pipeline
--------
1. Load images (32×32 RGB, normalised to [0, 1]).
2. Define a 3-block CNN architecture (VC dimension drives uncertainty).
3. Compute class-balance opinion via VC-dimension-based uncertainty and a
   tolerance-zone balance score over 43 classes.
4. Epsilon sweep: opinion as a function of the tolerance half-width ε.
5. Collaborative bias simulation: sweep over the number of data-collection
   nodes that under-represent warning-sign classes (18–31).
6. (Optional, train=True) Train the CNN and evaluate group accuracy.
"""

import os
import warnings
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from balance_opinion import (
    compute_balance_opinion,
    dataset_size_uncertainty,
    plot_epsilon_sweep,
)

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 18})

try:
    from keras.models import Sequential
    from keras.layers import (
        BatchNormalization, Conv2D, Dense, Dropout,
        Flatten, Input, MaxPooling2D,
    )
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Dense, Dropout,
        Flatten, Input, MaxPooling2D,
    )


N_CLASSES  = 43
IMAGE_SIZE = 32
EPOCHS     = 5
BATCH_SIZE = 36

WARNING_SIGN_CLASSES = list(range(18, 32))
OTHER_SIGN_CLASSES   = [c for c in range(N_CLASSES) if c not in WARNING_SIGN_CLASSES]

LABEL_NAMES: dict[int, str] = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}


def load_train_images(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training images from <dataset_path>/Train/<class_id>/*.png.

    Returns
    -------
    X : ndarray (N, 32, 32, 3)  normalised RGB images
    y : ndarray (N,)            integer class labels
    """
    train_dir = os.path.join(dataset_path, "Train")
    images, labels = [], []

    for label_dir in sorted(os.listdir(train_dir)):
        label_path = os.path.join(train_dir, label_dir)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, img_file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
            images.append(img)
            labels.append(int(label_dir))

    return np.array(images), np.array(labels)


def load_test_images(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test images from <dataset_path>/Test.csv + <dataset_path>/Test/.

    Returns
    -------
    X_test : ndarray (N, 32, 32, 3)
    y_test : ndarray (N,)
    """
    csv_path = os.path.join(dataset_path, "Test.csv")
    df_test = pd.read_csv(csv_path)[["ClassId", "Path"]]

    images, labels = [], []
    for _, row in df_test.iterrows():
        filename = row["Path"].split("/")[-1]
        img = cv2.imread(os.path.join(dataset_path, "Test", filename))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        images.append(img)
        labels.append(int(row["ClassId"]))

    return np.array(images), np.array(labels)


def build_cnn_model(n_classes: int = N_CLASSES) -> Sequential:
    """
    3-block CNN for traffic-sign classification (32×32 RGB input).

    Architecture: (Conv2D → BatchNorm → MaxPool → Dropout) × 3,
    followed by Dense(256) → Dense(128) → Dropout(0.5) → softmax.
    """
    model = Sequential()
    model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

    model.add(Conv2D(64,  kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """Train with an 80/20 split and return the Keras history object."""
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=1,
    )


def plot_class_distribution(y: np.ndarray, title: str, output_path: str = None):
    """Bar chart of label frequency, sorted by count."""
    y_series = pd.Series(y).map(LABEL_NAMES)
    plt.figure(figsize=(25, 8))
    ax = sns.countplot(x=y_series, palette="viridis",
                       order=y_series.value_counts().index)
    for container in ax.containers:
        ax.bar_label(container, fontsize=12, padding=5)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.show()


def plot_class_probability_distribution(
    y,
    title: str,
    eps: float = 0.02,
    output_path: str = None,
):
    """
    KDE + histogram of per-class probabilities with a ±eps tolerance zone
    centred on the uniform reference 1/K.
    """
    class_probs = compute_class_probabilities(y)
    p_uniform = 1.0 / N_CLASSES

    plt.figure(figsize=(10, 6))
    sns.histplot(class_probs, bins=len(class_probs), stat="density",
                 alpha=0.6, color="skyblue", label="histogram")
    sns.kdeplot(class_probs, color="red", linewidth=2,
                bw_adjust=0.2, label="density")
    plt.axvline(x=p_uniform, color="b", label="Expected probability (1/K)")

    ymin, ymax = plt.gca().get_ylim()
    plt.fill_between(
        [p_uniform - eps, p_uniform + eps], ymin, ymax,
        color="orange", alpha=0.3, label="Tolerance zone",
    )
    plt.title(title)
    plt.xlabel("Class probability")
    plt.ylabel("Density")
    plt.xlim(0, 0.1)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.show()


def compute_class_probabilities(y) -> list[float]:
    """Per-class frequency list aligned to LABEL_NAMES (classes 0–42)."""
    c = Counter(y)
    n = len(y)
    return [c[i] / n for i in range(N_CLASSES)]


def evaluate_model(
    model: Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
    history=None,
):
    """Print accuracy and display a confusion matrix heatmap."""
    if history is not None:
        print(f"Train accuracy: {history.history['accuracy'][-1]:.4f}")

    preds = model.predict(x_test).argmax(axis=-1)
    print(f"Test  accuracy: {accuracy_score(y_test, preds):.4f}")

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(20, 15))
    sns.heatmap(
        cm, annot=True, fmt="g", cmap="Blues",
        xticklabels=LABEL_NAMES.values(),
        yticklabels=LABEL_NAMES.values(),
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def evaluate_by_sign_group(
    model: Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """
    Accuracy breakdown by sign category: warning signs (classes 18–31) vs other.

    Returns
    -------
    (acc_overall, acc_warning, acc_other)
    """
    preds = model.predict(x_test).argmax(axis=-1)
    acc_overall = accuracy_score(y_test, preds)

    warn_mask = np.isin(y_test, WARNING_SIGN_CLASSES)
    acc_warning = (
        accuracy_score(y_test[warn_mask], preds[warn_mask])
        if warn_mask.any() else float("nan")
    )
    acc_other = (
        accuracy_score(y_test[~warn_mask], preds[~warn_mask])
        if (~warn_mask).any() else float("nan")
    )

    print(f"Overall accuracy  : {acc_overall:.4f}")
    print(f"Warning signs     : {acc_warning:.4f}")
    print(f"Other signs       : {acc_other:.4f}")
    return acc_overall, acc_warning, acc_other


def build_biased_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_unbalanced_nodes: int,
    n_total_nodes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate federated collection where `n_unbalanced_nodes` of `n_total_nodes`
    nodes retain only one sample per warning-sign class.

    The dataset is partitioned into `n_total_nodes` round-robin sub-datasets.
    Biased nodes have all but the first occurrence of each warning-sign class
    removed, then all sub-datasets are merged.

    Returns
    -------
    X_combined, y_combined : merged dataset after applying bias.
    """
    sub_X = [list(X[i::n_total_nodes]) for i in range(n_total_nodes)]
    sub_y = [list(y[i::n_total_nodes]) for i in range(n_total_nodes)]

    keep_once = {cls: True for cls in WARNING_SIGN_CLASSES}

    for node_idx in range(n_unbalanced_nodes):
        to_remove = []
        for t, label in enumerate(sub_y[node_idx]):
            if label in WARNING_SIGN_CLASSES:
                if keep_once[label]:
                    keep_once[label] = False
                else:
                    to_remove.append(t)
        for idx in sorted(to_remove, reverse=True):
            del sub_y[node_idx][idx]
            del sub_X[node_idx][idx]

    X_combined = np.concatenate([np.array(s) for s in sub_X])
    y_combined = np.concatenate([np.array(s) for s in sub_y])
    return X_combined, y_combined


def run_collaborative_bias_sweep(
    X: np.ndarray,
    y: np.ndarray,
    model: Sequential,
    n_total_nodes: int = 10,
    output_dir: str = "saved",
) -> list[tuple[float, float, float]]:
    """
    Sweep over increasing numbers of biased nodes (0 to `n_total_nodes`) and
    compute the class-balance opinion for each merged dataset.

    Returns
    -------
    opinions : list of (belief, disbelief, uncertainty) tuples
    """
    opinions = []
    for n_biased in range(n_total_nodes + 1):
        X_biased, y_biased = build_biased_dataset(X, y, n_biased, n_total_nodes)
        class_probs = compute_class_probabilities(y_biased)
        op = compute_balance_opinion(model, len(X_biased), class_probs)
        opinions.append(op)
        print(f"  {n_biased:>3} unbalanced node(s): "
              f"b={op[0]:.3f}, d={op[1]:.3f}, u={op[2]:.3f}")

    n_values = list(range(n_total_nodes + 1))
    plt.figure(figsize=(9, 6))
    plt.plot(n_values, [op[0] for op in opinions], label="belief",      linewidth=2)
    plt.plot(n_values, [op[1] for op in opinions], label="disbelief",   linewidth=2)
    plt.plot(n_values, [op[2] for op in opinions], label="uncertainty", linewidth=2)
    plt.xlabel("Number of unbalanced sub-datasets")
    plt.ylabel("Opinion value")
    plt.title(f"Opinion Under Increasing Bias ({n_total_nodes} nodes)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, f"collaborative_bias_{n_total_nodes}nodes.pdf")
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()

    return opinions


def run_gtsrb_analysis(
    dataset_path: str = None,
    train: bool = False,
    collaborative: bool = False,
    n_collab_nodes: int = 10,
    output_dir: str = "saved",
):
    """
    Full GTSRB opinion analysis pipeline.

    Parameters
    ----------
    dataset_path  : Root directory of the GTSRB dataset.  If None, downloaded
                    automatically via kagglehub.
    train         : Fit the CNN before evaluation (disabled by default).
    collaborative : Run the federated bias sweep.
    n_collab_nodes: Number of simulated data-collection nodes.
    output_dir    : Directory for saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    if dataset_path is None:
        try:
            import kagglehub
            dataset_path = kagglehub.dataset_download(
                "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
            )
            print(f"Dataset path: {dataset_path}")
        except ImportError:
            raise RuntimeError(
                "kagglehub is not installed and no dataset_path was provided.\n"
                "Install with:  pip install kagglehub\n"
                "or pass:       run_gtsrb_analysis(dataset_path='/your/path')"
            )

    print("Loading training images …")
    X, y = load_train_images(dataset_path)
    print(f"  Train: {X.shape},  labels: {y.shape}")

    plot_class_distribution(
        y, "GTSRB – Original Dataset",
        output_path=os.path.join(output_dir, "class_distribution.pdf"),
    )
    plot_class_probability_distribution(
        y, "GTSRB – Class Probabilities",
        output_path=os.path.join(output_dir, "class_probs.pdf"),
    )

    model = build_cnn_model()

    if train:
        print("\nTraining CNN …")
        x_test, y_test = load_test_images(dataset_path)
        history = train_model(model, X, y)
        evaluate_model(model, x_test, y_test, history)
        evaluate_by_sign_group(model, x_test, y_test)

    class_probs = compute_class_probabilities(y)
    op_normal = compute_balance_opinion(model, len(X), class_probs)
    print(
        f"\nOpinion (original dataset):  "
        f"b={op_normal[0]:.4f},  d={op_normal[1]:.4f},  u={op_normal[2]:.4f}"
    )

    plot_epsilon_sweep(
        model, len(X), class_probs,
        title="GTSRB – Opinion Evolution vs ε",
        stability_zone=(0.0185, 0.0235),
        output_path=os.path.join(output_dir, "epsilon_sweep.pdf"),
    )

    if collaborative:
        print(f"\nRunning collaborative bias sweep ({n_collab_nodes} nodes) …")
        run_collaborative_bias_sweep(X, y, model, n_collab_nodes, output_dir)


if __name__ == "__main__":
    run_gtsrb_analysis()
