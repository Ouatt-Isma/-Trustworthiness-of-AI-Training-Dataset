"""
Two-stage Subjective Logic reliability assessment for the CIFAR-10H
human-annotation dataset (Peterson et al., 2019).

Stages: annotation (IAA, OLR, USA) → processing (DIOR).
The final opinion is the conjunction ω_a ∧ ω_p.
"""

import numpy as np
import pandas as pd
from sl import BinomialOpinion

from utils import (
    lower_is_better,
    higher_is_better,
    metric_to_evidence,
    aggregate_evidence,
    bpq,
)

import annotation
import processing


HP = {
    "K": 5,
    "W": 5,

    "A_IAA": 0.60, "L_IAA": 1.00,
    "T_OLR": 0.30,
    "T_USA": 1.00,

    "T_DIOR": 0.01,
}


def run_cifar10h_evaluation(data_path: str = "data/cifar10h-raw.csv"):
    """
    Run the CIFAR-10H reliability assessment (annotation + processing).

    Returns
    -------
    final_opinion : BinomialOpinion  (annotation ∧ processing)
    trace_df      : pd.DataFrame
    """
    records: list[dict] = []

    def log_metric(stage, name, raw_value, q_value, direction, hyperparams, K, r, s):
        records.append({
            "kind": "metric", "stage": stage, "name": name,
            "raw_value": raw_value, "q_value": q_value,
            "direction": direction, "hyperparameters": hyperparams,
            "K": K, "r": r, "s": s,
            "b": np.nan, "d": np.nan, "u": np.nan,
        })

    def log_stage(stage, r_total, s_total, W, opinion):
        b, d, u = opinion
        records.append({
            "kind": "stage", "stage": stage, "name": "BPQ Quantified Opinion",
            "raw_value": None, "q_value": None, "direction": None,
            "hyperparameters": {"W": W}, "K": None,
            "r": r_total, "s": s_total,
            "b": b, "d": d, "u": u,
        })

    def record(stage, name, raw, q, direction, hyperparams, evidence_list):
        if raw is None or q is None:
            return
        if isinstance(raw, float) and np.isnan(raw):
            return
        if isinstance(q, float) and np.isnan(q):
            return
        r, s = metric_to_evidence(q, K=HP["K"])
        evidence_list.append((r, s))
        log_metric(stage, name, float(raw), float(q), direction, hyperparams, HP["K"], r, s)

    df = pd.read_csv(data_path)

    rt = df["reaction_time"].astype(float)
    t_low, t_high = rt.min(), rt.max()
    df["confidence"] = (
        (1.0 - (rt - t_low) / (t_high - t_low)).clip(0.0, 1.0)
        if t_high > t_low else 1.0
    )

    ev_a: list[tuple] = []

    pairs = [
        (g["chosen_label"].values[0], g["chosen_label"].values[1])
        for _, g in df.groupby("image_filename")
        if len(g) >= 2
    ]
    if pairs:
        y1, y2 = zip(*pairs)
        IAA = annotation.inter_annotator_agreement(y1, y2)
        record("annotation", "IAA", IAA,
               higher_is_better(IAA, A=HP["A_IAA"], L=HP["L_IAA"]),
               "higher_is_better", {"A": HP["A_IAA"], "L": HP["L_IAA"]}, ev_a)

    OLR = 1.0 - df["correct_guess"].mean()
    record("annotation", "OLR", OLR,
           lower_is_better(OLR, T=HP["T_OLR"]),
           "lower_is_better", {"T": HP["T_OLR"]}, ev_a)

    label_conf = (
        df.groupby(["image_filename", "chosen_label"])["confidence"]
        .sum()
        .unstack(fill_value=0)
    )
    label_probs = label_conf.div(label_conf.sum(axis=1), axis=0).values
    USA = annotation.uncertainty_score_from_annotators(label_probs)
    record("annotation", "USA(confidence-weighted)", USA,
           lower_is_better(USA, T=HP["T_USA"]),
           "lower_is_better",
           {"T": HP["T_USA"], "confidence": "reaction_time_minmax"}, ev_a)

    r_a, s_a = aggregate_evidence(ev_a)
    op_a_tuple = bpq(r_a, s_a, W=HP["W"])
    log_stage("annotation", r_a, s_a, HP["W"], op_a_tuple)

    ev_p: list[tuple] = []

    DIOR = processing.duplicate_instance_overlap_rate(
        df[["image_filename", "chosen_label"]]
    )
    record("processing", "DIOR", DIOR,
           lower_is_better(DIOR, T=HP["T_DIOR"]),
           "lower_is_better", {"T": HP["T_DIOR"]}, ev_p)

    r_p, s_p = aggregate_evidence(ev_p)
    op_p_tuple = bpq(r_p, s_p, W=HP["W"])
    log_stage("processing", r_p, s_p, HP["W"], op_p_tuple)

    op_a = BinomialOpinion(*op_a_tuple)
    op_p = BinomialOpinion(*op_p_tuple)
    final = op_a & op_p

    print("\n===== CIFAR-10H DATASET RELIABILITY ASSESSMENT =====\n")
    W = 12
    print(f"  {'Stage':<{W}}  {'b':>8}  {'d':>8}  {'u':>8}  {'E[ω]':>8}")
    print("  " + "-" * 50)
    for label, op in [("Annotation", op_a), ("Processing", op_p)]:
        print(f"  {label:<{W}}  {op.b:>8.4f}  {op.d:>8.4f}  "
              f"{op.u:>8.4f}  {op.expectation():>8.4f}")
    print("  " + "=" * 50)
    print(f"  {'FINAL (∧)':<{W}}  {final.b:>8.4f}  {final.d:>8.4f}  "
          f"{final.u:>8.4f}  {final.expectation():>8.4f}  (a={final.a:.4f})")
    print(f"\n{final}")

    trace_cols = [
        "kind", "stage", "name",
        "raw_value", "q_value", "direction",
        "hyperparameters", "K", "r", "s", "b", "d", "u",
    ]
    trace_df = pd.DataFrame(records)[trace_cols]
    trace_df.to_csv("cifar10h_reliability_trace.csv", index=False)
    print("\nSaved: cifar10h_reliability_trace.csv")

    return final, trace_df


if __name__ == "__main__":
    run_cifar10h_evaluation()
