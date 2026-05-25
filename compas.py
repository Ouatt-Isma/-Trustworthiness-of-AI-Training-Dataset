"""
Four-stage Subjective Logic reliability assessment for the COMPAS recidivism
dataset (Angwin et al., 2016).

Stages: sampling → collection → annotation → processing.
Each stage aggregates metric evidence into a BinomialOpinion via BPQ;
the final opinion is the conjunction ω_s ∧ ω_c ∧ ω_a ∧ ω_p.
"""

import numpy as np
import pandas as pd
from sl import BinomialOpinion
from utils import (
    aggregate_evidence, bpq, higher_is_better,
    lower_is_better, metric_to_evidence,
)
import sampling
import collection
import annotation
import processing


HP = {
    "K": 2,   # evidence strength per metric
    "W": 2,   # BPQ prior weight

    "T_DG":          0.30,
    "A_MCR":         0.50,  "L_MCR":  1.00,
    "A_LTCI":        0.30,  "L_LTCI": 1.00,
    "T_MSA":         0.10,
    "TMAX_DFI_DAYS": 365,
    "A_DFI":         0.00,  "L_DFI":  1.00,

    "T_ORR":  0.05,
    "T_MDD":  0.20,
    "T_PIID": 1 / 1000,
    "T_SALS": 0.70,
    "T_RRS":  0.50,

    "T_GLBI": 0.20,

    "T_DIOR": 0.01,
    "T_FLCA": 0.60,
}


def run_compas_evaluation(data_path: str = "data/compas-scores-two-years.csv"):
    """
    Run the full COMPAS reliability assessment and write compas_reliability_trace.csv.

    Returns
    -------
    final_opinion : BinomialOpinion
    trace_df      : pd.DataFrame  (full metric trace)
    """
    records: list[dict] = []

    def log_metric(stage, name, raw_value, q_value, direction, hyperparams, K, r, s):
        records.append({
            "kind": "metric", "stage": stage, "name": name,
            "raw_value": raw_value, "q_value": q_value, "direction": direction,
            "hyperparameters": hyperparams, "K": K, "r": r, "s": s,
            "b": np.nan, "d": np.nan, "u": np.nan,
        })

    def log_stage(stage, r_total, s_total, W, opinion):
        b, d, u = opinion
        records.append({
            "kind": "stage", "stage": stage, "name": "BPQ Quantified Opinion",
            "raw_value": None, "q_value": None, "direction": None,
            "hyperparameters": {"W": W}, "K": None,
            "r": r_total, "s": s_total, "b": b, "d": d, "u": u,
        })

    def log_final(op: BinomialOpinion):
        records.append({
            "kind": "final", "stage": "final_conjunction",
            "name": "ω_s ∧ ω_c ∧ ω_a ∧ ω_p",
            "raw_value": None, "q_value": None, "direction": None,
            "hyperparameters": {"operator": "conjunction (∧)", "stages": 4},
            "K": None, "r": None, "s": None,
            "b": op.b, "d": op.d, "u": op.u,
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
        log_metric(stage, name,
                   float(raw) if np.isscalar(raw) else raw,
                   float(q)   if np.isscalar(q)   else q,
                   direction, hyperparams, HP["K"], r, s)

    df = pd.read_csv(data_path)
    df = df[
        df["days_b_screening_arrest"].between(-30, 30) & (df["is_recid"] != -1)
    ].reset_index(drop=True)
    df["c_jail_in"] = pd.to_datetime(df["c_jail_in"], errors="coerce")
    median_date = df["c_jail_in"].dropna().median()

    population_dist = {
        "African-American": 0.143,
        "Caucasian":        0.491,
        "Hispanic":         0.287,
        "Native American":  0.0011,
        "Asian":            0.0302,
    }
    population_dist["Other"] = 1.0 - sum(population_dist.values())

    essential_cols = ["race", "sex", "age", "priors_count",
                      "decile_score", "two_year_recid"]

    # Sampling
    ev_s: list[tuple] = []

    DG = sampling.distribution_gap(df, "race", population_dist)
    record("sampling", "DG", DG, lower_is_better(DG, T=HP["T_DG"]),
           "lower_is_better", {"T": HP["T_DG"]}, ev_s)

    MCR = sampling.minority_coverage_rate(df, "race", "African-American", population_dist)
    record("sampling", "MCR(African-American)", MCR,
           higher_is_better(MCR, A=HP["A_MCR"], L=HP["L_MCR"]),
           "higher_is_better", {"A": HP["A_MCR"], "L": HP["L_MCR"]}, ev_s)

    ref_dist = df["c_charge_desc"].value_counts(normalize=True)
    long_tail_cats = set(ref_dist[ref_dist < 0.01].index)
    eval_df = df.dropna(subset=["c_jail_in"])
    if median_date is not None and not np.isnan(pd.Timestamp(median_date).value):
        eval_df = eval_df[eval_df["c_jail_in"] > median_date]
    LTCI = sampling.long_tail_coverage_index(eval_df, "c_charge_desc", long_tail_cats)
    record("sampling", "LTCI(c_charge_desc)", LTCI,
           higher_is_better(LTCI, A=HP["A_LTCI"], L=HP["L_LTCI"]),
           "higher_is_better",
           {"A": HP["A_LTCI"], "L": HP["L_LTCI"],
            "tail_def": "P_ref<0.01", "eval_slice": "post_median(c_jail_in)"},
           ev_s)

    df_dates = df.dropna(subset=["c_jail_in"])
    if len(df_dates) > 0:
        DFI = sampling.data_freshness_index(
            df_dates, "c_jail_in", df_dates["c_jail_in"].max(),
            pd.Timedelta(days=HP["TMAX_DFI_DAYS"]))
        record("sampling", "DFI(c_jail_in)", DFI,
               higher_is_better(DFI, A=HP["A_DFI"], L=HP["L_DFI"]),
               "higher_is_better",
               {"A": HP["A_DFI"], "L": HP["L_DFI"],
                "t_max_days": HP["TMAX_DFI_DAYS"]},
               ev_s)

    MSA = sampling.missing_at_source_audit(df, essential_cols)
    record("sampling", "MSA(essential_cols)", MSA,
           lower_is_better(MSA, T=HP["T_MSA"]),
           "lower_is_better", {"T": HP["T_MSA"], "cols": essential_cols}, ev_s)

    r_s, s_s = aggregate_evidence(ev_s)
    op_s_tuple = bpq(r_s, s_s, W=HP["W"])
    log_stage("sampling", r_s, s_s, HP["W"], op_s_tuple)

    # Collection
    ev_c: list[tuple] = []

    ORR = collection.out_of_range_rate(df, "decile_score", valid_min=1, valid_max=10)
    record("collection", "ORR(decile_score)", ORR,
           lower_is_better(ORR, T=HP["T_ORR"]),
           "lower_is_better", {"T": HP["T_ORR"], "min": 1, "max": 10}, ev_c)

    MDD = collection.missing_data_disparity(df, "race", cols=essential_cols)
    record("collection", "MDD(by_race)", MDD,
           lower_is_better(MDD, T=HP["T_MDD"]),
           "lower_is_better", {"T": HP["T_MDD"], "cols": essential_cols}, ev_c)

    pii_cols = [c for c in ["name", "dob", "first", "last"] if c in df.columns]
    PIID = collection.pii_density(df, pii_cols)
    record("collection", "PIID", PIID,
           lower_is_better(PIID, T=HP["T_PIID"]),
           "lower_is_better", {"T": HP["T_PIID"], "pii_cols": pii_cols}, ev_c)

    feature_cols = [c for c in ["age", "priors_count", "juv_fel_count",
                                 "juv_misd_count", "decile_score"]
                    if c in df.columns]
    df_feat = df.dropna(subset=feature_cols + ["race"])
    SALS = collection.sensitive_attribute_leakage(df_feat, "race", feature_cols)
    record("collection", "SALS(race|features)", SALS,
           lower_is_better(SALS, T=HP["T_SALS"]),
           "lower_is_better", {"T": HP["T_SALS"], "features": feature_cols}, ev_c)

    qid_cols = [c for c in ["sex", "race", "age_cat"] if c in df.columns]
    RRS = collection.reidentification_risk(df, quasi_identifiers=qid_cols)
    record("collection", "RRS(qid=sex,race,age_cat)", RRS,
           lower_is_better(RRS, T=HP["T_RRS"]),
           "lower_is_better", {"T": HP["T_RRS"], "quasi_identifiers": qid_cols}, ev_c)

    r_c, s_c = aggregate_evidence(ev_c)
    op_c_tuple = bpq(r_c, s_c, W=HP["W"])
    log_stage("collection", r_c, s_c, HP["W"], op_c_tuple)

    # Annotation
    ev_a: list[tuple] = []

    df_2races = df[df["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)
    if len(df_2races) > 0:
        GLBI = annotation.group_label_bias_index(df_2races, "two_year_recid", "race")
        record("annotation", "GLBI(two_year_recid|race)", GLBI,
               lower_is_better(GLBI, T=HP["T_GLBI"]),
               "lower_is_better",
               {"T": HP["T_GLBI"], "groups": ["African-American", "Caucasian"]},
               ev_a)

    r_a, s_a = aggregate_evidence(ev_a)
    op_a_tuple = bpq(r_a, s_a, W=HP["W"])
    log_stage("annotation", r_a, s_a, HP["W"], op_a_tuple)

    # Processing
    ev_p: list[tuple] = []

    DIOR = processing.duplicate_instance_overlap_rate(df)
    record("processing", "DIOR(duplicate rows)", DIOR,
           lower_is_better(DIOR, T=HP["T_DIOR"]),
           "lower_is_better", {"T": HP["T_DIOR"]}, ev_p)

    flca_features = [c for c in ["age", "priors_count", "juv_fel_count",
                                  "juv_misd_count", "juv_other_count"]
                     if c in df.columns]
    FLCA = processing.feature_label_correlation(
        df, feature_cols=flca_features, label_col="two_year_recid")
    record("processing", "FLCA(max|corr(feature,label)|)", FLCA,
           lower_is_better(FLCA, T=HP["T_FLCA"]),
           "lower_is_better", {"T": HP["T_FLCA"], "features": flca_features}, ev_p)

    r_p, s_p = aggregate_evidence(ev_p)
    op_p_tuple = bpq(r_p, s_p, W=HP["W"])
    log_stage("processing", r_p, s_p, HP["W"], op_p_tuple)

    # Final conjunction
    op_s = BinomialOpinion(*op_s_tuple)
    op_c = BinomialOpinion(*op_c_tuple)
    op_a = BinomialOpinion(*op_a_tuple)
    op_p = BinomialOpinion(*op_p_tuple)
    final = op_s & op_c & op_a & op_p
    log_final(final)

    print("\n===== COMPAS DATASET RELIABILITY ASSESSMENT =====\n")
    W = 12
    print(f"  {'Stage':<{W}}  {'b':>8}  {'d':>8}  {'u':>8}  {'E[ω]':>8}")
    print("  " + "-" * 50)
    for label, op in [("Sampling", op_s), ("Collection", op_c),
                      ("Annotation", op_a), ("Processing", op_p)]:
        print(f"  {label:<{W}}  {op.b:>8.4f}  {op.d:>8.4f}  "
              f"{op.u:>8.4f}  {op.expectation():>8.4f}")
    print("  " + "=" * 50)
    print(f"  {'FINAL (∧)':<{W}}  {final.b:>8.4f}  {final.d:>8.4f}  "
          f"{final.u:>8.4f}  {final.expectation():>8.4f}  (a={final.a:.4f})")
    print(f"\n{final}")

    trace_cols = ["kind", "stage", "name", "raw_value", "q_value", "direction",
                  "hyperparameters", "K", "r", "s", "b", "d", "u"]
    trace_df = pd.DataFrame(records)[trace_cols]
    trace_df.to_csv("compas_reliability_trace.csv", index=False)
    print("\nSaved: compas_reliability_trace.csv")
    pd.set_option("display.max_columns", None)
    print(trace_df)

    return final, trace_df


if __name__ == "__main__":
    run_compas_evaluation()
