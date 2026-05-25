# Trustworthiness of AI Training Datasets

A framework for assessing the **reliability of AI training datasets** using **Subjective Logic (SL)** and the **Binomial Probability Quantification (BPQ)** operator. This is the implementation accompanying the dissertation research.

---

## Overview

Training data quality is a critical—yet often overlooked—dimension of AI trustworthiness. This framework formalizes dataset reliability assessment as a four-stage pipeline:

```
Sampling → Collection → Annotation → Processing
```

Each stage computes quality metrics, maps them to evidence `(r, s)` via BPQ, and produces a **BinomialOpinion** `ω = (b, d, u, a)` where:
- `b` = **belief** (evidence of quality)
- `d` = **disbelief** (evidence of poor quality)
- `u` = **uncertainty** (lack of evidence)
- `a` = base rate (prior, default 0.5)
- `E[ω] = b + a·u` = projected probability of trustworthiness

The four stage opinions are then combined via the **Subjective Logic conjunction** operator `ω_s ∧ ω_c ∧ ω_a ∧ ω_p` to yield a single final dataset reliability opinion.

---

## Datasets

| Dataset | Description | File |
|---|---|---|
| **COMPAS** | Recidivism risk scores (Angwin et al., 2016) | `data/compas-scores-two-years.csv` |
| **CIFAR-10H** | Human annotation counts for CIFAR-10 images | `data/cifar10h-raw.csv`, `cifar10h-counts.npy` |
| **GTSRB** | German Traffic Sign Recognition Benchmark | Downloaded automatically via Keras |

---

## Repository Structure

```
.
├── main.py                      # Unified entry point — run any/all pipelines
├── sl.py                        # Subjective Logic opinion types (BinomialOpinion, MultinomialOpinion)
├── utils.py                     # BPQ evidence mapping (quality scores → evidence → opinions)
│
├── sampling.py                  # Stage 1: sampling quality metrics (DG, MCR, GRR, LTCI, DFI, MSA)
├── collection.py                # Stage 2: collection quality metrics (ORR, MDD, PIID, SALS, RRS)
├── annotation.py                # Stage 3: annotation quality metrics (IAA, GLBI, LCS, OLR, USA, ADT)
├── processing.py                # Stage 4: processing quality metrics (DIOR, FLCA, PCCR, FLR)
│
├── compas.py                    # Full four-stage pipeline for COMPAS
├── cifar_10h.py                 # Full four-stage pipeline for CIFAR-10H
├── cifar10h_labeling.py         # Annotation-level label analysis for CIFAR-10H
├── gtsrb.py                     # Class-balance opinion analysis for GTSRB (+ optional CNN training)
│
├── balance_opinion.py           # VC-dimension-based class-balance opinion computation
├── compas_balance.py            # Class-balance opinion for COMPAS
├── cifar10h_balance.py          # Class-balance opinion for CIFAR-10H
│
├── data/
│   ├── compas-scores-two-years.csv
│   └── cifar10h-raw.csv
│
├── compas_reliability_trace.csv     # Output: full metric trace for COMPAS
└── cifar10h_reliability_trace.csv   # Output: full metric trace for CIFAR-10H
```

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn matplotlib tensorflow
```

> **Note:** TensorFlow is only required for the GTSRB pipeline (`--pipeline gtsrb`).

---

## Usage

```bash
# Run all pipelines
python main.py

# Run a specific pipeline
python main.py --pipeline compas           # Four-stage COMPAS reliability trace
python main.py --pipeline cifar10h         # CIFAR-10H annotation + processing
python main.py --pipeline cifar10h_labeling  # Annotation-count opinion analysis
python main.py --pipeline gtsrb            # GTSRB class-balance opinion
python main.py --pipeline compas_balance   # COMPAS class-balance opinion
python main.py --pipeline cifar10h_balance # CIFAR-10H class-balance opinion

# GTSRB optional flags
python main.py --pipeline gtsrb --train          # Train CNN before analysis
python main.py --pipeline gtsrb --collaborative  # Federated bias sweep
```

---

## Pipeline: Four-Stage Reliability Assessment

### Stage 1 — Sampling

Measures how well the data collection *process* covered the target population.

| Metric | Symbol | Description |
|---|---|---|
| Distribution Gap | DG | KL divergence between dataset and reference population |
| Minority Coverage Rate | MCR | Dataset frequency / population frequency for a minority group |
| Group Representation Ratio | GRR | General version of MCR for any group |
| Long-Tail Coverage Index | LTCI | Fraction of rare categories present in the dataset |
| Data Freshness Index | DFI | Mean fractional age of records relative to a freshness window |
| Missing-at-Source Audit | MSA | Fraction of records missing at least one essential field |

### Stage 2 — Collection / Measurement

Measures data integrity and privacy risks.

| Metric | Symbol | Description |
|---|---|---|
| Out-of-Range Rate | ORR | Fraction of values outside valid bounds |
| Missing Data Disparity | MDD | Max − min missingness rate across demographic groups |
| PII Density | PIID | Ratio of PII columns to total columns |
| Sensitive Attribute Leakage Score | SALS | Logistic regression accuracy predicting a sensitive attribute from features |
| Re-identification Risk Score | RRS | Mean inverse anonymity set size (k-anonymity framework) |

### Stage 3 — Annotation

Measures labelling quality, consistency, and bias.

| Metric | Symbol | Description |
|---|---|---|
| Inter-Annotator Agreement | IAA | Percentage agreement between two annotators |
| Group Label Bias Index | GLBI | KL divergence between label distributions across demographic groups |
| Label Consistency Score | LCS | Fraction of labels agreeing across two annotation sources |
| Outlier Label Rate | OLR | Fraction of annotations disagreeing with a reference label |
| Uncertainty Score from Annotators | USA | Mean normalised Shannon entropy of per-item label distributions |
| Annotation Drift Over Time | ADT | KL divergence between label distributions in early vs. late annotations |

### Stage 4 — Processing

Measures data integrity issues introduced during cleaning and feature engineering.

| Metric | Symbol | Description |
|---|---|---|
| Duplicate Instance Overlap Rate | DIOR | Fraction of exact-duplicate rows |
| Feature–Label Correlation Audit | FLCA | Max absolute Pearson correlation between any feature and the label |
| Pre/Post Cleaning Change Ratio | PCCR | Fraction of instances removed during cleaning |
| Feature Loss Rate | FLR | Fraction of features dropped during processing |

---

## Subjective Logic Foundation

The core opinion algebra is implemented in [sl.py](sl.py):

- **`BinomialOpinion(b, d, u, a)`** — opinion over a binary proposition.
  - `conjunction(other)` — implements Jøsang (2016) Definition 14.4.
  - `from_beta(α, β)` — maps a Beta distribution to an opinion.
  - `expectation()` — projected probability `E[ω] = b + a·u`.

- **`MultinomialOpinion(belief, u, base_rate)`** — opinion over K outcomes.
  - `from_dirichlet(α)` — maps a Dirichlet distribution to an opinion.

Evidence mapping is in [utils.py](utils.py):

- `lower_is_better(x, T)` / `higher_is_better(x, A, L)` — quality score functions.
- `metric_to_evidence(q, K)` — maps quality `q ∈ [0, 1]` to BPQ evidence `(r, s)`.
- `bpq(r, s, W)` — Binomial Probability Quantification → `(b, d, u)` opinion triple.

---

## Class-Balance Opinion

[balance_opinion.py](balance_opinion.py) provides a VC-dimension-based opinion on dataset class balance:

- Uncertainty `u` is derived from the gap between the dataset size `N` and the PAC learning sample bound `N_req = (d/ε) · log(1/δ)`.
- Belief/disbelief are allocated based on how many classes fall within a tolerance `ε` of the uniform distribution `1/K`.
- The `plot_epsilon_sweep()` function visualises how the opinion evolves as `ε` varies.

---

## Output

Each pipeline writes a **reliability trace CSV** with one row per metric and per stage:

| Column | Description |
|---|---|
| `kind` | `metric`, `stage`, or `final` |
| `stage` | `sampling`, `collection`, `annotation`, `processing`, `final_conjunction` |
| `name` | Metric or opinion name |
| `raw_value` | Original metric value |
| `q_value` | Normalised quality score ∈ [0, 1] |
| `direction` | `lower_is_better` or `higher_is_better` |
| `hyperparameters` | Thresholds and config used |
| `K`, `r`, `s` | Evidence parameters |
| `b`, `d`, `u` | Resulting opinion triple |

---

## Reference

> Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty*. Springer.

> Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). Machine Bias. *ProPublica*.

---

## Citation

If you use this code in your research, please cite the accompanying dissertation:

```bibtex
@phdthesis{ouattara2026trustworthiness,
  author = {Ouattara, Koffi Ismael},
  title  = {Trust Assessment and Propagation in Neural Network},
  year   = {2025},
}
```
