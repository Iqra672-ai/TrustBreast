# TrustBreast

Code for **"TrustBreast: A Clinically Complete, Explainable, and Uncertainty-Aware Ensemble Framework
for Breast Cancer Diagnosis on the Wisconsin Diagnostic Dataset"** (Iqra Mushtaq, M. Umair Shahzad).

A leakage-free soft-voting ensemble (random forest + XGBoost + deep network) on the Wisconsin
Diagnostic Breast Cancer dataset (569 patients, 30 FNA features), with SHAP and LIME attributions,
Monte Carlo Dropout and conformal uncertainty, DiCE counterfactuals, and a measurement of
SMOTE-before-partitioning leakage on five clinical cohorts.

## Reproducibility

Every notebook ends with a **RESULTS SUMMARY** cell that compares its output with the numbers
reported in the paper and prints ✅ / ❌ for each quantity. All notebooks reproduce the paper
exactly under the setup below (verified in independent Colab sessions).

| | |
|---|---|
| Environment | Google Colab, Python 3.13, **CPU runtime** (the default; no GPU) |
| Libraries | exact versions in `requirements.txt` (TensorFlow 2.20.0, Keras 3.13.2, scikit-learn 1.6.1, XGBoost 3.4.1, SHAP 0.52.0, LIME 0.2.0.1, dice-ml 0.12) |
| Model | the locked, trained model in `models/TrustBreast_locked/` is **loaded, not re-trained**, by notebooks 01–04 |
| Seeds | fixed everywhere: data splits, SMOTE, all learners, per-fold network seeds (`keras.utils.set_random_seed`), TensorFlow op determinism, LIME generator reset, stateless MC-Dropout masks, per-patient DiCE seeds |

Results that re-train a neural network inside a resampling loop (cross-validation, repeated splits,
baselines, leakage experiment, deep ensemble) are exactly reproducible on the setup above; on
different hardware or library versions they may differ in the last decimal. Results computed from
the locked model (held-out metrics, SHAP, LIME, MC Dropout, conformal prediction, DiCE) do not
depend on re-training.

## How to run

Open each notebook in Google Colab and keep the default **CPU** runtime (**Runtime → Change runtime type → CPU**), then **Run all**.
The reported results were produced on the CPU runtime; a GPU runtime uses different floating-point kernels and can change re-trained results in the last decimal.
The first cell clones this repository and installs the pinned requirements.

| Notebook | Paper items | Approx. time (CPU) |
|---|---|---|
| `notebooks/01_objective1_ensemble.ipynb` | Tables 2, 3, 5, 6, 8, 10, 12; Figs 3–5 | ~3–4 h |
| `notebooks/02_objective2_shap_lime.ipynb` | Table 7; Figs 6–8 | ~30–45 min |
| `notebooks/03_objective3_uncertainty.ipynb` | Tables 9, 11, 13; Figs 9–12 | ~5–10 min |
| `notebooks/04_objective4_dice.ipynb` | Tables 14, 15; Figs 13–14 | ~10–20 min |
| `notebooks/05_leakage_five_cohorts.ipynb` | Table 4 (no model files needed) | ~30–50 min |
| `notebooks/06_coimbra_portability.ipynb` | Section 4.10, Coimbra portability check (no model files needed) | ~5–10 min |
| `notebooks/00_train_locked_model_OPTIONAL.ipynb` | how the locked model was trained (not needed to reproduce) | ~10 min |

Notebook 04 reads `O3_all_patients.csv`, written by notebook 03 (a copy is in `results/`).

## Key results

| Quantity | Value |
|---|---|
| Held-out split (seed 42, 114 patients) | 99.12% accuracy, AUC 0.9997 |
| 30 repeated 80/20 splits (representative) | 97.19 ± 1.65% |
| Leakage-corrected 10-fold CV | 96.49 ± 2.74%, pooled AUC 0.9945 |
| L2 logistic regression, same protocol | 97.48 ± 2.20% (no significant difference) |
| SHAP–LIME agreement, top-10 | Spearman ρ = 0.95, 8/10 overlap |
| MC-Dropout escalation set (std > 0.15) | 12/114 patients, contains every error |
| Cross-conformal coverage (α = 0.05) | 95.25% marginal, 95.75% malignant |
| SMOTE-before-partitioning inflation, 5 cohorts | +0.24 to +3.67 points, significant on none |

## Data

Wisconsin Diagnostic Breast Cancer (UCI Machine Learning Repository), downloaded by the notebooks.
The four additional cohorts of notebook 05 are fetched from OpenML (`diabetes`, `haberman`,
`blood-transfusion-service-center`, `sick`; version 1).

## Citation

(add after publication)
