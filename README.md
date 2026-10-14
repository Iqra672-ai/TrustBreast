# TrustBreast

Code for **"TrustBreast: A Clinically Complete, Explainable, and Uncertainty-Aware Ensemble Framework
for Breast Cancer Diagnosis on the Wisconsin Diagnostic Dataset"**.

## Description

A leakage-free soft-voting ensemble (random forest + XGBoost + deep network) on the Wisconsin
Diagnostic Breast Cancer dataset (569 patients, 30 FNA features), with SHAP and LIME attributions,
Monte Carlo Dropout and conformal uncertainty, DiCE counterfactuals, and a measurement of
SMOTE-before-partitioning leakage on five clinical cohorts.

## Code Information

| Folder / file | Contents |
|---|---|
| `notebooks/` | Eight Jupyter notebooks (00–07) that reproduce every table and figure in the paper |
| `scripts/` | Stand-alone Python scripts that draw Figures 1, 9 and 10 |
| `models/TrustBreast_locked/` | The trained (locked) ensemble used for the held-out analyses |
| `results/` | Saved outputs: per-patient MC-Dropout results, reliability-diagram data, all DiCE counterfactuals |
| `data/` | Copy of the Breast Cancer Coimbra dataset |
| `requirements.txt` | Exact library versions |

## Requirements

- Python 3.13 (Google Colab CPU runtime; no GPU needed)
- Exact library versions are in `requirements.txt`; install with `pip install -r requirements.txt`
- Main packages: TensorFlow 2.20.0, Keras 3.13.2, scikit-learn 1.6.1, XGBoost 3.4.1, imbalanced-learn 0.14.2, SHAP 0.52.0, LIME 0.2.0.1, dice-ml 0.12, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3

## Quick start — reproduce the paper in three clicks

1. Click an **Open in Colab** button below (a Google account is needed; nothing else to install).
2. In Colab keep the default **CPU** runtime: **Runtime → Change runtime type → CPU → Save**.
3. **Runtime → Run all**. If Colab warns *"This notebook was not authored by Google"*, click **Run anyway**.
   The first cell downloads this repository (code, locked model, data) and installs the exact library versions; no Google Drive access is needed.
4. Scroll to the last cell (**RESULTS SUMMARY**). Every quantity is compared with the value in the paper and marked ✅ (match) or ❌ (different).

| Notebook | Reproduces | Time (CPU) |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Iqra672-ai/TrustBreast/blob/main/notebooks/03_objective3_uncertainty.ipynb) `03_objective3_uncertainty.ipynb` | MC Dropout, conformal prediction — Tables 9, 11, 13; Figs 9–12 | ~5–10 min (fastest check) |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Iqra672-ai/TrustBreast/blob/main/notebooks/02_objective2_shap_lime.ipynb) `02_objective2_shap_lime.ipynb` | SHAP and LIME — Table 7; Figs 6–8 | ~30–45 min |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Iqra672-ai/TrustBreast/blob/main/notebooks/04_objective4_dice.ipynb) `04_objective4_dice.ipynb` | DiCE counterfactuals — Tables 14, 15; Figs 13–14 | ~10–20 min |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Iqra672-ai/TrustBreast/blob/main/notebooks/01_objective1_ensemble.ipynb) `01_objective1_ensemble.ipynb` | Held-out results, CV, leakage, repeated splits, baselines, calibration — Tables 2, 3, 5, 6, 8, 10, 12; Figs 3–5 | ~3–4 h |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Iqra672-ai/TrustBreast/blob/main/notebooks/05_leakage_five_cohorts.ipynb) `05_leakage_five_cohorts.ipynb` | Leakage on five cohorts — Table 4 | ~30–50 min |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Iqra672-ai/TrustBreast/blob/main/notebooks/06_coimbra_portability.ipynb) `06_coimbra_portability.ipynb` | Coimbra portability — Section 4.10 | ~5–15 min |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Iqra672-ai/TrustBreast/blob/main/notebooks/07_crossconformal_out_of_fold.ipynb) `07_crossconformal_out_of_fold.ipynb` | Held-out-fold cross-conformal coverage (Table 12, Fig. 10, §4.10) and errors inside the escalation set | ~20–30 min |

Notebook 04 uses `O3_all_patients.csv` from notebook 03; a copy is already in `results/`, so the notebooks can be run in any order.

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

## Usage Instructions

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
| `notebooks/07_crossconformal_out_of_fold.ipynb` | Table 12, Fig. 10 (cross-conformal coverage on held-out folds) | ~20–30 min |
| `notebooks/00_train_locked_model_OPTIONAL.ipynb` | how the locked model was trained (not needed to reproduce) | ~10 min |

Notebook 04 reads `O3_all_patients.csv`, written by notebook 03 (a copy is in `results/`).

## Figure scripts

Figures 1, 9 and 10 are drawn by small stand-alone scripts (no model or notebook state needed):

| Script | Figure | Input |
|---|---|---|
| `scripts/make_fig1_framework.py` | Fig. 1, framework overview | none (values as reported in the paper) |
| `scripts/make_fig9_mc.py` | Fig. 9, MC-Dropout uncertainty | `results/O3_all_patients.csv` (written by notebook 03) |
| `scripts/make_fig10_calibration.py` | Fig. 10, reliability diagram and conformal coverage | `results/O3_reliability_diagram.csv` (written by notebook 03); coverage values from notebooks 03 and 07 |

Run from the repository root, e.g. `python scripts/make_fig10_calibration.py`; output is written to `figures/`.
All other figures are produced by the notebooks listed above.

## Methodology

1. **Preprocessing:** stratified 80/20 split (seed 42); Min–Max scaling fitted on the training data only; a 91-patient validation subset carved out before oversampling; SMOTE and Gaussian-noise augmentation applied to the training data only.
2. **Objective 1 – Ensemble:** random forest, XGBoost and a deep neural network combined by soft voting; decision thresholds tuned on the validation subset; leakage-free 10-fold cross-validation and 30 repeated splits.
3. **Objective 2 – Explainability:** SHAP (normalized ensemble average) and LIME, compared with Spearman rank correlation.
4. **Objective 3 – Uncertainty:** Monte Carlo Dropout (100 passes, reproducible masks), split-conformal and cross-conformal prediction.
5. **Objective 4 – Counterfactuals:** DiCE counterfactuals for the high-uncertainty patients, with plausibility checks.
6. **Additional analyses:** SMOTE-leakage measurement on five cohorts and a portability check on the Coimbra cohort.

## Key results

| Quantity | Value |
|---|---|
| Held-out split (seed 42, 114 patients) | 99.12% accuracy, AUC 0.9997 |
| 30 repeated 80/20 splits (representative) | 97.19 ± 1.65% |
| Leakage-corrected 10-fold CV | 96.49 ± 2.74%, pooled AUC 0.9945 |
| L2 logistic regression, same protocol | 97.48 ± 2.20% (no significant difference) |
| SHAP–LIME agreement, top-10 | Spearman ρ = 0.95, 8/10 overlap |
| MC-Dropout escalation set (std > 0.15) | 12/114 patients, contains every error |
| Cross-conformal coverage (α = 0.05, held-out folds) | 95.08% marginal, 94.81% malignant |
| SMOTE-before-partitioning inflation, 5 cohorts | +0.24 to +3.67 points, significant on none |

## Dataset Information

- **Wisconsin Diagnostic Breast Cancer (WBCD/WDBC):** 569 patients (357 benign, 212 malignant), 30 continuous features computed from digitized fine-needle aspiration (FNA) images, plus an ID and the diagnosis label. Source: UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B. Downloaded automatically by the notebooks.
- **Breast Cancer Coimbra:** 116 participants (64 patients, 52 healthy controls), 9 clinical features. Source: UCI Machine Learning Repository, https://doi.org/10.24432/C52P59. A copy is in `data/coimbra_dataR2.csv`.
- **Four leakage-experiment cohorts (notebook 05):** fetched from OpenML (`diabetes`, `haberman`, `blood-transfusion-service-center`, `sick`; version 1).

## Citations

If you use this code, please cite the paper (full details will be added after publication):

Mushtaq I, Shahzad MU, Othman KM, Rehman HU, Yildirim Y, Gaid ASA. TrustBreast: A Clinically Complete, Explainable, and Uncertainty-Aware Ensemble Framework for Breast Cancer Diagnosis on the Wisconsin Diagnostic Dataset. Under review.

Datasets:
- Wolberg W, Mangasarian O, Street N, Street W. Breast Cancer Wisconsin (Diagnostic) [dataset]. UCI Machine Learning Repository; 1993. https://doi.org/10.24432/C5DW2B
- Patrício M, Pereira J, Crisóstomo J, Matafome P, Seiça R, Caramelo F. Breast Cancer Coimbra [dataset]. UCI Machine Learning Repository; 2018. https://doi.org/10.24432/C52P59
- Vanschoren J, van Rijn JN, Bischl B, Torgo L. OpenML: networked science in machine learning. SIGKDD Explorations. 2014;15(2):49–60.

## License

This project is released under the MIT License (see `LICENSE`).
