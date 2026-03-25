# Automated Brugada Syndrome Detection from 12-Lead ECG
### HybridECGNet — 1D-CNN + XGBoost Fusion with Grad-CAM & SHAP Interpretability

---

## Overview

Brugada syndrome (BrS) accounts for up to 20% of sudden cardiac deaths in structurally normal hearts, yet remains systematically under-diagnosed. This project implements an end-to-end automated detection pipeline trained on the **Brugada-HUCA** dataset — 356 expert-adjudicated 12-lead ECG recordings from Hospital Universitario Central de Asturias.

The pipeline fuses deep waveform embeddings (1D-CNN) with clinician-interpretable ECG features (ST elevation, QRS duration, T-wave inversion) via an XGBoost classifier, then provides dual-layer interpretability through **1D Grad-CAM** and **SHAP TreeExplainer**.

---

## Results

| Metric | Value |
|---|---|
| CNN PR-AUC (15-fold CV, mean ± std) | **0.802 ± 0.088** (4.2× random baseline) |
| CNN ROC-AUC (15-fold CV, mean) | 0.909 ± 0.045 |
| Fusion PR-AUC (held-out val set) | **0.840** |
| Fusion ROC-AUC | 0.847 |
| Brugada F1 @ 90% precision | 0.783 |
| Overall accuracy | 93.1% |

---

## Architecture

```
12-Lead ECG (12 × 1200)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  [ CNN Branch ]               [ Clinical Branch ]
  Conv1d × 3 blocks            NeuroKit2 delineation
  (12→32→64→128 ch)            on V1, V2, V3 leads
  BatchNorm + ReLU             ──────────────────
  MaxPool → AdaptiveAvgPool    • Mean HR
  128-dim embedding            • QRS Duration (ms)
                               • J-point ST Elevation (mV)
                               • T-wave Inversion Ratio
        │                              │
        └──────────────┬───────────────┘
                       ▼
             [ 140-dim Fused Vector ]
                       │
                  [ XGBoost ]
                       │
             Brugada / Normal
```

---

## Dataset

**Brugada-HUCA** — available on PhysioNet:
> Costa Cortez, N., & Garcia Iglesias, D. (2026). *Brugada-HUCA: 12-Lead ECG Recordings for the Study of Brugada Syndrome* (v1.0.0). PhysioNet. https://doi.org/10.13026/0m2w-dy83

| Property | Value |
|---|---|
| Total usable recordings | 356 (7 atypical label=2 excluded) |
| Class split | 287 Normal : 69 Brugada (4.16:1 imbalance) |
| Sampling rate | 100 Hz, 12 seconds |
| Annotation | Expert electrophysiologist consensus |

Set `BASE_DIR` in the notebook to your local dataset path before running.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU**: PyTorch will automatically use CUDA if available. For CPU-only training, the default `torch` install is sufficient but training will be slower.

---

## Usage

Open and run `idsc-brugada-huca-final.ipynb` top-to-bottom. The notebook is self-contained and divided into the following stages:

| Stage | Cells | Description |
|---|---|---|
| **Setup** | 1–3 | Install packages, imports, seeds, device config |
| **Data Loading** | 4–5 | Load metadata, filter atypical cases |
| **EDA** | 6–7 | Class distribution, ECG visualisation |
| **Preprocessing & Augmentation** | 8–9 | Dataset class, 4-way stochastic augmentation |
| **Model Definition** | 10–14 | `FocalLoss`, `HybridECGNet`, Grad-CAM hooks |
| **Cross-Validation** | 15–16 | 15-fold Repeated Stratified K-Fold CNN training |
| **Clinical Feature Extraction** | 17–20 | NeuroKit2 delineation on V1–V3 |
| **Fusion Model** | 21–23 | CNN embeddings + clinical features → XGBoost |
| **Interpretability** | 24–28 | 1D Grad-CAM overlays, SHAP beeswarm & waterfall |

---

## Key Design Decisions

**Focal Loss** — addresses 4.16:1 class imbalance by downweighting easy negatives and focusing gradient signal on hard Brugada samples (α = positive class proportion per fold, γ = 2).

**Repeated Stratified K-Fold (5×3 = 15 folds)** — with only 69 positive cases, a single holdout split yields ~14 Brugada validation samples. Repetition reduces variance in performance estimates.

**PR-AUC as primary metric** — at 19.4% prevalence, a random classifier achieves PR-AUC ≈ 0.19 vs ROC-AUC ≈ 0.50. PR-AUC is a more honest measure of minority-class discrimination.

**Recall-prioritised thresholding** — a missed Brugada case (false negative) carries higher clinical cost than a false positive. The optimal threshold targets ≥90% sensitivity over F1 maximisation.

---

## Interpretability

### 1D Grad-CAM
Temporal saliency maps highlight which time steps drove the Brugada prediction. For confirmed BrS patients, peak attention is expected over the ST-segment and J-point in V1–V3 — the same region cardiologists examine.

### SHAP TreeExplainer
- **Global beeswarm plot** — ranks all 140 features (128 CNN embeddings + 12 clinical) by mean absolute SHAP impact across the validation set.
- **Per-patient waterfall plot** — decomposes any individual prediction into additive feature contributions from the base rate.

CNN embedding dimensions dominate the top features, confirming that HybridECGNet learned complex morphological signatures beyond what standard diagnostic thresholds capture.

---

## Limitations

| Limitation | Impact | Proposed Resolution |
|---|---|---|
| n=356, single centre | Wide confidence intervals | Multi-centre data; transfer from PTB-XL |
| Brugada recall = 64.3% | 5/14 positive cases missed | Calibrate threshold to ≥90% sensitivity |
| Threshold instability (0.26–0.74 across folds) | No stable deployment point | Fixed-recall threshold on a held-out calibration set |
| 100 Hz sampling rate | J-wave fine structure may be aliased | Re-acquire at 500 Hz |
| No age/sex metadata | Cannot control confounders (BrS is 8:1 male) | Request de-identified demographics |

---

## References

- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
- Antzelevitch et al. (2005). *Brugada Syndrome: Second Consensus.* Circulation.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
- Goldberger et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation.

---

## Deployment Scope

> ⚠️ This pipeline is designed as a **screening flag**, not an autonomous diagnostic system. A positive prediction should trigger specialist review — it does not replace an electrophysiologist.
