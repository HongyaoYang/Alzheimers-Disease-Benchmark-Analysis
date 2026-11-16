# Alzheimer’s Disease Benchmark Analysis

BIO215 Capstone Project - classical machine-learning workflow for
Alzheimer’s disease classification using structured clinical data.

This repository contains the R-based part of the group project: data
preprocessing, seven-model benchmarking, independent hold-out evaluation,
model artifacts, performance visualisation, and LightGBM SHAP analysis.

> **Research-use disclaimer:** This is an educational project, not a validated
> clinical diagnostic system. The results must not be used for medical
> decision-making.

## Study overview

The pipeline compares:

- Logistic regression
- Elastic net
- Naive Bayes
- Random forest
- Support vector machine with an RBF kernel
- k-nearest neighbours
- LightGBM

The data are split using stratified sampling:

- 80% training data
- 20% independent hold-out test data
- Five-fold stratified cross-validation within the training set

`PatientID`, `DoctorInCharge`, and `FunctionalAssessment` are removed before
training. Pulse pressure is derived as:

```text
PulsePressure = SystolicBP - DiastolicBP
```

## Main results

The included independent-test leaderboard reports:

| Model | AUROC | AUPRC | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.873 | 0.811 | 0.870 | 0.691 | 0.807 |
| Random forest | 0.868 | 0.796 | 0.874 | 0.651 | 0.795 |
| Naive Bayes | 0.865 | 0.799 | 1.000 | 0.000 | 0.646 |
| Elastic net | 0.825 | 0.745 | 0.899 | 0.480 | 0.751 |
| Logistic regression | 0.822 | 0.748 | 0.834 | 0.572 | 0.741 |
| SVM | 0.813 | 0.741 | 0.848 | 0.572 | 0.751 |
| kNN | 0.682 | 0.535 | 0.874 | 0.322 | 0.678 |

These values are stored in `leaderboard_test_final.csv`. They describe this
specific split and should not be interpreted as external clinical validation.

The full group report also evaluates a Python TabNet model and describes a
Shiny application and an R package. Their source code is not present in this
folder, so they are not claimed as reproducible components of this repository.

## Repository structure

| File | Purpose |
|---|---|
| `ML_design.R` | Preprocessing, model training, cross-validation, test evaluation, and model export |
| `SHAP.R` | Load the exported LightGBM model and generate a SHAP summary visualisation |
| `performance.R` | Comparative performance figure including the report’s TabNet result |
| `performance_compare.R` | ROC and precision-recall comparison for LightGBM and elastic net |
| `alzheimers_disease_data.csv` | Public Alzheimer’s disease dataset used by the R pipeline |
| `leaderboard_train_cv.csv` | Five-fold training-set cross-validation results |
| `leaderboard_test_final.csv` | Independent hold-out results |
| `*.rds`, `*.txt` | Saved model artifacts |
| `*.png` | Generated result and interpretability figures |

## Run the analysis

### Requirements

- R 4.x
- The R packages listed at the top of `ML_design.R`

From the repository root:

```bash
Rscript ML_design.R
Rscript SHAP.R
```

The scripts install missing R packages automatically. LightGBM installation
can require a working C/C++ toolchain on some systems.

## Data source

Rabie El Kharoua (2024), *Alzheimer’s Disease Dataset*, Kaggle:

- Dataset page: <https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset>
- Dataset DOI: <https://doi.org/10.34740/KAGGLE/DSV/8668279>

The dataset contains 2,149 de-identified records with demographic, lifestyle,
clinical, cognitive, functional, and behavioural variables.

## Interpretation and limitations

- MMSE and ADL are prominent predictors in the included interpretability
  outputs, but model importance does not establish causality.
- Results come from a single public, cross-sectional dataset.
- No external or multi-centre validation is included.
- Threshold-dependent sensitivity and specificity must be selected for a
  predefined use case rather than chosen after inspecting test outcomes.
- The repository is an educational benchmark and not a clinical device.

## Attribution

Machine-learning modelling, analysis, and visualisation in this repository were
developed by Hongyao Yang as part of BIO215 at Xi’an
Jiaotong-Liverpool University.

The original report credits the wider group for the web application, R package,
writing, and discussion contributions. The report itself is not included here
because its cover contains student identifiers.
