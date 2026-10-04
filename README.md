<div align="center">

# 📊 Predictive Analytics Decision Suite

### End-to-End Machine Learning Classification & Streamlit Cloud Deployment

**Interactive decision-support applications for Customer Churn Prediction and NYC Airbnb Room-Type Classification.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-Data_Processing-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud_Deployment-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

Predictive Analytics Decision Suite is a collection of end-to-end machine learning applications for practical classification problems.

The repository includes two deployed Streamlit applications:

- **ChurnGuard** — predicts customer churn risk.
- **StayType AI** — predicts NYC Airbnb room type.

Both projects demonstrate a complete machine learning workflow, from data preparation and model selection to performance evaluation and interactive deployment.

```text
Raw Data → Cleaning & Preprocessing → Model Training → Optimization
→ Evaluation → Saved Pipeline → Streamlit Deployment
```

---

## Live Applications

| Application | Task | Live Demo |
|---|---|---|
| 📉 **ChurnGuard** | Binary Customer Churn Classification | [Launch Application](https://customer-churn-vhmvtn5oqq5sfqyjyijj2z.streamlit.app/) |
| 🏠 **StayType AI** | Multiclass NYC Airbnb Room-Type Classification | [Launch Application](https://machine-learning-model-ipfxzaz7jrahcapvwxhifm.streamlit.app/) |

---

## Projects

| Project | Classification Type | Objective |
|---|---|---|
| 📉 **ChurnGuard** | Binary Classification | Predict whether a customer is likely to churn |
| 🏠 **StayType AI** | Multiclass Classification | Predict whether an NYC Airbnb listing is an Entire home/apt, Private room, or Shared room |

---

## 📉 ChurnGuard

ChurnGuard is a binary classification application that estimates customer churn probability using customer age, gender, tenure, and monthly charges.

### Key Capabilities

- Data cleaning, validation, and categorical encoding.
- Stratified train/test splitting for reproducible evaluation.
- Model comparison across Logistic Regression, KNN, SVM, and Random Forest.
- Hyperparameter optimization using `GridSearchCV`.
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
- Real-time churn-risk prediction through an interactive Streamlit dashboard.
- Customer analytics and model-comparison visualizations.

### ML Workflow

```text
Customer Data
    ↓
Cleaning and Encoding
    ↓
Stratified Train/Test Split
    ↓
Preprocessing Pipelines
    ↓
GridSearchCV Optimization
    ↓
Model Comparison
    ↓
Best ROC-AUC Model
    ↓
Real-Time Churn Prediction
```

---

## 🏠 StayType AI

StayType AI is a calibrated multiclass classification application that predicts the likely room type of an NYC Airbnb listing.

### Target Classes

- Entire home/apt
- Private room
- Shared room

### Key Capabilities

- Leakage-safe feature engineering and preprocessing.
- Numeric and categorical imputation.
- One-hot encoding for categorical features.
- Optuna-based model selection between Random Forest and Histogram Gradient Boosting.
- Hyperparameter optimization using cross-validation.
- Isotonic probability calibration for more reliable confidence estimates.
- Decision-weight tuning to improve performance for the underrepresented Shared room class.
- Real-time prediction, class-probability visualization, confidence indicators, and evaluation dashboards.

### ML Workflow

```text
NYC Airbnb Listing Data
    ↓
Cleaning and Stratified Split
    ↓
Feature Engineering
    ↓
Imputation and One-Hot Encoding
    ↓
Optuna Model Optimization
    ↓
Random Forest vs Histogram Gradient Boosting
    ↓
Isotonic Probability Calibration
    ↓
Minority-Class Decision Adjustment
    ↓
Real-Time Room-Type Prediction
```

---

## Evaluation

The projects use multiple evaluation metrics rather than relying on accuracy alone.

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Reliability of positive predictions |
| Recall | Ability to identify relevant positive cases |
| F1-score | Balance between precision and recall |
| ROC-AUC | Classification ranking performance for churn prediction |
| Per-class F1 | Individual class performance for room-type classification |
| Confusion Matrix | Prediction error analysis across classes |
| Brier Score | Probability-calibration quality for StayType AI |

---

## Technology Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Hyperparameter Optimization | `GridSearchCV`, Optuna |
| Visualization | Plotly, Streamlit Charts |
| Model Persistence | Joblib |
| Application Framework | Streamlit |
| Deployment | Streamlit Community Cloud |
| Dependency Management | `uv`, `pyproject.toml`, `uv.lock`, `requirements.txt` |

---

## Project Structure

```text
machine-learning-model/
├── customer_churn/             # ChurnGuard application and dataset
├── NYC_Airbnb_Room/            # StayType AI pipeline, model, and reports
├── ui.py                       # Streamlit entry point
├── pyproject.toml              # Project configuration
├── uv.lock                     # Locked dependency versions
├── requirements.txt            # pip-compatible dependencies
├── LICENSE
└── README.md
```

---

## Quick Start

### Clone the repository

```bash
git clone [https://github.com/rj1230/machine-learning-model.git](https://github.com/rj1230/machine-learning-model.git)
cd machine-learning-model
```

### Install dependencies

Using `uv`:

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run ui.py
```

---

## Responsible Use

The applications provide machine-learning estimates based on historical data.

- Churn predictions are not guarantees of customer behavior.
- Airbnb room-type predictions are not verified listing attributes.
- Predictions should not be used as the sole basis for high-impact business, housing, or automated decisions.
- Low-confidence outputs should be reviewed by a human before action is taken.

---

## Key Learning Outcomes

This repository demonstrates practical experience with:

- Binary and multiclass machine learning classification.
- Model comparison and hyperparameter optimization.
- Reproducible preprocessing and model pipelines.
- Probability-based predictions and calibration.
- Minority-class performance considerations.
- Interactive ML application development.
- Streamlit Cloud deployment of machine learning systems.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built by [@rj1230](https://github.com/rj1230) ·  
[ChurnGuard Demo](https://customer-churn-vhmvtn5oqq5sfqyjyijj2z.streamlit.app/) ·  
[StayType AI Demo](https://machine-learning-model-ipfxzaz7jrahcapvwxhifm.streamlit.app/)

</div>
