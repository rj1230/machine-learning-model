<div align="center">

# 🏠 StayType AI

### NYC Airbnb Room-Type Intelligence

**An Optuna-tuned, calibrated multiclass classification system that estimates whether an NYC Airbnb listing is most likely an Entire home/apt, Private room, or Shared room.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live App](YOUR_DEPLOYED_APP_URL) · [Model Pipeline](#-modeling-approach) · [Evaluation](#-model-evaluation) · [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

**StayType AI** is an end-to-end machine-learning decision-support application for NYC Airbnb listings. Given listing attributes such as location, price, availability, review activity, and host characteristics, it predicts the most likely room category and presents a calibrated probability distribution across all classes.

The project is designed around three principles:

- **Reproducible modeling** — preprocessing, feature engineering, model selection, calibration, and decision adjustment are packaged in a single serialized pipeline.
- **Leakage-safe validation** — transformations and feature engineering are fitted only on training folds during cross-validation.
- **Interpretable deployment** — the interface exposes confidence, class probabilities, calibration quality, and low-confidence warnings rather than presenting a black-box prediction.

---

## 🎯 Business Problem

Airbnb room type is an important listing attribute, but it may be missing, inconsistently labelled, or require validation in downstream listing-intelligence workflows.

This project addresses that problem by estimating room type from observable listing characteristics:

| Predicted Class | Description |
|---|---|
| **Entire home/apt** | The listing represents an entire property or apartment |
| **Private room** | The listing represents a private room within a property |
| **Shared room** | The listing represents a shared sleeping arrangement |

The application is intended as a **decision-support tool**, not an automated system for verifying live Airbnb listings.

---

## ✨ Key Capabilities

### 🔮 Listing Classification

Users can enter a complete listing profile and receive:

- The most likely room type
- Calibrated prediction confidence
- Full probability distribution across all three classes
- A low-confidence warning when the decision boundary is ambiguous
- A structured record of the input profile used for prediction

### 📊 Model Performance Dashboard

The app dynamically loads metrics from `reports/metrics.json`, allowing the dashboard to reflect the latest successful training run.

Displayed metrics include:

- Test accuracy
- Test Macro-F1
- Best cross-validation Macro-F1
- Mean Brier score
- Per-class F1 scores
- Brier scores by class
- Tuned decision weights
- Best hyperparameters

### 🧠 Pipeline Transparency

A dedicated dashboard explains the complete modeling workflow, including feature engineering, preprocessing, model selection, calibration, and cost-sensitive decision adjustment.

### 🛡️ Responsible AI Controls

The system explicitly distinguishes between model confidence and verified ground truth. Predictions with low confidence are flagged for human review.

---

## 🧠 Modeling Approach

The final artifact is not a raw classifier. It is a complete reproducible pipeline saved as:

```text
models/Best_Model_Pipeline.pkl
```

### Pipeline Workflow

```text
Raw NYC Airbnb data
        ↓
Cleaning and stratified train/test split
        ↓
Leakage-safe feature engineering
        ↓
Numeric and categorical imputation
        ↓
One-hot encoding
        ↓
Optuna model selection
        ↓
Isotonic probability calibration
        ↓
Tuned class decision weights
        ↓
Final evaluation on held-out test set
```

### Modeling Components

| Component | Implementation |
|---|---|
| Feature engineering | Leakage-safe ratios, log transforms, capping, and binary indicators |
| Numeric preprocessing | Median imputation |
| Categorical preprocessing | Mode imputation and one-hot encoding |
| Candidate models | Random Forest and Histogram Gradient Boosting |
| Hyperparameter optimization | Optuna with pruned 3-fold cross-validation |
| Probability calibration | Isotonic calibration using out-of-fold predictions |
| Rare-class handling | Tuned class decision weights for Shared room |
| Final evaluation | Untouched stratified test set |

### Why Calibration Matters

The application does not only show the predicted label. It also displays the probability assigned to each room type.

Isotonic calibration improves the reliability of these confidence estimates, which is important because a prediction with 52% confidence should not be communicated with the same certainty as one with 88% confidence.

### Why Decision Weights Matter

Shared rooms are the rarest class in the dataset. A model optimized only for overall accuracy can achieve strong aggregate performance while performing poorly on this minority class.

The pipeline tunes decision weights using out-of-fold predictions to improve Shared-room recall without altering the underlying calibrated probability estimates.

---

## 📈 Model Evaluation

The dashboard reports both discriminative performance and probability quality.

| Metric | Purpose |
|---|---|
| **Test Accuracy** | Overall share of correctly classified listings |
| **Macro-F1** | Equal-weighted performance across all room types |
| **Per-class F1** | Class-specific performance, especially for rare classes |
| **Brier Score** | Calibration quality; lower is better |
| **Cross-validation Macro-F1** | Model-selection performance during Optuna tuning |

Macro-F1 is particularly important because it prevents strong performance on majority classes from masking weak performance on Shared rooms.

---

## 🧩 Input Features

| Feature | Purpose |
|---|---|
| `neighbourhood_group` | NYC borough context |
| `neighbourhood` | Local neighbourhood context |
| `latitude` | Geographic location |
| `longitude` | Geographic location |
| `price` | Nightly listing price |
| `minimum_nights` | Minimum stay requirement |
| `number_of_reviews` | Historical review volume |
| `reviews_per_month` | Review activity rate |
| `calculated_host_listings_count` | Host portfolio size |
| `availability_365` | Future listing availability |

---

## 🖥️ Application Interface

The Streamlit application contains three main views:

| Tab | Purpose |
|---|---|
| **🔮 Predict Room Type** | Submit listing details and inspect predictions, confidence, and probabilities |
| **📊 Model Performance** | Review test metrics, per-class F1, calibration, and decision policy |
| **🧠 Pipeline Intelligence** | Understand the full training and inference architecture |

The interface also maintains a session-level prediction history for quick review of recent classifications.

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or later
- `uv` or `pip`
- Trained model artifact at `models/Best_Model_Pipeline.pkl`

### Option 1: Using `uv`

```bash
# Install uv if required
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Install dependencies
uv sync
```

### Option 2: Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Train the model

If the model or metrics report does not exist, run:

```bash
uv run python ./NYC_Airbnb_Room/train_model.py
```

This generates:

```text
models/Best_Model_Pipeline.pkl
reports/metrics.json
```

### 2. Launch the Streamlit app

```bash
uv run streamlit run app.py
```

Or, with pip:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
.
├── app.py                         # Streamlit decision-support application
├── train_model.py                 # Training, tuning, calibration, and evaluation
├── nyc_features.py                # Custom pipeline and feature-engineering classes
├── models/
│   └── Best_Model_Pipeline.pkl    # Serialized end-to-end inference pipeline
├── reports/
│   └── metrics.json               # Model-selection and test-evaluation metrics
├── requirements.txt
├── LICENSE
└── README.md
```

> Update the paths above if your final repository layout differs.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Application | Streamlit, Plotly Express |
| Data processing | pandas |
| Machine learning | scikit-learn |
| Model tuning | Optuna |
| Model serialization | joblib |
| Core models | Random Forest, Histogram Gradient Boosting |
| Calibration | Isotonic regression |

---

## ⚠️ Responsible Use

This project predicts room-type patterns from historical listing attributes. Its outputs are probabilistic estimates, not verified Airbnb listing attributes.

The system should not be used as the sole basis for:

- Regulatory or compliance decisions
- Housing, rental, or pricing decisions
- High-impact automated classification
- Claims about the actual room type of a live listing

The application displays confidence levels and warns users when the top predicted probability is below 55%, indicating that manual review is recommended.

---

## 🔭 Future Improvements

Potential production-oriented upgrades include:

- Automated data-quality and schema-validation checks
- Drift monitoring for price, availability, and neighbourhood distributions
- SHAP-based local explanation for individual predictions
- Confidence-threshold routing to human review
- Model registry and versioned artifact tracking
- CI/CD pipeline for tests, training reproducibility, and deployment
- Containerized deployment with Docker

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Built by [cric_](https://github.com/YOUR_GITHUB_USERNAME)**  
*Machine-learning systems should be measurable, interpretable, and responsibly deployed.*

</div>
