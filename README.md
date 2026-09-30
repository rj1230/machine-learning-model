# 🤖 Machine Learning Models

**End-to-end machine learning projects built with Python and scikit-learn, with interactive Streamlit demos.**

Each project takes a real-world problem from raw data to a trained, evaluated model using classic machine learning techniques.

---

## 🔗 Live Demos

- 📉 **Customer Churn Prediction** — [customer-churn.streamlit.app](https://customer-churn-vhmvtn5oqq5sfqyjyijj2z.streamlit.app/)
- 🤖 **Machine Learning Models app** — [machine-learning-model.streamlit.app](https://machine-learning-model-ipfxzaz7jrahcapvwxhifm.streamlit.app/)

---

## 📦 Projects

| Project | Type | Goal | Folder |
|---|---|---|---|
| 📉 **Customer Churn Prediction** | Classification | Predict which customers are likely to leave | `customer_churn/` |
| 🏠 **NYC Airbnb Room Type Classification** | Classification | Classify NYC Airbnb listings by room type | `NYC_Airbnb_Room/` |

---

## ✨ What's Covered

- **Classification** — predicting categories such as churn vs. stay and Airbnb room type
- **Model comparison** — Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting compared on the Airbnb project
- **Hyperparameter tuning** — `RandomizedSearchCV`
- **Model persistence** — trained models saved with `joblib`
- **Interactive demos** — Streamlit apps deployed to Streamlit Cloud

**Flow:** Raw data → EDA & cleaning → feature engineering → model training → tuning & evaluation → saved model → Streamlit demo

---

## 🧩 Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Machine learning | scikit-learn |
| Data handling | pandas, NumPy |
| Model persistence | joblib |
| Demo UI | Streamlit |
| Dependencies | uv (`pyproject.toml` + `uv.lock`) and `requirements.txt` |

---

## 📂 Project Structure

```
machine-learning-model/
├── NYC_Airbnb_Room/     # NYC Airbnb room type classification
├── customer_churn/      # Customer churn prediction
├── ui.py                # Streamlit app
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/rj1230/machine-learning-model.git
cd machine-learning-model
pip install -r requirements.txt

# 2. Launch the Streamlit app
streamlit run ui.py
```

Each project lives in its own folder with its own data and code.
