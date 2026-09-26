from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard | Customer Risk Intelligence", page_icon="📉", layout="wide"
)
BASE = Path(__file__).resolve().parent
METRICS = BASE / "reports" / "metrics.json"
API = os.getenv("CHURN_API_URL", "http://localhost:8000").rstrip("/")
st.markdown(
    """<style>
.block-container{max-width:1450px;padding-top:1.8rem}
.hero{padding:2rem;border-radius:22px;background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;margin-bottom:1.3rem}
.hero h1{margin:0;font-size:2.4rem}
.hero p{color:#dbeafe;max-width:760px;line-height:1.6}
.card{padding:1.2rem;border:1px solid #e2e8f0;border-radius:16px;background:#fff;box-shadow:0 6px 18px #0f172a12}
.critical{border-left:6px solid #dc2626}
.high{border-left:6px solid #f59e0b}
.watch{border-left:6px solid #2563eb}
.low{border-left:6px solid #16a34a}
div[data-testid="stMetric"]{border:1px solid #e2e8f0;padding:1rem;border-radius:14px;background:#fff}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p{color:#475569 !important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#0f172a !important}
div[data-testid="stMetric"] [data-testid="stMetricDeltaValue"]{color:#16a34a !important}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def metrics():
    return json.loads(METRICS.read_text()) if METRICS.exists() else None


m = metrics()
st.markdown(
    """<div class='hero'><h1>📉 ChurnGuard</h1><p>Customer risk intelligence powered by calibrated machine learning. Estimate churn probability, prioritize retention actions, and inspect model quality.</p></div>""",
    unsafe_allow_html=True,
)
if not m:
    st.error("Training metrics not found. Run: python customer_churn/train_model.py")
    st.stop()
a, b, c, d = st.columns(4)
a.metric("Customers", f"{m['dataset_rows']:,}")
b.metric("Test ROC-AUC", f"{m['test_roc_auc']:.3f}")
c.metric("Test F1", f"{m['test_f1']:.3f}")
d.metric("Decision threshold", f"{m['decision_threshold']:.2f}")
pred, perf, insights = st.tabs(
    ["🔮 Predict Churn Risk", "📊 Model Performance", "🧠 Model Intelligence"]
)
with pred:
    st.subheader("Customer profile")
    with st.form("customer"):
        x, y, z = st.columns(3)
        with x:
            age = st.slider("Age", 18, 100, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.slider("Tenure (months)", 0, 120, 12)
        with y:
            monthly = st.number_input("Monthly charges", 0.0, 5000.0, 70.0)
            total = st.number_input("Total charges", 0.0, 100000.0, 840.0)
            contract = st.selectbox(
                "Contract type", ["Month-to-Month", "One Year", "Two Year"]
            )
        with z:
            internet = st.selectbox("Internet service", ["Fiber Optic", "DSL", "No"])
            support = st.selectbox("Tech support", ["Yes", "No"])
            submitted = st.form_submit_button(
                "Predict churn risk", type="primary", use_container_width=True
            )
    if submitted:
        payload = {
            "age": age,
            "gender": gender,
            "tenure": tenure,
            "monthly_charges": monthly,
            "contract_type": contract,
            "internet_service": internet,
            "total_charges": total,
            "tech_support": support,
        }
        try:
            r = requests.post(f"{API}/v1/predict", json=payload, timeout=30)
            r.raise_for_status()
            out = r.json()
            prob = out["churn_probability"]
            css = {
                "Critical": "critical",
                "High": "high",
                "Watchlist": "watch",
                "Low": "low",
            }[out["risk_band"]]
            st.markdown(
                f"<div class='card {css}'><h2>{out['risk_band']} churn risk</h2><p><b>{out['prediction']}</b><br>{out['recommended_action']}</p></div>",
                unsafe_allow_html=True,
            )
            q1, q2, q3 = st.columns(3)
            q1.metric("Churn probability", f"{prob:.1%}")
            q2.metric("Decision threshold", f"{out['decision_threshold']:.2f}")
            q3.metric("Model", out["model_name"])
            st.progress(prob)
        except requests.RequestException:
            st.error(
                "Cannot reach API. Run: uvicorn customer_churn.api.main:app --reload"
            )
with perf:
    st.subheader("Evaluation results")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Accuracy", f"{m['test_accuracy']:.2%}")
    p2.metric("Precision", f"{m['test_precision']:.3f}")
    p3.metric("Recall", f"{m['test_recall']:.3f}")
    p4.metric("Brier score", f"{m['mean_brier_score']:.4f}")
    compare = pd.DataFrame(m["cv_model_comparison"])
    st.dataframe(compare, use_container_width=True, hide_index=True)
    top = pd.DataFrame(m["top_features"])
    fig = px.bar(
        top.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top permutation feature importance",
    )
    st.plotly_chart(fig, use_container_width=True)
with insights:
    st.subheader("How ChurnGuard makes decisions")
    st.markdown("""**Pipeline:** validation → feature engineering → imputation and one-hot encoding → model selection → isotonic calibration → threshold tuning → retention recommendation.

**Why not a fixed 0.50 threshold?** Missing a likely churner can be more expensive than contacting a customer who stays. The threshold is selected from out-of-fold probabilities to balance precision and recall.

**Responsible use:** predictions are decision-support estimates, not guarantees. Review high-impact decisions and monitor model drift.""")
    st.json(
        {
            "selected_model": m["model_name"],
            "best_parameters": m["best_parameters"],
            "raw_features": m["raw_features"],
        }
    )
st.divider()
st.caption(
    "ChurnGuard · Calibrated customer churn prediction and retention decision support"
)
