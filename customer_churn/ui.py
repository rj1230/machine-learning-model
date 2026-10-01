from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


st.set_page_config(
    page_title="ChurnGuard | Customer Risk Intelligence",
    page_icon="📉",
    layout="wide",
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

# Local development:
# http://localhost:8000
#
# Streamlit Cloud:
# Add this value in App Settings -> Secrets after deploying FastAPI:
#
# CHURN_API_URL = "https://your-fastapi-api-url.example.com"
try:
    STREAMLIT_API_URL = st.secrets["CHURN_API_URL"]
except (KeyError, StreamlitSecretNotFoundError):
    STREAMLIT_API_URL = "http://localhost:8000"

API_URL = os.getenv(
    "CHURN_API_URL",
    STREAMLIT_API_URL,
).rstrip("/")


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: #ffffff;
        margin-bottom: 1.3rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
    }

    .hero p {
        color: #dbeafe;
        max-width: 760px;
        line-height: 1.6;
    }

    .card {
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 6px 18px #0f172a12;
    }

    .card h2 {
        color: #0f172a;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }

    .card p {
        color: #334155;
        margin: 0;
        line-height: 1.6;
    }

    .card p b {
        color: #0f172a;
    }

    .critical {
        border-left: 6px solid #dc2626;
    }

    .high {
        border-left: 6px solid #f59e0b;
    }

    .watch {
        border-left: 6px solid #2563eb;
    }

    .low {
        border-left: 6px solid #16a34a;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 14px;
        background: #ffffff;
    }

    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #475569 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDeltaValue"] {
        color: #16a34a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================


@st.cache_data
def load_metrics() -> dict | None:
    """Load metrics created by customer_churn/train_model.py."""
    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


metrics = load_metrics()


# ============================================================
# HEADER AND SUMMARY
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📉 ChurnGuard</h1>
        <p>
            Customer risk intelligence powered by calibrated machine learning.
            Estimate churn probability, prioritize retention actions, and inspect
            model quality.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if metrics is None:
    st.error(
        "Training metrics not found. Expected file: "
        "`customer_churn/reports/metrics.json`."
    )
    st.stop()


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Customers",
    f"{metrics['dataset_rows']:,}",
)

metric_2.metric(
    "Test ROC-AUC",
    f"{metrics['test_roc_auc']:.3f}",
)

metric_3.metric(
    "Test F1",
    f"{metrics['test_f1']:.3f}",
)

metric_4.metric(
    "Decision threshold",
    f"{metrics['decision_threshold']:.2f}",
)


prediction_tab, performance_tab, insights_tab = st.tabs(
    [
        "🔮 Predict Churn Risk",
        "📊 Model Performance",
        "🧠 Model Intelligence",
    ]
)


# ============================================================
# PREDICTION TAB
# ============================================================

with prediction_tab:
    st.subheader("Customer profile")

    with st.form("customer_profile_form"):
        column_1, column_2, column_3 = st.columns(3)

        with column_1:
            age = st.slider(
                "Age",
                min_value=18,
                max_value=100,
                value=35,
            )

            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
            )

            tenure = st.slider(
                "Tenure (months)",
                min_value=0,
                max_value=120,
                value=12,
            )

        with column_2:
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=5000.0,
                value=70.0,
            )

            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
            )

            contract_type = st.selectbox(
                "Contract type",
                options=[
                    "Month-to-Month",
                    "One Year",
                    "Two Year",
                ],
            )

        with column_3:
            internet_service = st.selectbox(
                "Internet service",
                options=[
                    "Fiber Optic",
                    "DSL",
                    "No",
                ],
            )

            tech_support = st.selectbox(
                "Tech support",
                options=["Yes", "No"],
            )

            submitted = st.form_submit_button(
                "Predict churn risk",
                type="primary",
                use_container_width=True,
            )

    if submitted:
        # This structure must match PredictionRequest in api/main.py.
        payload = {
            "customer": {
                "age": age,
                "gender": gender,
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "contract_type": contract_type,
                "internet_service": internet_service,
                "tech_support": tech_support,
            },
            "business": {
                "customer_lifetime_value": 500.0,
                "campaign_cost": 50.0,
                "campaign_success_rate": 0.30,
            },
            "threshold": float(metrics["decision_threshold"]),
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            response.raise_for_status()
            prediction_output = response.json()

            churn_probability = float(prediction_output["churn_probability"])

            risk_band = prediction_output["risk_band"]

            risk_class = {
                "Critical": "critical",
                "High": "high",
                "Watchlist": "watch",
                "Low": "low",
            }.get(risk_band, "watch")

            st.markdown(
                f"""
                <div class="card {risk_class}">
                    <h2>{risk_band} churn risk</h2>
                    <p>
                        <b>{prediction_output["prediction"]}</b><br>
                        {prediction_output["recommended_action"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            result_1, result_2, result_3 = st.columns(3)

            result_1.metric(
                "Churn probability",
                f"{churn_probability:.1%}",
            )

            result_2.metric(
                "Decision threshold",
                f"{float(prediction_output['decision_threshold']):.2f}",
            )

            result_3.metric(
                "Model",
                prediction_output["model_name"],
            )

            st.progress(churn_probability)

            with st.expander("Business impact details"):
                impact = prediction_output.get("business_impact", {})

                impact_1, impact_2, impact_3 = st.columns(3)

                impact_1.metric(
                    "Expected loss",
                    f"${impact.get('expected_loss_without_intervention', 0):,.2f}",
                )

                impact_2.metric(
                    "Expected net value",
                    f"${impact.get('expected_net_value', 0):,.2f}",
                )

                roi = impact.get("roi_percent")

                impact_3.metric(
                    "Campaign ROI",
                    f"{roi:.1f}%" if roi is not None else "N/A",
                )

                st.caption(f"Recommendation: {impact.get('recommendation', 'N/A')}")

        except requests.ConnectionError:
            st.error(f"Cannot reach the prediction API. Current API URL: `{API_URL}`.")

            st.info(
                "For local testing, start FastAPI in another terminal with:\n\n"
                "```powershell\n"
                '$env:PYTHONPATH = "$PWD\\customer_churn"\n'
                "uv run uvicorn customer_churn.api.main:app --reload\n"
                "```"
            )

        except requests.HTTPError as error:
            status_code = error.response.status_code

            st.error(f"The prediction API returned HTTP {status_code}.")

            try:
                st.json(error.response.json())
            except ValueError:
                st.code(error.response.text)

        except requests.RequestException as error:
            st.error(f"Prediction request failed: {error}")


# ============================================================
# PERFORMANCE TAB
# ============================================================

with performance_tab:
    st.subheader("Evaluation results")

    performance_1, performance_2, performance_3, performance_4 = st.columns(4)

    performance_1.metric(
        "Accuracy",
        f"{metrics['test_accuracy']:.2%}",
    )

    performance_2.metric(
        "Precision",
        f"{metrics['test_precision']:.3f}",
    )

    performance_3.metric(
        "Recall",
        f"{metrics['test_recall']:.3f}",
    )

    performance_4.metric(
        "Brier score",
        f"{metrics['mean_brier_score']:.4f}",
    )

    comparison = pd.DataFrame(metrics["cv_model_comparison"])

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    top_features = pd.DataFrame(metrics["top_features"])

    importance_chart = px.bar(
        top_features.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top permutation feature importance",
    )

    st.plotly_chart(
        importance_chart,
        use_container_width=True,
    )


# ============================================================
# MODEL INTELLIGENCE TAB
# ============================================================

with insights_tab:
    st.subheader("How ChurnGuard makes decisions")

    st.markdown(
        """
        **Pipeline:** validation → feature engineering → imputation and one-hot
        encoding → model selection → isotonic calibration → threshold tuning →
        retention recommendation.

        **Why not a fixed 0.50 threshold?** Missing a likely churner can be more
        expensive than contacting a customer who stays. The threshold is selected
        from out-of-fold probabilities to balance precision and recall.

        **Responsible use:** predictions are decision-support estimates, not
        guarantees. Review high-impact decisions and monitor model drift.
        """
    )

    st.json(
        {
            "selected_model": metrics["model_name"],
            "best_parameters": metrics["best_parameters"],
            "raw_features": metrics["raw_features"],
        }
    )


st.divider()

st.caption(
    "ChurnGuard · Calibrated customer churn prediction and retention decision support"
)
