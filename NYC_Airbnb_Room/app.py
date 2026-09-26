"""
NYC Airbnb Room Intelligence

Professional Streamlit UI for the trained NYC Airbnb room-type classifier.

The saved pipeline includes:
- Feature engineering
- Numeric/categorical imputation
- One-hot encoding
- Optuna-selected Random Forest or HistGradientBoosting
- Isotonic calibration
- Tuned class decision weights for Shared room
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="StayType AI | NYC Airbnb Room Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Best_Model_Pipeline.pkl"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

# Required for joblib to find custom classes such as FeatureEngineer
# and WeightedDecisionClassifier while loading the saved pipeline.
sys.path.insert(0, str(BASE_DIR))


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --navy: #0f172a;
            --navy-soft: #172033;
            --blue: #2563eb;
            --sky: #38bdf8;
            --green: #16a34a;
            --amber: #d97706;
            --red: #dc2626;
            --slate: #64748b;
            --border: #e2e8f0;
            --background: #f8fafc;
        }

        .main {
            background: var(--background);
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.9rem;
            padding-bottom: 2.4rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.2rem;
            border-radius: 22px;
            background:
                radial-gradient(
                    circle at 88% 5%,
                    rgba(56, 189, 248, 0.30),
                    transparent 30%
                ),
                linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
        }

        .eyebrow {
            color: #7dd3fc;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.45rem;
            font-weight: 850;
            letter-spacing: -0.04em;
        }

        .hero p {
            max-width: 760px;
            margin: 0.55rem 0 0;
            color: #dbeafe;
            font-size: 1.04rem;
            line-height: 1.6;
        }

        .section-title {
            margin: 1.1rem 0 0.8rem;
            color: #0f172a;
            font-size: 1.28rem;
            font-weight: 800;
        }

        .section-caption {
            margin-top: -0.45rem;
            margin-bottom: 0.9rem;
            color: #64748b;
            font-size: 0.92rem;
        }

        .result-card {
            padding: 1.5rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .prediction-high {
            border-left: 6px solid #16a34a;
        }

        .prediction-medium {
            border-left: 6px solid #d97706;
        }

        .prediction-low {
            border-left: 6px solid #dc2626;
        }

        .prediction-title {
            margin: 0;
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 800;
        }

        .prediction-description {
            margin: 0.55rem 0 0;
            color: #475569;
            font-size: 0.96rem;
            line-height: 1.55;
        }

        .insight-card {
            min-height: 125px;
            padding: 1rem 1.05rem;
            border: 1px solid #e2e8f0;
            border-radius: 15px;
            background: white;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
        }

        .insight-label {
            color: #64748b;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .insight-value {
            margin-top: 0.38rem;
            color: #0f172a;
            font-size: 1.18rem;
            font-weight: 800;
            overflow-wrap: anywhere;
        }

        .insight-detail {
            margin-top: 0.25rem;
            color: #64748b;
            font-size: 0.81rem;
            line-height: 1.4;
        }

        .pipeline-row {
            display: flex;
            flex-wrap: wrap;
            align-items: stretch;
            gap: 0.45rem;
            margin: 0.55rem 0 1.25rem;
        }

        .pipeline-stage {
            min-width: 154px;
            flex: 1 1 154px;
            padding: 0.76rem 0.88rem;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            background: #f8fafc;
            color: #334155;
            font-size: 0.82rem;
            line-height: 1.4;
        }

        .pipeline-stage b {
            display: block;
            margin-bottom: 0.2rem;
            color: #0f172a;
            font-size: 0.9rem;
        }

        .pipeline-stage.active {
            background: #eff6ff;
            border-color: #60a5fa;
        }

        .pipeline-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
            font-size: 1.2rem;
        }

        div[data-testid="stMetric"] {
            min-height: 112px;
            padding: 0.95rem 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: white;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 850 !important;
        }

        button[data-baseweb="tab"] {
            font-weight: 700;
            color: #475569;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0f172a;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------

PIPELINE_STAGES = [
    ("1. Raw data", "Load nyc.csv, inspect missing data, generate EDA"),
    ("2. Clean & split", "Remove ID/text columns and use stratified splitting"),
    ("3. Engineer", "Leakage-safe ratios, logs, capping, and binary flags"),
    ("4. Encode", "Imputation plus one-hot encoding within each fold"),
    ("5. Optuna", "Tune Random Forest versus HistGradientBoosting"),
    ("6. Calibrate", "Use isotonic calibration on out-of-fold probabilities"),
    ("7. Weight decisions", "Improve Shared-room recall using tuned class weights"),
    ("8. Evaluate", "Refit final model and evaluate on the held-out test set"),
]


# ---------------------------------------------------------------------
# Cached loading
# ---------------------------------------------------------------------


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def model_display_name(model_type: str | None) -> str:
    names = {
        "random_forest": "Random Forest",
        "hist_gradient_boosting": "Histogram Gradient Boosting",
    }
    return names.get(model_type or "", model_type or "Unknown")


def get_prediction_input(
    neighbourhood_group: str,
    neighbourhood: str,
    latitude: float,
    longitude: float,
    price: float,
    minimum_nights: int,
    number_of_reviews: int,
    reviews_per_month: float,
    calculated_host_listings_count: int,
    availability_365: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "neighbourhood_group": neighbourhood_group,
                "neighbourhood": neighbourhood.strip() or "Unknown",
                "latitude": latitude,
                "longitude": longitude,
                "price": price,
                "minimum_nights": minimum_nights,
                "number_of_reviews": number_of_reviews,
                "reviews_per_month": reviews_per_month,
                "calculated_host_listings_count": calculated_host_listings_count,
                "availability_365": availability_365,
            }
        ]
    )


def render_pipeline_flow() -> None:
    boxes = []

    for index, (title, description) in enumerate(PIPELINE_STAGES):
        active_class = " active" if index in {4, 5, 6, 7} else ""

        boxes.append(
            f"""
            <div class="pipeline-stage{active_class}">
                <b>{title}</b>
                {description}
            </div>
            """
        )

        if index < len(PIPELINE_STAGES) - 1:
            boxes.append('<div class="pipeline-arrow">→</div>')

    st.markdown(
        f'<div class="pipeline-row">{"".join(boxes)}</div>',
        unsafe_allow_html=True,
    )


def render_insight_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_label(probability: float) -> tuple[str, str, str]:
    if probability >= 0.80:
        return (
            "High confidence",
            "prediction-high",
            "The listing profile strongly matches the predicted room type.",
        )

    if probability >= 0.55:
        return (
            "Moderate confidence",
            "prediction-medium",
            "The prediction is plausible, but nearby room-type alternatives remain possible.",
        )

    return (
        "Low confidence",
        "prediction-low",
        "The model sees a close decision boundary. Manual review is recommended.",
    )


# ---------------------------------------------------------------------
# Validate files
# ---------------------------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found.\n\n"
        "Run this command first:\n\n"
        "`uv run python .\\NYC_Airbnb_Room\\train_model.py`"
    )
    st.stop()


try:
    model = load_model()
except Exception as exc:
    st.error(
        "The trained model could not be loaded. Ensure `nyc_features.py` is "
        "present beside this app and the artifact was created by the current code."
    )
    st.exception(exc)
    st.stop()

metrics = load_metrics()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning Decision Support</div>
        <h1>🏠 StayType AI</h1>
        <p>
            Predict whether an NYC Airbnb listing is most likely an Entire home/apt,
            Private room, or Shared room using an Optuna-tuned and calibrated
            multiclass classification pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------

best_parameters = metrics.get("best_parameters", {}) if metrics else {}
best_model_name = model_display_name(best_parameters.get("model_type"))

test_accuracy = float(metrics.get("test_accuracy", 0)) if metrics else 0.0
test_macro_f1 = float(metrics.get("test_macro_f1", 0)) if metrics else 0.0
best_cv_macro_f1 = float(metrics.get("best_cv_macro_f1", 0)) if metrics else 0.0
mean_brier_score = float(metrics.get("mean_brier_score", 0)) if metrics else 0.0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Test Accuracy",
        f"{test_accuracy:.2%}" if metrics else "N/A",
    )

with kpi2:
    st.metric(
        "Test Macro-F1",
        f"{test_macro_f1:.4f}" if metrics else "N/A",
    )

with kpi3:
    st.metric(
        "Selected Model",
        best_model_name,
    )

with kpi4:
    st.metric(
        "Calibration Quality",
        f"Brier {mean_brier_score:.4f}" if metrics else "N/A",
    )


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("StayType AI")

    st.caption("Room-type intelligence for listing-level classification.")

    st.divider()

    st.markdown("#### Model status")

    if metrics:
        st.success("Model artifact loaded")
        st.caption(f"Best CV Macro-F1: {best_cv_macro_f1:.4f}")
        st.caption(f"Test Macro-F1: {test_macro_f1:.4f}")
    else:
        st.warning("Metrics report unavailable")

    st.divider()

    st.markdown("#### Responsible use")

    st.caption(
        "Predictions are machine-learning estimates based on historic listing "
        "patterns. They are not verified Airbnb listing attributes."
    )

    st.divider()

    st.caption("NYC Airbnb Room Intelligence")


# ---------------------------------------------------------------------
# Application tabs
# ---------------------------------------------------------------------

predict_tab, performance_tab, pipeline_tab = st.tabs(
    [
        "🔮 Predict Room Type",
        "📊 Model Performance",
        "🧠 Pipeline Intelligence",
    ]
)


# ---------------------------------------------------------------------
# Prediction tab
# ---------------------------------------------------------------------

with predict_tab:
    st.markdown(
        '<div class="section-title">Listing classification</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-caption">
            Enter listing details to estimate the most likely Airbnb room category.
            The form runs only when you click the prediction button.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("listing_prediction_form", clear_on_submit=False):
        st.markdown("#### Location and pricing")

        location_col, coordinates_col, pricing_col = st.columns(3)

        with location_col:
            neighbourhood_group = st.selectbox(
                "Neighbourhood group",
                [
                    "Manhattan",
                    "Brooklyn",
                    "Queens",
                    "Bronx",
                    "Staten Island",
                ],
            )

            neighbourhood = st.text_input(
                "Neighbourhood",
                value="Harlem",
                help="Examples: Harlem, Williamsburg, Astoria, Chelsea.",
            )

        with coordinates_col:
            latitude = st.number_input(
                "Latitude",
                min_value=40.0,
                max_value=41.0,
                value=40.8116,
                step=0.0001,
                format="%.4f",
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-74.5,
                max_value=-73.0,
                value=-73.9465,
                step=0.0001,
                format="%.4f",
            )

        with pricing_col:
            price = st.number_input(
                "Price per night ($)",
                min_value=0.0,
                value=150.0,
                step=5.0,
            )

            minimum_nights = st.number_input(
                "Minimum nights",
                min_value=1,
                value=2,
                step=1,
            )

        st.markdown("#### Listing activity and host profile")

        activity_col, host_col, availability_col = st.columns(3)

        with activity_col:
            number_of_reviews = st.number_input(
                "Number of reviews",
                min_value=0,
                value=10,
                step=1,
            )

            reviews_per_month = st.number_input(
                "Reviews per month",
                min_value=0.0,
                value=1.0,
                step=0.1,
            )

        with host_col:
            calculated_host_listings_count = st.number_input(
                "Host listing count",
                min_value=1,
                value=1,
                step=1,
            )

        with availability_col:
            availability_365 = st.slider(
                "Availability over next 365 days",
                min_value=0,
                max_value=365,
                value=180,
            )

        submitted = st.form_submit_button(
            "🔮 Predict Room Type",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = get_prediction_input(
            neighbourhood_group=neighbourhood_group,
            neighbourhood=neighbourhood,
            latitude=latitude,
            longitude=longitude,
            price=price,
            minimum_nights=minimum_nights,
            number_of_reviews=number_of_reviews,
            reviews_per_month=reviews_per_month,
            calculated_host_listings_count=calculated_host_listings_count,
            availability_365=availability_365,
        )

        with st.spinner("Analyzing listing features and generating a prediction..."):
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

        probability_map = {
            str(room_type): float(probability)
            for room_type, probability in zip(classes, probabilities)
        }

        predicted_probability = probability_map[str(prediction)]
        confidence, confidence_class, confidence_message = confidence_label(
            predicted_probability
        )

        probability_df = (
            pd.DataFrame(
                {
                    "Room Type": list(probability_map.keys()),
                    "Probability": list(probability_map.values()),
                }
            )
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )

        prediction_descriptions = {
            "Entire home/apt": (
                "The listing profile most closely matches an entire property "
                "or apartment rental."
            ),
            "Private room": (
                "The listing profile most closely matches a private-room rental."
            ),
            "Shared room": (
                "The listing profile most closely matches a shared-room rental."
            ),
        }

        st.divider()

        result_col, probability_col = st.columns([1.15, 1])

        with result_col:
            st.markdown("### Prediction result")

            st.markdown(
                f"""
                <div class="result-card {confidence_class}">
                    <p class="prediction-title">🏠 {prediction}</p>
                    <p class="prediction-description">
                        {
                    prediction_descriptions.get(
                        str(prediction), "The model selected this room category."
                    )
                }
                    </p>
                    <p class="prediction-description">
                        <b>{confidence}:</b> {confidence_message}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Prediction confidence",
                    f"{predicted_probability:.2%}",
                )

            with metric2:
                st.metric(
                    "Predicted room type",
                    str(prediction),
                )

            with metric3:
                st.metric(
                    "Price per night",
                    f"${price:,.0f}",
                )

            st.progress(min(max(predicted_probability, 0.0), 1.0))

            if predicted_probability < 0.55:
                st.warning(
                    "Low-confidence result: this listing may resemble multiple "
                    "room categories. Review the listing manually before using "
                    "the prediction in a high-impact workflow."
                )
            else:
                st.info(
                    "This is a machine-learning estimate based on historical "
                    "listing patterns, not verified Airbnb listing data."
                )

        with probability_col:
            st.markdown("### Class probabilities")

            st.dataframe(
                probability_df.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )

            probability_chart = px.bar(
                probability_df,
                x="Probability",
                y="Room Type",
                orientation="h",
                text="Probability",
                color="Probability",
                color_continuous_scale="Blues",
                range_color=[0, 1],
            )

            probability_chart.update_traces(
                texttemplate="%{text:.1%}",
                textposition="outside",
            )

            probability_chart.update_layout(
                height=300,
                margin=dict(l=10, r=35, t=10, b=10),
                coloraxis_showscale=False,
                xaxis=dict(tickformat=".0%", range=[0, 1]),
            )

            st.plotly_chart(
                probability_chart,
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-title">Listing profile used for prediction</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )

        history_item = {
            "Timestamp (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Prediction": str(prediction),
            "Confidence": predicted_probability,
            "Neighbourhood Group": neighbourhood_group,
            "Neighbourhood": neighbourhood,
            "Price": price,
        }

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        st.session_state.prediction_history.insert(0, history_item)

    else:
        st.info(
            "Enter a listing profile and click **Predict Room Type** to see "
            "the predicted class, confidence, and probability distribution."
        )

    if st.session_state.get("prediction_history"):
        st.markdown(
            '<div class="section-title">Session prediction history</div>',
            unsafe_allow_html=True,
        )

        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df["Confidence"] = history_df["Confidence"].map(
            lambda value: f"{value:.2%}"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------
# Model performance tab
# ---------------------------------------------------------------------

with performance_tab:
    st.markdown(
        '<div class="section-title">Model performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-caption">
            Evaluation metrics are loaded dynamically from
            <code>reports/metrics.json</code>, so the dashboard reflects the
            most recent successful training run.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics is None:
        st.warning(
            "No metrics report was found. Run `train_model.py` to generate "
            "`reports/metrics.json`."
        )
    else:
        perf1, perf2, perf3, perf4 = st.columns(4)

        with perf1:
            st.metric("Test Accuracy", f"{test_accuracy:.2%}")

        with perf2:
            st.metric("Test Macro-F1", f"{test_macro_f1:.4f}")

        with perf3:
            st.metric("Best CV Macro-F1", f"{best_cv_macro_f1:.4f}")

        with perf4:
            st.metric("Mean Brier Score", f"{mean_brier_score:.4f}")

        per_class_f1 = metrics.get("per_class_f1", {})

        left, right = st.columns([1.1, 1])

        with left:
            st.markdown("### Per-class F1 score")

            if per_class_f1:
                f1_df = pd.DataFrame(
                    {
                        "Room Type": list(per_class_f1.keys()),
                        "F1 Score": list(per_class_f1.values()),
                    }
                )

                fig = px.bar(
                    f1_df,
                    x="Room Type",
                    y="F1 Score",
                    text="F1 Score",
                    color="F1 Score",
                    color_continuous_scale="Blues",
                    range_color=[0, 1],
                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )

                fig.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                    yaxis=dict(range=[0, 1]),
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Per-class F1 metrics are unavailable.")

        with right:
            st.markdown("### Evaluation interpretation")

            shared_room_f1 = float(per_class_f1.get("Shared room", 0))

            render_insight_card(
                "Overall strength",
                f"{test_macro_f1:.4f}",
                "Macro-F1 weights every room class equally, including rare Shared rooms.",
            )

            render_insight_card(
                "Minority-class challenge",
                f"{shared_room_f1:.4f}",
                "Shared room is the rarest class and remains the primary improvement opportunity.",
            )

        st.markdown("### Calibration quality")

        brier_scores = metrics.get("brier_scores_by_class", {})

        if brier_scores:
            brier_df = pd.DataFrame(
                {
                    "Room Type": list(brier_scores.keys()),
                    "Brier Score": list(brier_scores.values()),
                }
            )

            calibration_col, table_col = st.columns([1.1, 1])

            with calibration_col:
                fig = px.bar(
                    brier_df,
                    x="Room Type",
                    y="Brier Score",
                    text="Brier Score",
                    color="Brier Score",
                    color_continuous_scale="RdYlGn_r",
                )

                fig.update_traces(
                    texttemplate="%{text:.4f}",
                    textposition="outside",
                )

                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                )

                st.plotly_chart(fig, use_container_width=True)

            with table_col:
                st.markdown(
                    """
                    **What is a Brier score?**

                    It measures probability calibration. Lower values indicate
                    that predicted probabilities better match observed outcomes.

                    This matters because the interface shows model confidence,
                    not only the final predicted class.
                    """
                )

                st.dataframe(
                    brier_df.style.format({"Brier Score": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Decision policy")

        decision_weights = metrics.get("decision_weights", {})

        weight_col, params_col = st.columns([1, 1.2])

        with weight_col:
            if decision_weights:
                weights_df = pd.DataFrame(
                    {
                        "Room Type": list(decision_weights.keys()),
                        "Decision Weight": list(decision_weights.values()),
                    }
                )

                st.dataframe(
                    weights_df.style.format({"Decision Weight": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Decision weights affect the final predicted class. They are "
                    "tuned to improve rare Shared-room recall without changing "
                    "the underlying calibrated probability estimates."
                )

        with params_col:
            st.markdown("#### Best model configuration")

            if best_parameters:
                st.json(best_parameters)
            else:
                st.info("Best hyperparameters are unavailable.")


# ---------------------------------------------------------------------
# Pipeline intelligence tab
# ---------------------------------------------------------------------

with pipeline_tab:
    st.markdown(
        '<div class="section-title">Training pipeline intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-caption">
            The final artifact is not a raw classifier. It is a reproducible
            end-to-end pipeline that performs feature engineering,
            preprocessing, calibration, and cost-sensitive decision adjustment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_pipeline_flow()

    architecture_col, feature_col = st.columns(2)

    with architecture_col:
        st.markdown("### Pipeline design")

        st.markdown(
            """
            - **Model selection:** Optuna compares Random Forest and
              Histogram Gradient Boosting across pruned 3-fold trials.
            - **Preprocessing:** Numeric columns use median imputation;
              categorical columns use mode imputation plus one-hot encoding.
            - **No scaling:** Tree models do not require feature scaling or
              Yeo-Johnson transformations.
            - **Leakage protection:** Feature engineering and preprocessing
              are fitted only on training folds during cross-validation.
            - **Final artifact:** The selected pipeline is refit on the full
              training partition before evaluation on the untouched test set.
            """
        )

    with feature_col:
        st.markdown("### Decision intelligence")

        st.markdown(
            """
            - **Feature engineering:** The pipeline can use price transforms,
              neighbourhood-relative price signals, occupancy proxies, review
              indicators, and host/listing characteristics.
            - **Calibration:** Isotonic calibration makes confidence scores
              more meaningful than raw uncalibrated model scores.
            - **Rare-class support:** Shared rooms are underrepresented, so
              decision weights are tuned using out-of-fold predictions.
            - **Responsible output:** The app displays class probabilities and
              warns users when the top predicted probability is low.
            """
        )

    st.markdown("### Input features")

    feature_table = pd.DataFrame(
        {
            "Feature": [
                "neighbourhood_group",
                "neighbourhood",
                "latitude",
                "longitude",
                "price",
                "minimum_nights",
                "number_of_reviews",
                "reviews_per_month",
                "calculated_host_listings_count",
                "availability_365",
            ],
            "Purpose": [
                "Broad NYC borough context",
                "Local neighbourhood context",
                "Location coordinate",
                "Location coordinate",
                "Nightly listing price",
                "Minimum stay requirement",
                "Historical review volume",
                "Review activity rate",
                "Host portfolio size",
                "Future listing availability",
            ],
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Responsible-use note")

    st.warning(
        "This project predicts room-type patterns from historical listing "
        "attributes. It should not be used as a substitute for verified listing "
        "data, regulatory decisions, housing decisions, or high-impact automated "
        "classification without human review."
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "StayType AI · NYC Airbnb Room Type Classification · "
    "Optuna-tuned, calibrated multiclass machine-learning pipeline"
)
