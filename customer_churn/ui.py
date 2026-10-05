from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ChurnGuard AI | Telco Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR

DATA_PATH = PROJECT_DIR / "Telco_customer_churn.xlsx"
MODEL_PATH = PROJECT_DIR / "models" / "best_tuned_churn_model.pkl"

TUNED_METRICS_PATH = PROJECT_DIR / "reports" / "tuned_model_metrics.json"
MODEL_COMPARISON_PATH = PROJECT_DIR / "reports" / "hyperparameter_results.csv"
THRESHOLD_ANALYSIS_PATH = PROJECT_DIR / "reports" / "threshold_analysis.csv"
THRESHOLD_PATH = PROJECT_DIR / "reports" / "optimal_threshold.json"

TARGET_COLUMN = "Churn Label"

LEAKAGE_COLUMNS = {
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
}


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
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2.2rem;
            border-radius: 22px;
            background:
                radial-gradient(
                    circle at 88% 5%,
                    rgba(56, 189, 248, 0.28),
                    transparent 30%
                ),
                linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.18);
        }

        .eyebrow {
            color: #7dd3fc;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.1em;
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
            max-width: 780px;
            margin: 0.55rem 0 0;
            color: #dbeafe;
            font-size: 1.04rem;
            line-height: 1.6;
        }

        .section-title {
            margin: 1.1rem 0 0.75rem;
            color: #0f172a;
            font-size: 1.28rem;
            font-weight: 800;
        }

        .section-caption {
            margin-top: -0.4rem;
            margin-bottom: 0.95rem;
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

        .risk-high {
            border-left: 6px solid #dc2626;
        }

        .risk-medium {
            border-left: 6px solid #d97706;
        }

        .risk-low {
            border-left: 6px solid #16a34a;
        }

        .prediction-title {
            margin: 0;
            color: #0f172a;
            font-size: 1.48rem;
            font-weight: 800;
        }

        .prediction-description {
            margin: 0.55rem 0 0;
            color: #475569;
            font-size: 0.96rem;
            line-height: 1.55;
        }

        .insight-card {
            min-height: 120px;
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
            font-size: 1.15rem;
            font-weight: 800;
            overflow-wrap: anywhere;
        }

        .insight-detail {
            margin-top: 0.25rem;
            color: #64748b;
            font-size: 0.81rem;
            line-height: 1.4;
        }

        div[data-testid="stMetric"] {
            min-height: 110px;
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
# Cached loading
# ---------------------------------------------------------------------


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_json(path: str) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def display_name(column_name: str) -> str:
    return column_name.replace("_", " ").replace("-", " ").title()


def get_model_input_columns(model, dataset: pd.DataFrame) -> list[str]:
    """
    Uses the feature names stored in a fitted sklearn pipeline whenever
    available. Falls back to all non-leakage dataset columns.
    """
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is not None:
        return list(feature_names)

    return [column for column in dataset.columns if column not in LEAKAGE_COLUMNS]


def get_positive_class_index(model) -> int:
    classes = list(model.classes_)

    if 1 in classes:
        return classes.index(1)

    positive_labels = {
        "yes",
        "churn",
        "true",
        "1",
        "positive",
    }

    for index, label in enumerate(classes):
        if str(label).strip().lower() in positive_labels:
            return index

    return min(1, len(classes) - 1)


def extract_threshold(threshold_data: dict[str, Any]) -> float:
    """
    Supports common JSON structures produced by threshold optimization scripts.
    Defaults to 0.50 if no saved threshold is found.
    """
    possible_keys = [
        "best_f1_threshold",
        "optimal_threshold",
        "decision_threshold",
        "threshold",
    ]

    for key in possible_keys:
        value = threshold_data.get(key)

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, dict):
            nested_value = value.get("threshold")

            if isinstance(nested_value, (int, float)):
                return float(nested_value)

    return 0.50


def get_risk_details(
    probability: float,
    threshold: float,
) -> tuple[str, str, str, str]:
    high_risk_threshold = max(threshold, 0.65)
    medium_risk_threshold = max(threshold * 0.75, 0.35)

    if probability >= high_risk_threshold:
        return (
            "High churn risk",
            "risk-high",
            "⚠️",
            "The customer has a high estimated probability of churn. "
            "Consider proactive retention outreach or tailored offers.",
        )

    if probability >= medium_risk_threshold:
        return (
            "Moderate churn risk",
            "risk-medium",
            "⚠️",
            "The customer shows meaningful churn risk. Review account context "
            "and consider targeted retention actions.",
        )

    return (
        "Lower churn risk",
        "risk-low",
        "✅",
        "The customer currently has a lower estimated probability of churn.",
    )


def render_insight_card(
    label: str,
    value: str,
    detail: str,
) -> None:
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


def get_default_value(
    series: pd.Series,
    is_numeric: bool,
) -> Any:
    non_null = series.dropna()

    if non_null.empty:
        return 0.0 if is_numeric else "Unknown"

    if is_numeric:
        return float(pd.to_numeric(non_null, errors="coerce").median())

    mode = non_null.astype(str).mode()

    if not mode.empty:
        return str(mode.iloc[0])

    return str(non_null.iloc[0])


def create_input_widget(
    column: str,
    source_data: pd.DataFrame,
) -> Any:
    series = source_data[column]
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric:
        numeric_values = pd.to_numeric(series, errors="coerce").dropna()
        default = get_default_value(series, is_numeric=True)

        if numeric_values.empty:
            return st.number_input(
                display_name(column),
                value=0.0,
                step=1.0,
            )

        min_value = float(numeric_values.min())
        max_value = float(numeric_values.max())

        if min_value == max_value:
            max_value = min_value + 1.0

        step = 1.0

        if max_value - min_value < 10:
            step = 0.1

        return st.number_input(
            display_name(column),
            min_value=min_value,
            max_value=max_value,
            value=min(max(default, min_value), max_value),
            step=step,
        )

    values = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .index.tolist()
    )

    default = get_default_value(series, is_numeric=False)

    if default not in values:
        values.insert(0, default)

    if not values:
        values = ["Unknown"]

    if len(values) <= 30:
        return st.selectbox(
            display_name(column),
            options=values,
        )

    return st.text_input(
        display_name(column),
        value=default,
        help="Enter a value consistent with the training data.",
    )


# ---------------------------------------------------------------------
# Validate required artifacts
# ---------------------------------------------------------------------

missing_files = [path for path in [DATA_PATH, MODEL_PATH] if not path.exists()]

if missing_files:
    st.error(
        "Required project files are missing:\n\n"
        + "\n".join(f"- `{path}`" for path in missing_files)
        + "\n\nRun the pipeline first:\n\n"
        "`python -m customer_churn.run_all`"
    )
    st.stop()


# ---------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------

try:
    raw_data = load_dataset(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "Unable to load the dataset or trained model artifact. "
        "Confirm that the saved model was created from the current pipeline."
    )
    st.exception(exc)
    st.stop()

tuned_metrics = load_json(str(TUNED_METRICS_PATH))
threshold_data = load_json(str(THRESHOLD_PATH))
comparison_df = load_csv(str(MODEL_COMPARISON_PATH))
threshold_df = load_csv(str(THRESHOLD_ANALYSIS_PATH))

threshold = extract_threshold(threshold_data)
model_columns = get_model_input_columns(model, raw_data)

missing_model_columns = [
    column for column in model_columns if column not in raw_data.columns
]

if missing_model_columns:
    st.error(
        "The saved model expects features that are not available in the raw "
        "dataset used by this UI:\n\n"
        + ", ".join(f"`{column}`" for column in missing_model_columns)
        + "\n\nThis usually means feature engineering occurs outside the saved "
        "pipeline. Include your feature-engineering transformer inside the "
        "serialized sklearn pipeline, then retrain and save the model."
    )
    st.stop()

source_data = raw_data[model_columns].copy()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning Decision Support</div>
        <h1>📉 ChurnGuard AI</h1>
        <p>
            Estimate Telco customer churn risk using a tuned machine-learning
            pipeline. Review predicted probability, risk category, model
            performance, threshold behavior, and feature-level insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------

test_f1 = float(tuned_metrics.get("test_f1", 0))
test_recall = float(tuned_metrics.get("test_recall", 0))
test_roc_auc = float(tuned_metrics.get("test_roc_auc", 0))
best_cv_f1 = float(tuned_metrics.get("best_cv_f1", 0))
selected_model_name = (
    str(tuned_metrics.get("model", "Random Forest")).replace("_", " ").title()
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Selected Model",
        selected_model_name,
    )

with kpi2:
    st.metric(
        "Test F1 Score",
        f"{test_f1:.3f}" if tuned_metrics else "N/A",
    )

with kpi3:
    st.metric(
        "Test Recall",
        f"{test_recall:.1%}" if tuned_metrics else "N/A",
    )

with kpi4:
    st.metric(
        "Test ROC-AUC",
        f"{test_roc_auc:.3f}" if tuned_metrics else "N/A",
    )


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("ChurnGuard AI")

    st.caption("Retention-focused churn-risk intelligence for Telco customer profiles.")

    st.divider()

    st.markdown("#### Model status")
    st.success("Tuned model artifact loaded")

    st.caption(f"Decision threshold: {threshold:.2f}")

    if best_cv_f1:
        st.caption(f"Best CV F1 score: {best_cv_f1:.3f}")

    st.divider()

    st.markdown("#### Responsible use")

    st.caption(
        "Predictions are statistical estimates based on historical customer "
        "patterns. Use them to support retention outreach, not as the sole "
        "basis for automated customer decisions."
    )

    st.divider()

    st.caption("Telco Customer Churn · ML Decision Support")


# ---------------------------------------------------------------------
# Application tabs
# ---------------------------------------------------------------------

predict_tab, performance_tab, threshold_tab, model_tab = st.tabs(
    [
        "🔮 Predict Churn Risk",
        "📊 Model Performance",
        "🎯 Threshold Analysis",
        "🧠 Model Intelligence",
    ]
)


# ---------------------------------------------------------------------
# Prediction tab
# ---------------------------------------------------------------------

with predict_tab:
    st.markdown(
        '<div class="section-title">Customer churn-risk prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-caption">
            Enter a customer profile to estimate churn probability. Input fields
            are generated from the feature schema expected by the saved model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("churn_prediction_form", clear_on_submit=False):
        input_values: dict[str, Any] = {}

        columns_per_row = 3
        feature_groups = [
            model_columns[index : index + columns_per_row]
            for index in range(0, len(model_columns), columns_per_row)
        ]

        for feature_group in feature_groups:
            input_columns = st.columns(columns_per_row)

            for index, feature in enumerate(feature_group):
                with input_columns[index]:
                    input_values[feature] = create_input_widget(
                        column=feature,
                        source_data=source_data,
                    )

        submitted = st.form_submit_button(
            "🔮 Predict Churn Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame([input_values])[model_columns]

        try:
            probabilities = model.predict_proba(input_df)[0]
            positive_index = get_positive_class_index(model)
            churn_probability = float(probabilities[positive_index])
            predicted_churn = churn_probability >= threshold
        except Exception as exc:
            st.error(
                "Prediction failed. Confirm that the model artifact includes "
                "all required preprocessing and feature-engineering steps."
            )
            st.exception(exc)
            st.stop()

        risk_label, risk_class, icon, risk_message = get_risk_details(
            probability=churn_probability,
            threshold=threshold,
        )

        st.divider()

        result_col, probability_col = st.columns([1.15, 1])

        with result_col:
            st.markdown("### Prediction result")

            prediction_text = "Likely to churn" if predicted_churn else "Likely to stay"

            st.markdown(
                f"""
                <div class="result-card {risk_class}">
                    <p class="prediction-title">{icon} {risk_label}</p>
                    <p class="prediction-description">
                        <b>Prediction:</b> {prediction_text}
                    </p>
                    <p class="prediction-description">
                        {risk_message}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Churn probability",
                    f"{churn_probability:.1%}",
                )

            with metric2:
                st.metric(
                    "Decision threshold",
                    f"{threshold:.2f}",
                )

            with metric3:
                st.metric(
                    "Predicted outcome",
                    "Churn" if predicted_churn else "Stay",
                )

            st.progress(min(max(churn_probability, 0.0), 1.0))

            if churn_probability >= threshold:
                st.warning(
                    "Retention recommendation: prioritize this profile for "
                    "customer outreach, offer review, or service-support follow-up."
                )
            else:
                st.info(
                    "The model currently estimates churn probability below the "
                    "configured operating threshold."
                )

        with probability_col:
            st.markdown("### Probability distribution")

            probability_df = pd.DataFrame(
                {
                    "Outcome": ["Stay", "Churn"],
                    "Probability": [
                        1 - churn_probability,
                        churn_probability,
                    ],
                }
            )

            st.dataframe(
                probability_df.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )

            probability_chart = px.bar(
                probability_df,
                x="Probability",
                y="Outcome",
                orientation="h",
                text="Probability",
                color="Outcome",
                color_discrete_map={
                    "Stay": "#16a34a",
                    "Churn": "#dc2626",
                },
            )

            probability_chart.update_traces(
                texttemplate="%{text:.1%}",
                textposition="outside",
            )

            probability_chart.update_layout(
                height=300,
                margin=dict(l=10, r=35, t=10, b=10),
                showlegend=False,
                xaxis=dict(tickformat=".0%", range=[0, 1]),
            )

            st.plotly_chart(
                probability_chart,
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-title">Customer profile used for prediction</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )

        prediction_history_item = {
            "Prediction": "Churn" if predicted_churn else "Stay",
            "Churn Probability": churn_probability,
            "Threshold": threshold,
        }

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        st.session_state.prediction_history.insert(
            0,
            prediction_history_item,
        )

    else:
        st.info(
            "Enter customer information and click **Predict Churn Risk** to "
            "generate a churn probability and retention recommendation."
        )

    if st.session_state.get("prediction_history"):
        st.markdown(
            '<div class="section-title">Session prediction history</div>',
            unsafe_allow_html=True,
        )

        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df["Churn Probability"] = history_df["Churn Probability"].map(
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
        '<div class="section-title">Model performance comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-caption">
            Results are loaded from the latest successful hyperparameter-tuning
            run. The selected model prioritizes cross-validated F1 performance
            for retention-oriented churn detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if comparison_df.empty:
        st.warning(
            "No model comparison report was found. Run:\n\n"
            "`python -m customer_churn.tune`"
        )
    else:
        metric_columns = [
            "test_f1",
            "test_recall",
            "test_roc_auc",
            "test_average_precision",
        ]

        available_metrics = [
            column for column in metric_columns if column in comparison_df.columns
        ]

        if available_metrics:
            performance_display = comparison_df.copy()

            for column in available_metrics:
                performance_display[column] = performance_display[column].map(
                    lambda value: f"{float(value):.3f}"
                )

            rename_map = {
                "model": "Model",
                "best_cv_f1": "CV F1",
                "test_accuracy": "Test Accuracy",
                "test_balanced_accuracy": "Balanced Accuracy",
                "test_precision": "Precision",
                "test_recall": "Recall",
                "test_f1": "Test F1",
                "test_roc_auc": "ROC-AUC",
                "test_average_precision": "Average Precision",
                "fit_seconds": "Fit Time (s)",
            }

            display_columns = [
                column for column in rename_map if column in performance_display.columns
            ]

            st.dataframe(
                performance_display[display_columns].rename(columns=rename_map),
                use_container_width=True,
                hide_index=True,
            )

        left_col, right_col = st.columns(2)

        with left_col:
            if "test_f1" in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="model",
                    y="test_f1",
                    text="test_f1",
                    color="test_f1",
                    color_continuous_scale="Blues",
                    labels={
                        "model": "Model",
                        "test_f1": "Test F1 Score",
                    },
                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )

                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=20, b=10),
                    coloraxis_showscale=False,
                    yaxis=dict(range=[0, 1]),
                )

                st.plotly_chart(fig, use_container_width=True)

        with right_col:
            if "test_roc_auc" in comparison_df.columns:
                fig = px.bar(
                    comparison_df,
                    x="model",
                    y="test_roc_auc",
                    text="test_roc_auc",
                    color="test_roc_auc",
                    color_continuous_scale="Greens",
                    labels={
                        "model": "Model",
                        "test_roc_auc": "Test ROC-AUC",
                    },
                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )

                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=20, b=10),
                    coloraxis_showscale=False,
                    yaxis=dict(range=[0, 1]),
                )

                st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Evaluation interpretation")

    insight1, insight2, insight3 = st.columns(3)

    with insight1:
        render_insight_card(
            "Primary selection metric",
            f"{best_cv_f1:.3f}" if best_cv_f1 else "F1-score",
            "F1 balances churn precision and recall for an imbalanced target.",
        )

    with insight2:
        render_insight_card(
            "Retention priority",
            f"{test_recall:.1%}" if test_recall else "Recall",
            "Recall measures how many true churners the model identifies.",
        )

    with insight3:
        render_insight_card(
            "Ranking quality",
            f"{test_roc_auc:.3f}" if test_roc_auc else "ROC-AUC",
            "ROC-AUC measures the model's ability to rank churn risk.",
        )


# ---------------------------------------------------------------------
# Threshold analysis tab
# ---------------------------------------------------------------------

with threshold_tab:
    st.markdown(
        '<div class="section-title">Threshold optimization</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-caption">
            The operating threshold determines when predicted churn probability
            becomes a churn-risk classification. Lower thresholds capture more
            churners but can increase false-positive retention outreach.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if threshold_df.empty:
        st.warning(
            "No threshold analysis report was found. Run:\n\n"
            "`python -m customer_churn.threshold`"
        )
    else:
        required_columns = {
            "threshold",
            "precision",
            "recall",
            "f1",
        }

        if required_columns.issubset(threshold_df.columns):
            threshold_long = threshold_df.melt(
                id_vars=["threshold"],
                value_vars=["precision", "recall", "f1"],
                var_name="Metric",
                value_name="Score",
            )

            figure = px.line(
                threshold_long,
                x="threshold",
                y="Score",
                color="Metric",
                markers=True,
                color_discrete_map={
                    "precision": "#2563eb",
                    "recall": "#16a34a",
                    "f1": "#d97706",
                },
                labels={
                    "threshold": "Classification Threshold",
                    "Score": "Metric Score",
                },
            )

            figure.add_vline(
                x=threshold,
                line_width=2,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text=f"Configured threshold: {threshold:.2f}",
                annotation_position="top right",
            )

            figure.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(range=[0, 1]),
            )

            st.plotly_chart(figure, use_container_width=True)

        st.markdown("### Threshold results")

        threshold_display = threshold_df.copy()

        metric_columns = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]

        for column in metric_columns:
            if column in threshold_display.columns:
                threshold_display[column] = threshold_display[column].map(
                    lambda value: f"{float(value):.3f}"
                )

        st.dataframe(
            threshold_display,
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "For strict final-model evaluation, select hyperparameters and the "
        "classification threshold using cross-validation or a validation set "
        "within the training partition, then evaluate once on an untouched "
        "held-out test set."
    )


# ---------------------------------------------------------------------
# Model intelligence tab
# ---------------------------------------------------------------------

with model_tab:
    st.markdown(
        '<div class="section-title">Model intelligence</div>',
        unsafe_allow_html=True,
    )

    architecture_col, feature_col = st.columns(2)

    with architecture_col:
        st.markdown("### Pipeline architecture")

        st.markdown(
            """
            - **Data preparation:** Cleans Telco customer data and removes
              leakage-prone fields such as churn scores, churn reasons, CLTV,
              and churn-value columns.
            - **Preprocessing:** Handles numeric and categorical variables
              through a reusable scikit-learn pipeline.
            - **Model benchmarking:** Compares Logistic Regression, Random
              Forest, and Histogram Gradient Boosting.
            - **Tuning:** Uses Windows-safe `GridSearchCV` with `n_jobs=1`.
            - **Selection:** Prioritizes cross-validated F1-score for
              retention-oriented churn detection.
            - **Decision policy:** Applies a saved threshold associated with
              the final tuned model artifact.
            """
        )

    with feature_col:
        st.markdown("### Business interpretation")

        st.markdown(
            """
            - **Recall:** Higher recall identifies more customers who are
              actually likely to churn.
            - **Precision:** Higher precision reduces unnecessary retention
              outreach to customers unlikely to churn.
            - **F1-score:** Balances the trade-off between precision and recall.
            - **ROC-AUC:** Measures churn-risk ranking quality independently of
              a specific classification threshold.
            - **Threshold tuning:** Aligns the model's operating point with
              business priorities, such as retention coverage versus campaign cost.
            """
        )

    st.markdown("### Feature schema")

    numeric_features = [
        column
        for column in model_columns
        if pd.api.types.is_numeric_dtype(source_data[column])
    ]

    categorical_features = [
        column for column in model_columns if column not in numeric_features
    ]

    schema_col1, schema_col2 = st.columns(2)

    with schema_col1:
        st.markdown("#### Numeric features")

        st.dataframe(
            pd.DataFrame({"Feature": numeric_features}),
            use_container_width=True,
            hide_index=True,
        )

    with schema_col2:
        st.markdown("#### Categorical features")

        st.dataframe(
            pd.DataFrame({"Feature": categorical_features}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Responsible-use note")

    st.warning(
        "This application provides churn-risk estimates based on historical "
        "customer patterns. It should support human retention decisions, not "
        "replace customer-service judgment or be used as the sole basis for "
        "high-impact automated decisions."
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "ChurnGuard AI · Telco Customer Churn Prediction · "
    "Leakage-safe preprocessing, model tuning, threshold analysis, "
    "and interactive decision support"
)

