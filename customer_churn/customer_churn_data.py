import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="ChurnGuard | Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Professional styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #111827 0%, #1e3a5f 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
    }

    .hero p {
        margin: 0.55rem 0 0 0;
        color: #dbeafe;
        font-size: 1.05rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #111827;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .result-card h2 {
        color: #111827;
        margin-top: 0;
    }

    .result-card p {
        color: #374151;
        margin-bottom: 0;
    }

    .risk-high {
        border-left: 6px solid #dc2626;
    }

    .risk-low {
        border-left: 6px solid #16a34a;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.88rem;
    }

    /* ---- KPI / metric cards ---- */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem 1.1rem;
        border-radius: 14px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #16a34a !important;
    }

    /* ---- Tabs ---- */
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #374151;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #111827;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #f3f4f6;
    }

    section[data-testid="stSidebar"] .stButton button {
        border-radius: 12px;
        font-weight: 700;
    }

    /* ---- Dataframes ---- */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "customer_churn_data.csv"

FEATURES = ["Age", "Gender", "Tenure", "MonthlyCharges"]
TARGET = "Churn"


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv("customer_churn/customer_churn_data.csv")

    if "InternetService" in df.columns:
        df["InternetService"] = df["InternetService"].fillna("")

    return df


# -------------------------------------------------------------------
# Model training
# -------------------------------------------------------------------


@st.cache_resource
def train_models(df: pd.DataFrame):

    # ---------------------------------------------------------------
    # Select required columns
    # ---------------------------------------------------------------

    data = df[FEATURES + [TARGET]].copy()

    # ---------------------------------------------------------------
    # Encode Gender
    # Male   = 0
    # Female = 1
    # ---------------------------------------------------------------

    data["Gender"] = (
        data["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "male": 0,
                "female": 1,
            }
        )
    )

    # ---------------------------------------------------------------
    # Encode Churn
    # No  = 0
    # Yes = 1
    # ---------------------------------------------------------------

    data["Churn"] = (
        data["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "no": 0,
                "yes": 1,
            }
        )
    )

    # ---------------------------------------------------------------
    # Convert numerical columns safely
    # ---------------------------------------------------------------

    for column in ["Age", "Tenure", "MonthlyCharges"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ---------------------------------------------------------------
    # Remove invalid rows
    # ---------------------------------------------------------------

    data = data.dropna()

    # ---------------------------------------------------------------
    # Validate dataset
    # ---------------------------------------------------------------

    if data.empty:
        raise ValueError("No valid rows remain after data cleaning.")

    if data[TARGET].nunique() < 2:
        raise ValueError("The Churn column must contain both Yes and No classes.")

    # ---------------------------------------------------------------
    # Features and target
    # ---------------------------------------------------------------

    X = data[FEATURES]
    y = data[TARGET]

    # ---------------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------------------
    # Model configurations
    #
    # IMPORTANT:
    # n_jobs=1 is intentional.
    #
    # Your previous configuration used:
    #
    # GridSearchCV n_jobs=-1
    # +
    # RandomForest n_jobs=-1
    #
    # This created nested parallel processing and exhausted
    # Windows virtual memory.
    # ---------------------------------------------------------------

    model_configs = {
        # -----------------------------------------------------------
        # Logistic Regression
        # -----------------------------------------------------------
        "Logistic Regression": (
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(max_iter=2000),
                    ),
                ]
            ),
            {
                "model__C": [
                    0.1,
                    1,
                    10,
                ],
                "model__solver": [
                    "liblinear",
                    "lbfgs",
                ],
            },
        ),
        # -----------------------------------------------------------
        # KNN
        # -----------------------------------------------------------
        "KNN": (
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        KNeighborsClassifier(),
                    ),
                ]
            ),
            {
                "model__n_neighbors": [
                    3,
                    5,
                    7,
                    9,
                ],
                "model__weights": [
                    "uniform",
                    "distance",
                ],
            },
        ),
        # -----------------------------------------------------------
        # SVM
        # -----------------------------------------------------------
        "SVM": (
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        SVC(
                            probability=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "model__kernel": [
                    "rbf",
                    "linear",
                ],
                "model__C": [
                    0.1,
                    1,
                    10,
                ],
                "model__gamma": [
                    "scale",
                    "auto",
                ],
            },
        ),
        # -----------------------------------------------------------
        # Random Forest
        #
        # Reduced parameter grid to lower memory usage.
        # -----------------------------------------------------------
        "Random Forest": (
            RandomForestClassifier(
                random_state=42,
                n_jobs=1,
            ),
            {
                "n_estimators": [
                    100,
                    200,
                ],
                "max_depth": [
                    None,
                    10,
                ],
                "max_features": [
                    "sqrt",
                ],
            },
        ),
    }

    # ---------------------------------------------------------------
    # Train models
    # ---------------------------------------------------------------

    trained_models = {}
    rows = []

    for name, (estimator, params) in model_configs.items():
        # -----------------------------------------------------------
        # GridSearchCV
        #
        # cv=3 instead of cv=5:
        # reduces total training operations.
        #
        # n_jobs=1:
        # prevents Joblib multiprocessing.
        # -----------------------------------------------------------

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            cv=3,
            scoring="roc_auc",
            n_jobs=1,
        )

        grid.fit(
            X_train,
            y_train,
        )

        # -----------------------------------------------------------
        # Best model
        # -----------------------------------------------------------

        model = grid.best_estimator_

        # -----------------------------------------------------------
        # Predictions
        # -----------------------------------------------------------

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "F1 Score": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "ROC-AUC": roc_auc_score(
                y_test,
                probabilities,
            ),
            "Best Parameters": str(grid.best_params_),
        }

        trained_models[name] = model

        rows.append(metrics)

    # ---------------------------------------------------------------
    # Model comparison
    # ---------------------------------------------------------------

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "ROC-AUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Best model
    # ---------------------------------------------------------------

    best_name = results.iloc[0]["Model"]

    best_model = trained_models[best_name]

    # ---------------------------------------------------------------
    # Return everything required by UI
    # ---------------------------------------------------------------

    return (
        trained_models,
        results,
        best_name,
        best_model,
        X_test,
        y_test,
    )


# -------------------------------------------------------------------
# Validate files
# -------------------------------------------------------------------

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH}\n\n"
        "Place customer_churn_data.csv in the same folder as this Streamlit app."
    )

    st.stop()


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

df = load_data(str(DATA_PATH))


# -------------------------------------------------------------------
# Validate required columns
# -------------------------------------------------------------------

missing_columns = [column for column in FEATURES + [TARGET] if column not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")

    st.stop()


# -------------------------------------------------------------------
# Train models
# -------------------------------------------------------------------

with st.spinner("Training and evaluating classification models..."):
    (
        models,
        model_results,
        best_model_name,
        best_model,
        X_test,
        y_test,
    ) = train_models(df)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📉 ChurnGuard</h1>
        <p>
            Customer churn prediction powered by classification machine learning.
            Estimate churn probability, understand customer risk, and compare models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# KPI row
# -------------------------------------------------------------------

total_customers = len(df)

churn_rate = df[TARGET].astype(str).str.strip().str.lower().eq("yes").mean() * 100


c1, c2, c3, c4 = st.columns(4)


with c1:
    st.metric(
        "Customers",
        f"{total_customers:,}",
    )


with c2:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%",
    )


with c3:
    st.metric(
        "Best Model",
        best_model_name,
    )


with c4:
    best_auc = model_results.iloc[0]["ROC-AUC"]

    st.metric(
        "Best ROC-AUC",
        f"{best_auc:.3f}",
    )


# -------------------------------------------------------------------
# Sidebar prediction form
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Customer Profile")

    st.caption("Enter customer information to estimate churn risk.")

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
        ],
    )

    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=24,
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=1.0,
    )

    predict_clicked = st.button(
        "🔮 Predict Churn Risk",
        type="primary",
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

if predict_clicked:
    customer = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": (1 if gender == "Female" else 0),
                "Tenure": tenure,
                "MonthlyCharges": monthly_charges,
            }
        ]
    )

    # ---------------------------------------------------------------
    # Churn probability
    # ---------------------------------------------------------------

    probability = float(best_model.predict_proba(customer)[0, 1])

    # ---------------------------------------------------------------
    # Classification threshold
    # ---------------------------------------------------------------

    prediction = int(probability >= 0.50)

    # ---------------------------------------------------------------
    # Risk UI
    # ---------------------------------------------------------------

    if prediction == 1:
        risk_class = "risk-high"

        title = "⚠️ High Churn Risk"

        message = (
            "This customer has a high predicted probability of leaving. "
            "Consider proactive retention actions."
        )

    else:
        risk_class = "risk-low"

        title = "✅ Low Churn Risk"

        message = (
            "This customer currently has a lower predicted probability of leaving."
        )

    # ---------------------------------------------------------------
    # Prediction section
    # ---------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result-card {risk_class}">
            <h2>{title}</h2>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Churn Probability",
            f"{probability:.1%}",
        )

    with p2:
        st.metric(
            "Predicted Class",
            "Yes" if prediction else "No",
        )

    with p3:
        st.metric(
            "Model Used",
            best_model_name,
        )

    st.progress(probability)

    st.info(
        "This prediction is a machine-learning estimate, not a guarantee of customer behavior."
    )


# -------------------------------------------------------------------
# Dashboard tabs
# -------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Model Performance",
        "👥 Customer Analytics",
        "🧠 About the Model",
    ]
)


# -------------------------------------------------------------------
# Tab 1 - Model Performance
# -------------------------------------------------------------------

with tab1:
    st.markdown(
        '<div class="section-title">Classification Model Comparison</div>',
        unsafe_allow_html=True,
    )

    display_results = model_results[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ]
    ].copy()

    for col in display_results.columns[1:]:
        display_results[col] = display_results[col].round(3)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Best Model")

    st.success(
        f"**{best_model_name}** currently has the strongest cross-validated ROC-AUC "
        f"among the models evaluated."
    )

    st.markdown("### ROC-AUC by Model")

    chart_data = model_results.set_index("Model")[["ROC-AUC"]]

    st.bar_chart(chart_data)


# -------------------------------------------------------------------
# Tab 2 - Customer Analytics
# -------------------------------------------------------------------

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn Distribution")

        churn_counts = df[TARGET].value_counts()

        st.bar_chart(churn_counts)

    with col2:
        st.markdown("### Average Monthly Charges")

        avg_charges = df.groupby(TARGET)["MonthlyCharges"].mean()

        st.bar_chart(avg_charges)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Average Tenure")

        avg_tenure = df.groupby(TARGET)["Tenure"].mean()

        st.bar_chart(avg_tenure)

    with col4:
        st.markdown("### Average Age")

        avg_age = df.groupby(TARGET)["Age"].mean()

        st.bar_chart(avg_age)

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------
# Tab 3 - About the Model
# -------------------------------------------------------------------

with tab3:
    st.markdown("### What is this project?")

    st.write(
        """
        ChurnGuard is a binary classification machine-learning application.
        It predicts whether a customer is likely to churn (`Yes`) or stay
        (`No`) using customer-level features.
        """
    )

    st.markdown("### Features Used")

    feature_table = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Meaning": [
                "Customer age",
                "Gender encoded as Male = 0, Female = 1",
                "Customer tenure in months",
                "Customer monthly charges",
            ],
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Models Evaluated")

    st.write(
        """
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Random Forest
        """
    )

    st.markdown("### ML Pipeline")

    st.code(
        """
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Preprocessing Pipeline
        ↓
GridSearchCV
        ↓
Model Comparison
        ↓
Best Classification Model
        ↓
Churn Probability
        """,
        language="text",
    )
