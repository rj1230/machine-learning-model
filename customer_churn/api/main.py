"""
ChurnGuard FastAPI inference service.

Run locally from the project root:

    $env:PYTHONPATH = "$PWD\customer_churn"
    uv run uvicorn customer_churn.api.main:app --reload

API documentation:

    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

CUSTOMER_CHURN_DIR = Path(__file__).resolve().parent.parent

# Required so joblib can restore:
# churn_features.ChurnFeatureEngineer
if str(CUSTOMER_CHURN_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOMER_CHURN_DIR))

MODEL_PATH = CUSTOMER_CHURN_DIR / "models" / "Best_Churn_Pipeline.pkl"

MODEL_VERSION = "1.0.0"
FALLBACK_THRESHOLD = 0.50


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================


class CustomerInput(BaseModel):
    """
    Raw customer fields expected by the churn prediction pipeline.

    Field aliases preserve the original CSV/training-column names.
    The Streamlit API client may send lowercase Python names because
    populate_by_name=True is enabled.
    """

    age: int = Field(
        ...,
        alias="Age",
        ge=18,
        le=100,
        examples=[35],
    )

    gender: Literal["Male", "Female"] = Field(
        ...,
        alias="Gender",
        examples=["Female"],
    )

    tenure: int = Field(
        ...,
        alias="Tenure",
        ge=0,
        le=120,
        examples=[24],
    )

    monthly_charges: float = Field(
        ...,
        alias="MonthlyCharges",
        ge=0,
        le=10000,
        examples=[70.0],
    )

    total_charges: float = Field(
        ...,
        alias="TotalCharges",
        ge=0,
        le=100000,
        examples=[1680.0],
    )

    contract_type: Literal[
        "Month-to-Month",
        "One Year",
        "Two Year",
    ] = Field(
        ...,
        alias="ContractType",
        examples=["Month-to-Month"],
    )

    internet_service: Literal[
        "Fiber Optic",
        "DSL",
        "No",
    ] = Field(
        ...,
        alias="InternetService",
        examples=["Fiber Optic"],
    )

    tech_support: Literal["Yes", "No"] = Field(
        ...,
        alias="TechSupport",
        examples=["No"],
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 35,
                    "Gender": "Female",
                    "Tenure": 24,
                    "MonthlyCharges": 70.0,
                    "TotalCharges": 1680.0,
                    "ContractType": "Month-to-Month",
                    "InternetService": "Fiber Optic",
                    "TechSupport": "No",
                }
            ]
        },
    }


class BusinessSettings(BaseModel):
    """Optional values for retention-campaign expected-value analysis."""

    customer_lifetime_value: float = Field(
        default=500.0,
        ge=1.0,
        le=1_000_000.0,
    )

    campaign_cost: float = Field(
        default=50.0,
        ge=0.0,
        le=100_000.0,
    )

    campaign_success_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )


class PredictionRequest(BaseModel):
    """
    Request format:

    {
        "customer": {
            "age": 35,
            "gender": "Female",
            ...
        },
        "business": {
            "customer_lifetime_value": 500,
            "campaign_cost": 50,
            "campaign_success_rate": 0.30
        }
    }
    """

    customer: CustomerInput
    business: BusinessSettings = Field(default_factory=BusinessSettings)

    # None means: use the optimized threshold saved during training.
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class BusinessImpact(BaseModel):
    expected_loss_without_intervention: float
    expected_retention_benefit: float
    campaign_cost: float
    expected_net_value: float
    roi_percent: float | None
    recommendation: str


class PredictionResponse(BaseModel):
    model_name: str
    churn_probability: float
    prediction: str
    predicted_churn: bool
    risk_band: Literal["Low", "Watchlist", "High", "Critical"]
    decision_threshold: float
    recommended_action: str
    business_impact: BusinessImpact


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_name: str
    model_version: str


# ============================================================
# MODEL LIFECYCLE
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model artifact once during application startup."""

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Run customer_churn/train_model.py first."
        )

    # The training script saves a dictionary:
    #
    # {
    #     "model": calibrated_pipeline,
    #     "threshold": 0.19,
    #     "model_name": "random_forest",
    #     "raw_features": [...]
    # }
    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise RuntimeError(
            "Invalid model artifact format. "
            "Expected the dictionary generated by train_model.py."
        )

    app.state.model = artifact["model"]
    app.state.threshold = float(artifact.get("threshold", FALLBACK_THRESHOLD))
    app.state.model_name = str(artifact.get("model_name", "churn_model"))
    app.state.raw_features = artifact.get("raw_features", [])

    yield

    app.state.model = None
    app.state.threshold = None
    app.state.model_name = None
    app.state.raw_features = None


app = FastAPI(
    title="ChurnGuard Prediction API",
    description=(
        "Calibrated customer churn prediction and retention campaign value estimation."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ============================================================
# HELPERS
# ============================================================


def get_risk_band(
    probability: float,
    threshold: float,
) -> str:
    """Map churn probability to a human-readable risk category."""

    if probability >= max(0.85, threshold + 0.30):
        return "Critical"

    if probability >= max(0.65, threshold + 0.15):
        return "High"

    if probability >= threshold:
        return "Watchlist"

    return "Low"


def calculate_business_impact(
    churn_probability: float,
    settings: BusinessSettings,
) -> BusinessImpact:
    """
    Calculate simple expected-value estimates for a retention campaign.

    Expected loss:
        churn_probability × customer_lifetime_value

    Expected retention benefit:
        churn_probability × campaign_success_rate × customer_lifetime_value
    """

    expected_loss = churn_probability * settings.customer_lifetime_value

    expected_benefit = (
        churn_probability
        * settings.campaign_success_rate
        * settings.customer_lifetime_value
    )

    expected_net_value = expected_benefit - settings.campaign_cost

    roi_percent = None

    if settings.campaign_cost > 0:
        roi_percent = (expected_net_value / settings.campaign_cost) * 100

    recommendation = (
        "Intervene with a retention offer."
        if expected_net_value > 0
        else "Do not intervene; expected campaign value is negative."
    )

    return BusinessImpact(
        expected_loss_without_intervention=round(
            expected_loss,
            2,
        ),
        expected_retention_benefit=round(
            expected_benefit,
            2,
        ),
        campaign_cost=round(
            settings.campaign_cost,
            2,
        ),
        expected_net_value=round(
            expected_net_value,
            2,
        ),
        roi_percent=(round(roi_percent, 2) if roi_percent is not None else None),
        recommendation=recommendation,
    )


# ============================================================
# ENDPOINTS
# ============================================================


@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "service": "ChurnGuard Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health(request: Request):
    """Health check for local development and cloud deployment."""

    model = getattr(request.app.state, "model", None)

    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=getattr(
            request.app.state,
            "model_name",
            "unknown",
        ),
        model_version=MODEL_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(
    payload: PredictionRequest,
    request: Request,
):
    """Predict calibrated customer churn probability."""

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service unavailable.",
        )

    try:
        # Use aliases to produce the exact original raw training columns:
        #
        # Age, Gender, Tenure, MonthlyCharges, TotalCharges,
        # ContractType, InternetService, TechSupport
        raw_customer = payload.customer.model_dump(by_alias=True)

        input_df = pd.DataFrame([raw_customer])

        probability = float(model.predict_proba(input_df)[0, 1])

        saved_threshold = float(
            getattr(
                request.app.state,
                "threshold",
                FALLBACK_THRESHOLD,
            )
        )

        decision_threshold = (
            float(payload.threshold)
            if payload.threshold is not None
            else saved_threshold
        )

        predicted_churn = probability >= decision_threshold

        risk_band = get_risk_band(
            probability=probability,
            threshold=decision_threshold,
        )

        impact = calculate_business_impact(
            churn_probability=probability,
            settings=payload.business,
        )

        prediction_text = "Likely to churn" if predicted_churn else "Unlikely to churn"

        return PredictionResponse(
            model_name=getattr(
                request.app.state,
                "model_name",
                "churn_model",
            ),
            churn_probability=round(probability, 6),
            prediction=prediction_text,
            predicted_churn=predicted_churn,
            risk_band=risk_band,
            decision_threshold=round(
                decision_threshold,
                6,
            ),
            recommended_action=impact.recommendation,
            business_impact=impact,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=(f"Invalid feature values supplied to the model: {str(error)}"),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(f"Prediction failed unexpectedly: {str(error)}"),
        ) from error
