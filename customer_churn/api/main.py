"""
ChurnGuard FastAPI inference service.

Run locally:
    uv run uvicorn api.main:app --reload

Docs:
    http://127.0.0.1:8000/docs
"""

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_model.pkl"

MODEL_VERSION = "1.0.0"
DEFAULT_THRESHOLD = 0.50


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================


class CustomerInput(BaseModel):
    """Raw customer data expected by the saved sklearn pipeline."""

    age: int = Field(
        ...,
        alias="Age",
        ge=18,
        le=100,
        examples=[35],
        description="Customer age in years.",
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        alias="Gender",
        examples=["Female"],
        description="Customer gender category used by the trained model.",
    )
    tenure: int = Field(
        ...,
        alias="Tenure",
        ge=0,
        le=120,
        examples=[24],
        description="Customer tenure in months.",
    )
    monthly_charges: float = Field(
        ...,
        alias="MonthlyCharges",
        ge=0,
        le=10000,
        examples=[70.0],
        description="Customer monthly charges.",
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
                }
            ]
        },
    }


class BusinessSettings(BaseModel):
    """Optional business assumptions used for intervention economics."""

    customer_lifetime_value: float = Field(
        default=500.0,
        ge=1,
        le=1_000_000,
        description="Estimated customer lifetime value in dollars.",
    )
    campaign_cost: float = Field(
        default=50.0,
        ge=0,
        le=100_000,
        description="Cost of one retention campaign in dollars.",
    )
    campaign_success_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Probability that an intervention retains a likely churner.",
    )


class PredictionRequest(BaseModel):
    customer: CustomerInput
    business: BusinessSettings = BusinessSettings()
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold for labeling churn risk.",
    )


class BusinessImpact(BaseModel):
    expected_loss_without_intervention: float
    expected_retention_benefit: float
    campaign_cost: float
    expected_net_value: float
    roi_percent: float | None
    recommendation: str


class PredictionResponse(BaseModel):
    model_version: str
    churn_probability: float
    predicted_churn: bool
    risk_level: Literal["low", "medium", "high"]
    decision_threshold: float
    business_impact: BusinessImpact


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str


# ============================================================
# MODEL LIFECYCLE
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the artifact once before accepting requests.
    """

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Run the training script first to create artifacts/churn_model.pkl."
        )

    app.state.model = joblib.load(MODEL_PATH)

    yield

    app.state.model = None


app = FastAPI(
    title="ChurnGuard Prediction API",
    description=(
        "Production API for calibrated customer churn prediction and "
        "retention-campaign ROI estimation."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ============================================================
# HELPERS
# ============================================================


def get_risk_level(probability: float, threshold: float) -> str:
    """
    Map calibrated probability to an interpretable risk segment.
    """

    if probability >= max(threshold, 0.75):
        return "high"

    if probability >= threshold:
        return "medium"

    return "low"


def calculate_business_impact(
    churn_probability: float,
    settings: BusinessSettings,
) -> BusinessImpact:
    """
    Expected-value calculation:

    Expected loss = P(churn) × CLV
    Expected benefit = P(churn) × campaign success rate × CLV
    Net value = expected benefit - campaign cost
    """

    expected_loss = churn_probability * settings.customer_lifetime_value

    expected_benefit = (
        churn_probability
        * settings.campaign_success_rate
        * settings.customer_lifetime_value
    )

    net_value = expected_benefit - settings.campaign_cost

    roi_percent = None
    if settings.campaign_cost > 0:
        roi_percent = (net_value / settings.campaign_cost) * 100

    recommendation = "intervene" if net_value > 0 else "do_not_intervene"

    return BusinessImpact(
        expected_loss_without_intervention=round(expected_loss, 2),
        expected_retention_benefit=round(expected_benefit, 2),
        campaign_cost=round(settings.campaign_cost, 2),
        expected_net_value=round(net_value, 2),
        roi_percent=round(roi_percent, 2) if roi_percent is not None else None,
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
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health(request: Request):
    """Health check for Docker, cloud deployment, or load balancers."""

    is_loaded = getattr(request.app.state, "model", None) is not None

    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
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
    """
    Predict calibrated churn probability from raw customer attributes.
    """

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Service is temporarily unavailable.",
        )

    try:
        # Preserve original feature names required by the sklearn pipeline.
        raw_customer = payload.customer.model_dump(by_alias=True)

        input_df = pd.DataFrame([raw_customer])

        probability = float(model.predict_proba(input_df)[0, 1])

        predicted_churn = probability >= payload.threshold

        impact = calculate_business_impact(
            churn_probability=probability,
            settings=payload.business,
        )

        return PredictionResponse(
            model_version=MODEL_VERSION,
            churn_probability=round(probability, 6),
            predicted_churn=predicted_churn,
            risk_level=get_risk_level(
                probability=probability,
                threshold=payload.threshold,
            ),
            decision_threshold=payload.threshold,
            business_impact=impact,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values for the model: {str(exc)}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed unexpectedly.",
        ) from exc
