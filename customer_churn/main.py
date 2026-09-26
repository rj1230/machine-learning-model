from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
MODEL_PATH = BASE_DIR / "models" / "Best_Churn_Pipeline.pkl"

app = FastAPI(title="ChurnGuard API", version="2.0.0")
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Run train_model.py first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class CustomerInput(BaseModel):
    age: int = Field(ge=18, le=100)
    gender: str
    tenure: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=5000)
    contract_type: str
    internet_service: str
    total_charges: float = Field(ge=0)
    tech_support: str


def action(probability: float) -> tuple[str, str]:
    if probability >= 0.75:
        return (
            "Critical",
            "Priority retention call, plan review, and targeted incentive assessment.",
        )
    if probability >= 0.55:
        return (
            "High",
            "Offer proactive retention outreach and review support/contract options.",
        )
    if probability >= 0.30:
        return (
            "Watchlist",
            "Send a satisfaction survey or personalized engagement message.",
        )
    return "Low", "Continue normal engagement; no immediate retention action required."


@app.get("/healthz")
def healthz():
    try:
        bundle()
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(503, detail=str(exc))


@app.post("/v1/predict")
def predict(customer: CustomerInput):
    try:
        saved = bundle()
        model = saved["model"]
        frame = pd.DataFrame(
            [
                {
                    "Age": customer.age,
                    "Gender": customer.gender,
                    "Tenure": customer.tenure,
                    "MonthlyCharges": customer.monthly_charges,
                    "ContractType": customer.contract_type,
                    "InternetService": customer.internet_service,
                    "TotalCharges": customer.total_charges,
                    "TechSupport": customer.tech_support,
                }
            ]
        )
        probability = float(model.predict_proba(frame)[0, 1])
        threshold = float(saved["threshold"])
        risk, recommendation = action(probability)
        return {
            "prediction": "Likely to churn"
            if probability >= threshold
            else "Likely to stay",
            "churn_probability": probability,
            "decision_threshold": threshold,
            "risk_band": risk,
            "recommended_action": recommendation,
            "model_name": saved["model_name"],
        }
    except Exception as exc:
        raise HTTPException(500, detail="Prediction failed.") from exc
