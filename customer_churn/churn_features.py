from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = [
    "Age",
    "Gender",
    "Tenure",
    "MonthlyCharges",
    "ContractType",
    "InternetService",
    "TotalCharges",
    "TechSupport",
]
TARGET = "Churn"


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-safe deterministic feature engineering for churn data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for column in ["Age", "Tenure", "MonthlyCharges", "TotalCharges"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        tenure_safe = df["Tenure"].clip(lower=1)
        df["charges_per_tenure_month"] = df["TotalCharges"] / tenure_safe
        df["total_charge_gap"] = df["TotalCharges"] - (
            df["MonthlyCharges"] * df["Tenure"]
        )
        df["is_new_customer"] = (df["Tenure"] <= 3).astype(int)
        df["is_long_tenure_customer"] = (df["Tenure"] >= 24).astype(int)
        df["is_senior_customer"] = (df["Age"] >= 60).astype(int)
        df["is_high_monthly_charge"] = (df["MonthlyCharges"] >= 75).astype(int)
        df["month_to_month_contract"] = (
            df["ContractType"].astype(str).str.lower() == "month-to-month"
        ).astype(int)
        df["has_tech_support"] = (
            df["TechSupport"].astype(str).str.lower() == "yes"
        ).astype(int)
        df["fiber_optic_customer"] = (
            df["InternetService"].astype(str).str.lower() == "fiber optic"
        ).astype(int)
        return df.drop(columns=["CustomerID"], errors="ignore")


def clean_target(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if values.isna().any():
        raise ValueError("Churn must contain only Yes/No values.")
    return values.astype(int)
