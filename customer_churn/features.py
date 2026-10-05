"""
Telco Customer Churn — feature engineering.

Public API:
    prepare_dataset(df) -> (X, y)

The function intentionally accepts only the dataframe.  This matches the
current working project contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "Churn Label"

LEAKAGE_COLUMNS = [
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

# The current working model schema.
NUMERIC_FEATURES = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Paperless Billing",
    "Monthly Charges",
    "Total Charges",
    "charges_per_tenure_month",
    "is_new_customer",
    "is_long_tenure_customer",
    "is_senior_customer",
    "is_high_monthly_charge",
    "month_to_month_contract",
    "long_contract",
    "has_tech_support",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "fiber_optic_customer",
    "has_streaming",
    "electronic_payment",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_binary_series(series: pd.Series) -> pd.Series:
    """
    Convert common yes/no, true/false, 1/0 representations to 0/1.
    Unknown values become NaN.
    """
    mapping = {
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    cleaned = _clean_text(series).str.lower()
    mapped = cleaned.map(mapping)

    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.fillna(numeric)

    return mapped.astype("float64")


def clean_target(series: pd.Series) -> pd.Series:
    """Normalize Churn Label to integer 0/1."""
    cleaned = _clean_text(series).str.lower()

    mapping = {
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    target = cleaned.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    target = target.fillna(numeric)

    if target.isna().any():
        bad = series[target.isna()].drop_duplicates().tolist()
        raise ValueError(f"Unrecognized target values in '{TARGET_COLUMN}': {bad}")

    target = target.astype(int)

    if not set(target.unique()).issubset({0, 1}):
        raise ValueError(
            f"'{TARGET_COLUMN}' must contain only 0/1 after cleaning. "
            f"Found: {sorted(target.unique().tolist())}"
        )

    return target


def _service_flag(series: pd.Series) -> pd.Series:
    """
    Service fields use values such as Yes/No/No internet service.
    Only an active Yes counts as 1.
    """
    return (
        _clean_text(series)
        .str.lower()
        .eq("yes")
        .astype(float)
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete current feature set.

    Leakage columns are excluded. Original categorical service fields remain
    categorical; selected behavioral signals are added as numeric features.
    """
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    required = [
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Paperless Billing",
        "Monthly Charges",
        "Total Charges",
        "Contract",
        "Tech Support",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Internet Service",
        "Streaming TV",
        "Streaming Movies",
    ]

    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(
            "Required columns missing from dataset:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # Normalize numeric source columns.
    for column in ["Tenure Months", "Monthly Charges", "Total Charges"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    # Explicitly fix Senior Citizen to 0/1.
    work["Senior Citizen"] = normalize_binary_series(work["Senior Citizen"])

    # Explicitly normalize the binary source columns used as model features.
    for column in ["Partner", "Dependents", "Phone Service", "Paperless Billing"]:
        work[column] = normalize_binary_series(work[column])

    # Behavioral / service indicators.
    work["charges_per_tenure_month"] = (
        work["Total Charges"]
        / work["Tenure Months"].replace(0, np.nan)
    )
    work["charges_per_tenure_month"] = work["charges_per_tenure_month"].replace(
        [np.inf, -np.inf], np.nan
    )

    work["is_new_customer"] = (work["Tenure Months"] <= 6).astype(float)
    work["is_long_tenure_customer"] = (work["Tenure Months"] >= 48).astype(float)
    work["is_senior_customer"] = work["Senior Citizen"].fillna(0)
    work["is_high_monthly_charge"] = (
        work["Monthly Charges"] >= work["Monthly Charges"].median()
    ).astype(float)

    contract_clean = _clean_text(work["Contract"]).str.lower()
    work["month_to_month_contract"] = contract_clean.eq(
        "month-to-month"
    ).astype(float)
    work["long_contract"] = contract_clean.isin(
        ["one year", "two year"]
    ).astype(float)

    work["has_tech_support"] = _service_flag(work["Tech Support"])
    work["has_online_security"] = _service_flag(work["Online Security"])
    work["has_online_backup"] = _service_flag(work["Online Backup"])
    work["has_device_protection"] = _service_flag(work["Device Protection"])

    internet_clean = _clean_text(work["Internet Service"]).str.lower()
    work["fiber_optic_customer"] = internet_clean.eq("fiber optic").astype(float)

    work["has_streaming"] = (
        _clean_text(work["Streaming TV"]).str.lower().eq("yes")
        | _clean_text(work["Streaming Movies"]).str.lower().eq("yes")
    ).astype(float)

    payment_clean = _clean_text(work["Payment Method"]).str.lower()
    work["electronic_payment"] = payment_clean.isin(
        [
            "electronic check",
            "mailed check",
        ]
    ).astype(float)

    # Remove leakage columns if present.
    work = work.drop(
        columns=[c for c in LEAKAGE_COLUMNS if c in work.columns],
        errors="ignore",
    )

    return work


def validate_features(X: pd.DataFrame) -> None:
    missing = [c for c in MODEL_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(
            "Engineered features are missing:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    senior = X["Senior Citizen"]
    if senior.notna().sum() == 0:
        raise ValueError("'Senior Citizen' contains no valid values.")

    values = set(senior.dropna().unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(
            "'Senior Citizen' must contain only 0/1 values. "
            f"Found: {values}"
        )


def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target together.

    IMPORTANT: this function takes exactly one argument.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    features = engineer_features(df)
    target = clean_target(df[TARGET_COLUMN])

    validate_features(features)

    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")

    return features[MODEL_FEATURES].copy(), target
