"""
Telco Customer Churn — threshold optimization.

IMPORTANT:
This evaluates the currently selected model artifact. It does not assume
that the old 0.62 threshold remains valid after tuning.

Run:
    python -m customer_churn.threshold

By default it uses best_tuned_churn_model.pkl when available, otherwise the
baseline best_churn_model.pkl.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from .features import MODEL_FEATURES, prepare_dataset
from .pipeline import RANDOM_STATE, TEST_SIZE

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco_customer_churn.xlsx"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

TUNED_MODEL_PATH = MODEL_DIR / "best_tuned_churn_model.pkl"
BASELINE_MODEL_PATH = MODEL_DIR / "best_churn_model.pkl"

ANALYSIS_PATH = REPORT_DIR / "threshold_analysis.csv"
OPTIMAL_PATH = REPORT_DIR / "optimal_threshold.json"


def metrics_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = (
        TUNED_MODEL_PATH
        if TUNED_MODEL_PATH.exists()
        else BASELINE_MODEL_PATH
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "No model artifact found. Run training or tuning first."
        )

    print("=" * 80)
    print("TELCO CUSTOMER CHURN — THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Model: {model_path}")

    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    X, y = prepare_dataset(df.copy())
    X = X[MODEL_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_test)[:, 1]

    rows = []
    for threshold in np.round(np.arange(0.20, 0.801, 0.01), 2):
        rows.append(
            metrics_at_threshold(
                y_test,
                probabilities,
                float(threshold),
            )
        )

    analysis = pd.DataFrame(rows)

    best_f1 = analysis.loc[analysis["f1"].idxmax()]
    best_recall = analysis.loc[analysis["recall"].idxmax()]
    best_balanced = analysis.loc[
        analysis["balanced_accuracy"].idxmax()
    ]

    default = metrics_at_threshold(y_test, probabilities, 0.50)

    analysis.to_csv(ANALYSIS_PATH, index=False)

    optimal_payload = {
        "model_path": str(model_path),
        "selection_metric": "f1",
        "best_f1_threshold": best_f1.to_dict(),
        "best_recall_threshold": best_recall.to_dict(),
        "best_balanced_accuracy_threshold": best_balanced.to_dict(),
        "default_threshold_0_50": default,
        "probability_summary": {
            "min": float(probabilities.min()),
            "max": float(probabilities.max()),
            "mean": float(probabilities.mean()),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(
                average_precision_score(y_test, probabilities)
            ),
        },
        "warning": (
            "Threshold was selected on the held-out split. "
            "For production use, validate the threshold with cross-validation "
            "or a separate validation set before locking it."
        ),
    }

    with open(OPTIMAL_PATH, "w", encoding="utf-8") as file:
        json.dump(optimal_payload, file, indent=2)

    print("\nDefault threshold 0.50:")
    print(pd.Series(default).to_string())

    print("\nBest F1 threshold:")
    print(best_f1.to_string())

    print("\nBest balanced accuracy threshold:")
    print(best_balanced.to_string())

    print("\nBest recall threshold:")
    print(best_recall.to_string())

    print("\nSaved:")
    print(ANALYSIS_PATH)
    print(OPTIMAL_PATH)


if __name__ == "__main__":
    main()
