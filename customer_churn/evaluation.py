"""
Evaluation and JSON-safe reporting utilities.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(pipeline: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    predictions = pipeline.predict(X_test)
    probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
            if probabilities is not None
            else np.nan
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
            if probabilities is not None
            else np.nan
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def save_json(payload: dict[str, Any], path: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(payload), file, indent=2)
