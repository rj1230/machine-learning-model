"""
Telco Customer Churn — consolidated baseline training.

Run:
    python -m customer_churn.train
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .evaluation import evaluate_model, save_json
from .features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    prepare_dataset,
)
from .pipeline import (
    CV_FOLDS,
    GRID_SEARCH_N_JOBS,
    RANDOM_STATE,
    TEST_SIZE,
    build_models,
    build_pipeline,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco_customer_churn.xlsx"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "best_churn_model.pkl"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
METRICS_PATH = REPORT_DIR / "metrics.json"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> None:
    print("=" * 80)
    print("TELCO CUSTOMER CHURN — CONSOLIDATED BASELINE TRAINING")
    print("=" * 80)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = prepare_dataset(df.copy())

    print(f"\nDataset shape: {df.shape}")
    print(f"Prepared feature shape: {X.shape}")
    print(f"Prepared target shape:  {y.shape}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    print("\nSenior Citizen validation:")
    print(f"  Valid values: {X['Senior Citizen'].notna().sum()}")
    print(f"  Missing values: {X['Senior Citizen'].isna().sum()}")
    print(
        "  Unique values:",
        sorted(X["Senior Citizen"].dropna().unique().tolist()),
    )

    X = X[MODEL_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    results = []
    detailed = {}
    fitted = {}

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"TRAINING — {model_name}")
        print("=" * 80)

        pipeline = build_pipeline(model)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=GRID_SEARCH_N_JOBS,
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        print(f"CV F1:               {cv_mean:.4f} +/- {cv_std:.4f}")
        print(f"Accuracy:            {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
        print(f"Precision:            {metrics['precision']:.4f}")
        print(f"Recall:               {metrics['recall']:.4f}")
        print(f"F1:                   {metrics['f1']:.4f}")
        print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
        print(f"Average Precision:    {metrics['average_precision']:.4f}")
        print("\nConfusion matrix:")
        print(np.array(metrics["confusion_matrix"]))

        results.append(
            {
                "model": model_name,
                **{k: metrics[k] for k in [
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "average_precision",
                ]},
                "cv_f1_mean": cv_mean,
                "cv_f1_std": cv_std,
            }
        )

        detailed[model_name] = {
            "test_metrics": metrics,
            "cv_f1_mean": cv_mean,
            "cv_f1_std": cv_std,
        }
        fitted[model_name] = pipeline

    comparison = (
        pd.DataFrame(results)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )

    best_name = str(comparison.iloc[0]["model"])
    best_pipeline = fitted[best_name]

    joblib.dump(best_pipeline, MODEL_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)

    payload = {
        "dataset": {
            "rows": len(df),
            "columns": len(df.columns),
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "churn_rate": float(y.mean()),
        },
        "features": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "total_model_features": len(MODEL_FEATURES),
        },
        "models": detailed,
        "best_model": {
            "name": best_name,
            "test_f1": float(comparison.iloc[0]["f1"]),
            "test_roc_auc": float(comparison.iloc[0]["roc_auc"]),
            "cv_f1": float(comparison.iloc[0]["cv_f1_mean"]),
        },
    }
    save_json(payload, METRICS_PATH)

    print("\n" + "=" * 80)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model: {best_name}")
    print(f"Model:      {MODEL_PATH}")
    print(f"Comparison: {COMPARISON_PATH}")
    print(f"Metrics:    {METRICS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print("\nERROR:")
        print(str(exc))
        sys.exit(1)
