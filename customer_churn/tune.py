"""
Telco Customer Churn — consolidated hyperparameter tuning.

Run:
    python -m customer_churn.tune

This preserves the current Windows-safe grids:
- Logistic Regression: 14 combinations / 70 fits
- Random Forest: 144 combinations / 720 fits
- HistGradientBoosting: 324 combinations / 1620 fits

All GridSearchCV jobs use n_jobs=1 to avoid the previous Windows resource
exhaustion.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .evaluation import evaluate_model, save_json
from .features import MODEL_FEATURES, prepare_dataset
from .pipeline import (
    CV_FOLDS,
    GRID_SEARCH_N_JOBS,
    PARAMETER_GRIDS,
    RANDOM_STATE,
    SCORING,
    TEST_SIZE,
    build_models,
    build_pipeline,
    grid_size,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco_customer_churn.xlsx"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

TUNED_MODEL_PATH = MODEL_DIR / "best_tuned_churn_model.pkl"
RESULTS_PATH = REPORT_DIR / "hyperparameter_results.csv"
METRICS_PATH = REPORT_DIR / "tuned_model_metrics.json"


def main() -> None:
    print("=" * 80)
    print("TELCO CUSTOMER CHURN — CONSOLIDATED WINDOWS-SAFE TUNING")
    print("=" * 80)
    print(f"\nGridSearchCV n_jobs: {GRID_SEARCH_N_JOBS}")
    print("Random Forest n_jobs: 1")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    all_results = []
    best_overall = None

    for model_name, model in models.items():
        grid = PARAMETER_GRIDS[model_name]
        combinations = grid_size(grid)
        total_fits = combinations * CV_FOLDS

        print("\n" + "=" * 80)
        print(f"TUNING — {model_name.upper()}")
        print("=" * 80)
        print(f"Parameter combinations: {combinations}")
        print(f"CV folds: {CV_FOLDS}")
        print(f"Total model fits: {total_fits}")
        print(f"GridSearchCV workers: {GRID_SEARCH_N_JOBS}")
        print("\nStarting GridSearchCV...")

        pipeline = build_pipeline(model)

        start = time.perf_counter()
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=SCORING,
            cv=cv,
            n_jobs=GRID_SEARCH_N_JOBS,
            refit=True,
            verbose=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        metrics = evaluate_model(search.best_estimator_, X_test, y_test)
        cv_f1 = float(search.best_score_)

        print(f"\nCompleted in {elapsed:.1f} seconds.")
        print(f"Best CV F1: {cv_f1:.4f}")
        print("\nBest parameters:")
        for key, value in search.best_params_.items():
            print(f"  {key}: {value}")

        print("\nHeld-out test results:")
        for key in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]:
            print(f"{key.replace('_', ' ').title():20s}: {metrics[key]:.4f}")

        row = {
            "model": model_name,
            "best_cv_f1": cv_f1,
            "test_accuracy": metrics["accuracy"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "test_average_precision": metrics["average_precision"],
            "fit_seconds": elapsed,
            "best_params": repr(search.best_params__ if False else search.best_params_),
        }
        all_results.append(row)

        if best_overall is None or cv_f1 > best_overall["cv_f1"]:
            best_overall = {
                "model": model_name,
                "cv_f1": cv_f1,
                "test_metrics": metrics,
                "params": search.best_params_,
                "pipeline": search.best_estimator_,
            }

    comparison = (
        pd.DataFrame(all_results)
        .sort_values(
            ["best_cv_f1", "test_f1"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    comparison.to_csv(RESULTS_PATH, index=False)
    joblib.dump(best_overall["pipeline"], TUNED_MODEL_PATH)

    payload = {
        "selection_metric": "cross_validated_f1",
        "models": all_results,
        "best_model": {
            "name": best_overall["model"],
            "cv_f1": best_overall["cv_f1"],
            "test_metrics": best_overall["test_metrics"],
            "best_params": best_overall["params"],
        },
        "resource_configuration": {
            "grid_search_n_jobs": GRID_SEARCH_N_JOBS,
            "random_forest_n_jobs": 1,
        },
    }
    save_json(payload, METRICS_PATH)

    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("BEST TUNED MODEL")
    print("=" * 80)
    print(f"Model:       {best_overall['model']}")
    print(f"CV F1:       {best_overall['cv_f1']:.4f}")
    print(f"Test F1:     {best_overall['test_metrics']['f1']:.4f}")
    print(f"Test ROC-AUC:{best_overall['test_metrics']['roc_auc']:.4f}")
    print(f"\nSaved model: {TUNED_MODEL_PATH}")
    print(f"Saved CSV:   {RESULTS_PATH}")
    print(f"Saved JSON:  {METRICS_PATH}")


if __name__ == "__main__":
    main()
