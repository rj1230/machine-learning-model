from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib

os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from churn_features import ChurnFeatureEngineer, RAW_FEATURES, TARGET, clean_target

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn_data.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = MODELS_DIR / "Best_Churn_Pipeline.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
RANDOM_STATE = 42

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)


def build_preprocessor() -> ColumnTransformer:
    numeric = [
        "Age",
        "Tenure",
        "MonthlyCharges",
        "TotalCharges",
        "charges_per_tenure_month",
        "total_charge_gap",
        "is_new_customer",
        "is_long_tenure_customer",
        "is_senior_customer",
        "is_high_monthly_charge",
        "month_to_month_contract",
        "has_tech_support",
        "fiber_optic_customer",
    ]
    categorical = ["Gender", "ContractType", "InternetService", "TechSupport"]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def pipeline_for(estimator) -> Pipeline:
    return Pipeline(
        [
            ("features", ChurnFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def threshold_for(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.15, 0.76, 0.01):
        predicted = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predicted, zero_division=0)
        recall = recall_score(y_true, predicted, zero_division=0)
        f1 = f1_score(y_true, predicted, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "utility": float(0.65 * recall + 0.35 * precision),
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(["utility", "f1"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def save_plots(y_test, probabilities, predictions, model, X_test, thresholds):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    observed, predicted = calibration_curve(y_test, probabilities, n_bins=8)
    fig, ax = plt.subplots()
    ax.plot(predicted, observed, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
        title="Calibration Curve",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    for column in ["precision", "recall", "f1"]:
        ax.plot(thresholds["threshold"], thresholds[column], label=column.title())
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Threshold Analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "threshold_analysis.png", dpi=150)
    plt.close(fig)
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=12,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    table = pd.DataFrame(
        {"Feature": X_test.columns, "Importance": importance.importances_mean}
    ).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(table["Feature"], table["Importance"], color="#2563eb")
    ax.set_title("Permutation Feature Importance (ROC-AUC)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    return table.sort_values("Importance", ascending=False).head(10)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    missing = set(RAW_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    X, y = df[RAW_FEATURES].copy(), clean_target(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": (
            pipeline_for(
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                )
            ),
            {"model__C": [0.01, 0.05, 0.2, 1, 5, 20]},
        ),
        "random_forest": (
            pipeline_for(
                RandomForestClassifier(
                    random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1
                )
            ),
            {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 5, 8, 12],
                "model__min_samples_leaf": [1, 2, 5],
            },
        ),
        "hist_gradient_boosting": (
            pipeline_for(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            {
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_iter": [150, 250, 350],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 35],
            },
        ),
    }
    results, best = [], None
    for name, (estimator, params) in candidates.items():
        search = RandomizedSearchCV(
            estimator,
            params,
            n_iter=min(10, int(np.prod([len(v) for v in params.values()]))),
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X_train, y_train)
        results.append(
            {
                "model": name,
                "cv_roc_auc": float(search.best_score_),
                "best_parameters": search.best_params_,
            }
        )
        if best is None or search.best_score_ > best[0]:
            best = (
                search.best_score_,
                name,
                search.best_estimator_,
                search.best_params_,
            )
    _, name, selected, params = best
    calibrated = CalibratedClassifierCV(
        estimator=selected, method="isotonic", cv=3, n_jobs=1
    )
    oof = cross_val_predict(
        calibrated, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    threshold, threshold_table = threshold_for(y_train.to_numpy(), oof)
    calibrated.fit(X_train, y_train)
    probabilities = calibrated.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    top = save_plots(
        y_test, probabilities, predictions, calibrated, X_test, threshold_table
    )
    metrics = {
        "model_name": name,
        "best_parameters": params,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_pr_auc": float(average_precision_score(y_test, probabilities)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "mean_brier_score": float(brier_score_loss(y_test, probabilities)),
        "decision_threshold": threshold,
        "cv_model_comparison": results,
        "class_distribution": {
            "stay": int((y == 0).sum()),
            "churn": int((y == 1).sum()),
        },
        "classification_report": classification_report(
            y_test, predictions, target_names=["No churn", "Churn"], output_dict=True
        ),
        "top_features": top.to_dict(orient="records"),
        "dataset_rows": int(len(df)),
        "raw_features": RAW_FEATURES,
    }
    joblib.dump(
        {
            "model": calibrated,
            "threshold": threshold,
            "model_name": name,
            "raw_features": RAW_FEATURES,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )
    print("Training complete")
    print(f"Best model: {name}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"Test F1: {metrics['test_f1']:.4f}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
