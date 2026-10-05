"""
Telco Customer Churn — consolidated ML configuration and pipeline utilities.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, MODEL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 2
SCORING = "f1"

# Windows-safe defaults preserved from the successful tuning run.
GRID_SEARCH_N_JOBS = 1
MODEL_N_JOBS = 1


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            n_jobs=MODEL_N_JOBS,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


# Exact tuning grids used by the current tuning stage.
PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__solver": ["liblinear", "lbfgs"],
    },
    "random_forest": {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_iter": [150, 250, 350],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 30],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def grid_size(grid: dict[str, list[Any]]) -> int:
    total = 1
    for values in grid.values():
        total *= len(values)
    return total
