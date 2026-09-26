#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NYC Airbnb Room Type Classification
-----------------------------------
Production-grade ML pipeline:

1. Load data
2. Exploratory Data Analysis
3. Data cleaning
4. Feature engineering
5. Train/test split (BEFORE any fitting)
6. Preprocessing pipeline (fit only on train)
7. Model comparison
8. Optuna hyperparameter tuning with pruning
9. Calibration (isotonic) + reliability diagram + Brier score
10. Model evaluation
11. Model persistence (single artifact: capper → preprocessor → classifier)
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import optuna
from optuna.exceptions import TrialPruned

# ============================================================
# 2. CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.33
N_SPLITS = 5

DATA_PATH = Path("nyc.csv")
MODEL_PATH = Path("Best_Model_Pipeline.pkl")

sns.set_theme(
    style="whitegrid",
    context="notebook",
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("NYC AIRBNB ROOM TYPE CLASSIFICATION")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Shape: {df.shape}")


# ============================================================
# 4. BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nColumn names:")
print(df.columns.tolist())

print(f"\nNumber of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]:,}")


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values found.")
else:
    print(missing)


# ============================================================
# 6. TARGET VARIABLE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TARGET VARIABLE ANALYSIS")
print("=" * 70)

print("\nRoom types:")
print(df["room_type"].unique())

print("\nRoom type distribution:")
print(df["room_type"].value_counts())


plt.figure(figsize=(9, 5))

ax = sns.countplot(
    data=df,
    x="room_type",
    order=df["room_type"].value_counts().index,
)

plt.title(
    "Distribution of Airbnb Room Types",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Room Type")
plt.ylabel("Number of Listings")

plt.xticks(rotation=10)

for container in ax.containers:
    ax.bar_label(container, fmt="%d")

plt.tight_layout()
plt.show()


# ============================================================
# 7. NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

numeric_cols = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

df[numeric_cols].hist(
    bins=30,
    figsize=(14, 9),
    edgecolor="black",
)

plt.suptitle(
    "Distribution of Numerical Features",
    fontsize=18,
    fontweight="bold",
)

plt.tight_layout()
plt.show()


# ============================================================
# 8. NEIGHBOURHOOD GROUP ANALYSIS
# ============================================================

plt.figure(figsize=(9, 5))

ax = sns.countplot(
    data=df,
    x="neighbourhood_group",
    order=df["neighbourhood_group"].value_counts().index,
)

plt.title(
    "Listings by Neighbourhood Group",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Neighbourhood Group")
plt.ylabel("Number of Listings")

for container in ax.containers:
    ax.bar_label(container, fmt="%d")

plt.tight_layout()
plt.show()


# ============================================================
# 9. PRICE VS ROOM TYPE
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="room_type",
    y="price",
)

plt.title(
    "Price Distribution by Room Type",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Room Type")
plt.ylabel("Price")

plt.ylim(
    0,
    df["price"].quantile(0.99) * 1.05,
)

plt.tight_layout()
plt.show()


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

correlation_cols = numeric_cols + [
    "latitude",
    "longitude",
]

corr = df[correlation_cols].corr()

plt.figure(figsize=(11, 8))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
)

plt.title(
    "Correlation Matrix",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()
plt.show()


# ============================================================
# 11. GEOGRAPHICAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(11, 8))

sns.scatterplot(
    data=df,
    x="longitude",
    y="latitude",
    hue="room_type",
    alpha=0.45,
    s=20,
)

plt.title(
    "NYC Airbnb Listings by Location and Room Type",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.legend(
    title="Room Type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()
plt.show()


# ============================================================
# 12. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Create a separate cleaned dataframe
df_clean = df.drop(
    columns=[
        "id",
        "name",
        "host_name",
        "last_review",
        "host_id",
    ],
    errors="ignore",
).copy()

# NOTE: reviews_per_month imputation will be handled INSIDE the pipeline
# to avoid leakage. We do NOT fill it here anymore.

print("\nColumns removed:")
print(
    [
        "id",
        "name",
        "host_name",
        "last_review",
        "host_id",
    ]
)

print("\nCleaned dataset shape:")
print(df_clean.shape)


# ============================================================
# 13. OUTLIER HANDLING (LEAKAGE-FREE)
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER HANDLING (LEAKAGE-FREE)")
print("=" * 70)

# Compute caps on the FULL dataset ONLY for EDA reporting.
# The actual capping will be fit on X_train inside the pipeline.

price_cap_eda = df_clean["price"].quantile(0.99)
night_cap_eda = df_clean["minimum_nights"].quantile(0.99)

print(f"Price 99th percentile (EDA): {price_cap_eda:.2f}")
print(f"Minimum nights 99th percentile (EDA): {night_cap_eda:.2f}")

# Do NOT apply clipping here. It will be done in the pipeline.


# ============================================================
# 14. FEATURES AND TARGET
# ============================================================

X = df_clean.drop(columns=["room_type"])
y = df_clean["room_type"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 15. TRAIN / TEST SPLIT (BEFORE ANY FIT)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")


# ============================================================
# 16. FEATURE DEFINITIONS
# ============================================================

numerical_cols = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

categorical_cols = [
    "neighbourhood_group",
    "neighbourhood",
]


# ============================================================
# 17. CUSTOM OUTLIER CAPPER TRANSFORMER
# ============================================================

from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps numerical features at specified quantiles.
    Fit on training data only, transform on any data.
    Also fills missing reviews_per_month with 0 (domain logic).
    """

    def __init__(self, quantile=0.99, cols_to_cap=None):
        self.quantile = quantile
        self.cols_to_cap = cols_to_cap
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X = X.copy()
        if self.cols_to_cap is None:
            self.cols_to_cap = X.select_dtypes(include=[np.number]).columns.tolist()

        self.upper_bounds_ = {}
        for col in self.cols_to_cap:
            if col in X.columns:
                self.upper_bounds_[col] = X[col].quantile(self.quantile)
        return self

    def transform(self, X):
        X = X.copy()
        for col, bound in self.upper_bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=bound)
        # Domain-specific: missing reviews_per_month means no reviews → 0
        if "reviews_per_month" in X.columns:
            X["reviews_per_month"] = X["reviews_per_month"].fillna(0)
        return X


# ============================================================
# 18. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "power_transformer",
            PowerTransformer(method="yeo-johnson"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numerical_cols,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols,
        ),
    ]
)


print("\nPreprocessor created successfully.")


# ============================================================
# 19. MODEL DEFINITIONS
# ============================================================

models = {
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_estimators=200,
        n_jobs=-1,
    ),
}


# ============================================================
# 20. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = []

for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            (
                "capper",
                OutlierCapper(quantile=0.99, cols_to_cap=["price", "minimum_nights"]),
            ),
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring=[
            "accuracy",
            "f1_macro",
        ],
        n_jobs=1,
    )

    accuracy_mean = scores["test_accuracy"].mean()
    accuracy_std = scores["test_accuracy"].std()
    f1_mean = scores["test_f1_macro"].mean()
    f1_std = scores["test_f1_macro"].std()

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_mean,
            "Accuracy Std": accuracy_std,
            "Macro F1": f1_mean,
            "F1 Std": f1_std,
        }
    )

    print(f"\n{name}")
    print(f"Accuracy : {accuracy_mean:.4f} (± {accuracy_std:.4f})")
    print(f"Macro F1 : {f1_mean:.4f} (± {f1_std:.4f})")


results_df = (
    pd.DataFrame(results)
    .sort_values(
        "Macro F1",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\n" + "-" * 70)
print("MODEL PERFORMANCE SUMMARY")
print("-" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 21. OPTUNA HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("OPTUNA HYPERPARAMETER TUNING")
print("=" * 70)


def objective(trial: TrialPruned) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 8, 30, step=2),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    clf = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "capper",
                OutlierCapper(quantile=0.99, cols_to_cap=["price", "minimum_nights"]),
            ),
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        return_train_score=False,
    )

    f1_scores = scores["test_score"]
    mean_f1 = f1_scores.mean()

    # Report intermediate result for pruning
    trial.report(mean_f1, trial.number)

    if trial.should_prune():
        raise TrialPruned()

    return mean_f1


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)

study.optimize(
    objective,
    n_trials=30,
    timeout=None,
    show_progress_bar=True,
)


print("\n" + "=" * 70)
print("OPTUNA BEST TRIAL")
print("=" * 70)

print(f"\nBest Macro-F1 (CV): {study.best_value:.4f}")
print("\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# ============================================================
# 22. BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

best_rf = RandomForestClassifier(
    **study.best_params,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

best_model_pipeline = Pipeline(
    steps=[
        (
            "capper",
            OutlierCapper(quantile=0.99, cols_to_cap=["price", "minimum_nights"]),
        ),
        ("preprocessor", preprocessor),
        ("classifier", best_rf),
    ]
)

best_model_pipeline.fit(X_train, y_train)


# ============================================================
# 23. CALIBRATION (ISOTONIC)
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION (ISOTONIC)")
print("=" * 70)

calibrated_clf = CalibratedClassifierCV(
    estimator=best_model_pipeline,
    method="isotonic",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
)

calibrated_clf.fit(X_train, y_train)


# ============================================================
# 24. TEST SET EVALUATION
# ============================================================

y_pred = calibrated_clf.predict(X_test)
y_proba = calibrated_clf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Test Macro-F1 : {macro_f1:.4f}")


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
    )
)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=calibrated_clf.classes_,
    yticklabels=calibrated_clf.classes_,
    linewidths=0.5,
)

plt.title(
    "Confusion Matrix - Calibrated Random Forest",
    fontsize=16,
    fontweight="bold",
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.show()


# ============================================================
# 27. CALIBRATION CURVE + BRIER SCORE
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS")
print("=" * 70)

# For multi-class, compute calibration per class (One-vs-Rest)
classes = calibrated_clf.classes_
n_classes = len(classes)

fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
if n_classes == 1:
    axes = [axes]

brier_scores = []

for idx, cls in enumerate(classes):
    # Binary indicator for this class
    y_test_binary = (y_test == cls).astype(int)
    y_proba_cls = y_proba[:, idx]

    # Calibration curve
    frac_pos, prob_pred = calibration_curve(y_test_binary, y_proba_cls, n_bins=10)

    # Brier score
    brier = brier_score_loss(y_test_binary, y_proba_cls)
    brier_scores.append(brier)

    ax = axes[idx]
    ax.plot(prob_pred, frac_pos, "s-", label="Calibrated")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Class: {cls}\nBrier score: {brier:.4f}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.suptitle("Reliability Diagrams (One-vs-Rest)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

print("\nBrier scores per class:")
for cls, brier in zip(classes, brier_scores):
    print(f"  {cls}: {brier:.4f}")
print(f"  Mean Brier: {np.mean(brier_scores):.4f}")


# ============================================================
# 28. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    calibrated_clf,
    MODEL_PATH,
)

print(f"\nModel saved successfully to:\n{MODEL_PATH.resolve()}")


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(
    f"""
Dataset Shape       : {df.shape}
Training Samples    : {len(X_train):,}
Testing Samples     : {len(X_test):,}

Best Model          : Calibrated Random Forest (Optuna-tuned)
Best CV Macro-F1    : {study.best_value:.4f}
Test Accuracy       : {accuracy:.4f}
Test Macro-F1       : {macro_f1:.4f}
Mean Brier Score    : {np.mean(brier_scores):.4f}

Saved Model         : {MODEL_PATH}
"""
)

print("=" * 70)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
