"""
Backward-compatible wrapper.

New code should import from customer_churn.features.
Existing code using:
    from churn_features import prepare_dataset
may continue to work when executed from the customer_churn directory.
"""

from .features import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    clean_target,
    engineer_features,
    normalize_binary_series,
    prepare_dataset,
    validate_features,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "LEAKAGE_COLUMNS",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "clean_target",
    "engineer_features",
    "normalize_binary_series",
    "prepare_dataset",
    "validate_features",
]
