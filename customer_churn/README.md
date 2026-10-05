# Telco Customer Churn — Consolidated ML Pipeline

## Dataset

Place the dataset here:

`customer_churn/Telco_customer_churn.xlsx`

Target:

`Churn Label`

## Current architecture

```text
Telco_customer_churn.xlsx
        |
        v
features.py
  cleaning + feature engineering
        |
        v
pipeline.py
  preprocessing + model definitions + tuning grids
        |
        +--------------------+
        |                    |
        v                    v
     train.py             tune.py
  baseline models      GridSearchCV
        |                    |
        +---------+----------+
                  |
                  v
             threshold.py
                  |
                  v
             final reports
```

## Commands

Baseline:

```powershell
python -m customer_churn.train
```

Hyperparameter tuning:

```powershell
python -m customer_churn.tune
```

Threshold optimization:

```powershell
python -m customer_churn.threshold
```

Everything:

```powershell
python -m customer_churn.run_all
```

## Current model configuration

### Baseline models

- LogisticRegression
  - max_iter=3000
  - class_weight="balanced"
  - random_state=42

- RandomForestClassifier
  - n_estimators=400
  - class_weight="balanced"
  - n_jobs=1
  - random_state=42

- HistGradientBoostingClassifier
  - max_iter=300
  - learning_rate=0.05
  - max_leaf_nodes=31
  - random_state=42

### Validation

- train/test = 80/20
- stratified split
- random_state=42
- 5-fold StratifiedKFold
- primary CV scoring = F1

### Windows-safe tuning

GridSearchCV uses `n_jobs=1`.

Current grids:

Logistic Regression:
- C = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
- solver = [liblinear, lbfgs]
- 14 combinations / 70 fits

Random Forest:
- n_estimators = [300, 500]
- max_depth = [None, 8, 12, 16]
- min_samples_split = [2, 5, 10]
- min_samples_leaf = [1, 2, 4]
- max_features = [sqrt, log2]
- 144 combinations / 720 fits

HistGradientBoosting:
- learning_rate = [0.03, 0.05, 0.08, 0.1]
- max_iter = [150, 250, 350]
- max_leaf_nodes = [15, 31, 63]
- min_samples_leaf = [10, 20, 30]
- l2_regularization = [0.0, 0.1, 1.0]
- 324 combinations / 1620 fits

## Feature schema

The current working schema contains 22 numeric and 11 categorical model
features. Leakage columns such as Churn Value, Churn Score, CLTV and Churn
Reason are excluded.

## Important model-selection rule

Do not reuse the old 0.62 threshold automatically after tuning. The threshold
belongs to a specific model artifact. Run threshold.py after the winning tuned
model is selected.
