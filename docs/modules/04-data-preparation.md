
# Data Preparation

Data preparation is often the highest-effort stage of ML delivery. This module teaches
how to move from raw data to model-ready data with quality, reproducibility, and
leakage prevention.

## Data lifecycle overview

This sequence illustrates the lifecycle: business framing, data collection, feature
engineering, and dataset sizing for reliable model training.

![ML process by stages](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/ml_process_by_stages.png)

> Image explanation: This visual shows ml process by stages. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Collect data and targets](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/collect_data_init_primary_second_targets.png)

> Image explanation: This visual shows collect data and targets. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Feature engineering while collecting data](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/feature_engineering_collect_data.png)

> Image explanation: This visual shows feature engineering while collecting data. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Microsoft-style dataset size guidance](../assets/img/msft-dataset-size-guidance.svg)

> Image explanation: This visual shows microsoft-style dataset size guidance. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

## Preparation checklist

- Remove duplicates and nulls
- Validate schema and dtypes
- Split train and test sets
- Register datasets in Azure ML

## Data quality dimensions

| Dimension | Why it matters |
|---|---|
| Completeness | Missing values can bias training |
| Consistency | Schema/type drift breaks pipelines |
| Accuracy | Noisy labels reduce model ceiling |
| Timeliness | Stale data hurts production relevance |

## Minimal preprocessing pipeline

1. Remove duplicates and invalid records.
2. Define feature and target columns.
3. Handle missing values (imputation strategy).
4. Encode categorical features.
5. Split data with leakage-safe strategy.

Useful split:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```

For time-series forecasting, use chronological splits (never random shuffle across time).

The next visuals reinforce how supervised datasets are split and validated before
training, plus a dtype reference to prevent schema and conversion errors.

![Training/testing data flow](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/training_testing_data_flow.png)

> Image explanation: This visual shows training/testing data flow. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Training and test split](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/training_test_split.png)

> Image explanation: This visual shows training and test split. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Python dtype overview](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/python_dtype.png)

> Image explanation: This visual shows python dtype overview. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

## Data leakage warning

Leakage happens when future/target information enters training features. Typical causes:

- Fitting preprocessors on full data before split.
- Including post-outcome fields.
- Random split on temporal data.

Leakage creates inflated offline metrics and poor production behavior.

### Correct vs incorrect pipeline pattern

```python
# WRONG: fit scaler on full dataset before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # leaks test statistics into train
X_train, X_test = train_test_split(X_scaled, ...)

# CORRECT: fit scaler only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled = scaler.transform(X_test)         # transform test using train stats
```

Wrap this into a `sklearn.pipeline.Pipeline` so that fit/transform are always applied consistently:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipeline.fit(X_train, y_train)   # scaler.fit only on X_train inside
pipeline.score(X_test, y_test)   # scaler.transform on X_test
```

## Data contract (recommended)

Define a contract before training so all producers/consumers align:

| Field | Type | Nullable | Allowed range/pattern | Notes |
|---|---|---|---|---|
| `customer_id` | string | No | UUID regex | Unique identifier |
| `event_ts` | datetime | No | ISO-8601 | Event timestamp (UTC) |
| `label` | int | Yes | 0 or 1 | Null for inference-only rows |
| `amount` | float | No | >= 0 | Monetary feature |

## Validation gates before training

1. **Schema gate**: columns and dtypes match contract.
2. **Quality gate**: null rates, duplicate rates, outlier checks within thresholds.
3. **Drift gate**: feature distribution shift below configured limits.
4. **Leakage gate**: no post-outcome features in training set.

## Split strategies by problem type

| Problem | Recommended split | Notes |
|---|---|---|
| IID tabular classification/regression | Random train/val/test split | Use stratified split if class imbalance exists |
| Time series | Chronological split (rolling/expanding windows) | Random shuffle destroys temporal order |
| Entity-correlated data (users/devices) | Group split by entity key | Prevents entity bleed-through |
| Rare event detection | Stratified random split | Ensures minority class in each fold |

### Stratified split example

```python
from sklearn.model_selection import train_test_split

# stratify= ensures label proportions are preserved in each split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## Feature engineering patterns

- Numeric: scaling, clipping, log transforms.
- Categorical: one-hot, target encoding (with leakage-safe folds).
- Time: lags, rolling aggregates, calendar/seasonality features.
- Text: tokenization, TF-IDF, embeddings.

### Log transform example (skewed numeric)

```python
import numpy as np
import pandas as pd

df["amount_log"] = np.log1p(df["amount"])  # log1p = log(1+x), safe for 0 values
```

### Rolling aggregate (time-series features)

```python
df = df.sort_values("event_ts")
df["spend_7d"] = df.groupby("customer_id")["amount"].transform(
    lambda x: x.rolling(window=7, min_periods=1).sum()
)
```

### Target encoding with leakage protection (cross-fold)

```python
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score

enc = TargetEncoder(smoothing=10)
X_encoded = enc.fit_transform(X_train[["category"]], y_train)
# The encoder estimates within-fold statistics when used inside a cross-validation pipeline
```

## Reproducibility checklist

- Persist transformation pipeline with model artifacts.
- Version dataset snapshots and schema definitions.
- Store split seeds and split indices for exact reruns.
- Record feature list and feature order used for training.

## Quick self-check

1. Why is random split wrong for most forecasting tasks?
2. Which quality dimension is impacted by schema mismatch?
3. What is one common source of data leakage?

