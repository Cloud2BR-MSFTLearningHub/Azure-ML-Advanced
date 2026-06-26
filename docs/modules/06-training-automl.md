---
title: 06 Training and AutoML
layout: default
nav_order: 8
---

# Training and AutoML

This module explains how models are trained in Azure ML, what AutoML does in the backend,
and how to move from baseline experiments to reliable model selection.

## Learning goals

1. Understand manual training vs AutoML.
2. Configure an AutoML run with useful constraints.
3. Interpret run outputs and choose a production candidate.

![AutoML diagram](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_diagram.png)

![AutoML process expectations](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_process_what_to_expect.png)

![Detailed ML-based time-series forecast steps](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/Detailed-steps-of-ML-based-time-series-forecast.png)

## AutoML workflow

1. Choose task type
2. Provide training data
3. Select compute target
4. Set metric and constraints
5. Submit run and compare candidates

## What AutoML does behind the scenes

- Tries multiple algorithms and hyperparameters.
- Runs cross-validation/validation scoring.
- Applies feature transformations when configured.
- Logs metrics, artifacts, and lineage.
- Returns best run/model based on chosen primary metric.

### AutoML algorithm candidates (tabular classification)

AutoML typically evaluates some or all of the following:

| Candidate model | Notes |
|---|---|
| LightGBM | Often best on tabular; fast and memory-efficient |
| XGBoost | Strong competition; more hyperparams |
| LogisticRegression | Fast baseline; reveals if linear structure is sufficient |
| RandomForest | Good stability, less tuning |
| ExtraTrees | Faster training variant of random forest |
| Voting Ensemble | AutoML-specific ensemble of top runs |
| Stack Ensemble | AutoML-specific meta-model over top runs |

The `VotingEnsemble` or `StackEnsemble` at the end is AutoML's way of squeezing extra performance beyond single models — they are often the final winner.

## Compute and performance

Performance relation:

$$
\text{Performance}=\frac{1}{\text{Execution Time}}
$$

Execution time is affected by:

- Data volume and feature dimensionality
- Algorithm complexity
- Compute size (CPU/GPU, memory)
- Parallelization and max concurrent iterations

## Minimal AutoML configuration checklist

| Setting | Why it matters |
|---|---|
| task | Defines candidate model family |
| primary metric | Aligns optimization with business objective |
| iterations/timeout | Controls search budget |
| cross-validation | Improves robustness of ranking |
| featurization settings | Impacts model quality and reproducibility |

### Minimal AutoML code example (Azure SDK v2)

```python
from azure.ai.ml import MLClient, automl
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="<rg>",
    workspace_name="<ws>"
)

classification_job = automl.classification(
    compute="cpu-cluster",
    experiment_name="fraud-automl",
    training_data=ml_client.data.get("fraud-train", version="1"),
    target_column_name="is_fraud",
    primary_metric="AUC_weighted",
    n_cross_validations=5,
    enable_model_explainability=True,
    timeout_minutes=60,
    max_concurrent_trials=4,
)

returned_job = ml_client.jobs.create_or_update(classification_job)
```

Key flags:
- `AUC_weighted` is safer than `accuracy` for fraud (imbalanced classes).
- `enable_model_explainability=True` generates SHAP-based feature importance.
- `max_concurrent_trials` should match compute cluster core count.

## Common mistakes

- Choosing accuracy for imbalanced classification.
- Running too few iterations and over-trusting the first winner.
- Ignoring latency/cost while selecting best score.

## Search-space design (important)

AutoML quality depends on search space, not only iteration count.

| Parameter | Too narrow | Too wide | Practical approach |
|---|---|---|---|
| Model families | Misses better model type | Wastes budget | Start broad, prune after baseline |
| Learning rate | Can miss convergence sweet spot | Slow exploration | Use log-scale ranges |
| Tree depth/leaves | Underfit risk | Overfit + latency risk | Constrain by latency budget |
| Regularization | Under-regularized noise fit | Over-regularized underfit | Tune with CV and holdout checks |

## Validation strategy choices

| Context | Validation approach |
|---|---|
| Standard tabular | K-fold cross-validation |
| Temporal forecasting | Rolling-origin validation |
| Grouped entities | GroupKFold-like entity splits |

## Experiment tracking fields to persist

Minimum metadata for reproducibility:

- Run ID, parent run ID
- Code snapshot/version
- Dataset asset version
- Environment version
- Feature set/hash
- Hyperparameters
- Metrics by split
- Output model URI/version

## Candidate selection policy

Select deployment candidate using multi-objective criteria:

$$
	ext{Score}_{deploy}=w_1\cdot\text{Quality}-w_2\cdot\text{Latency}-w_3\cdot\text{Cost}+w_4\cdot\text{Stability}
$$

where weights $w_i$ reflect business priorities.

## Promotion gates (dev to prod)

1. Offline metric threshold met.
2. Inference latency under SLO on representative hardware.
3. Security scan and dependency policy passed.
4. Explainability/fairness review completed.
5. Approval workflow sign-off recorded.

## Quick self-check

1. Why is primary metric choice critical in AutoML?
2. What trade-off does max concurrent iterations control?
3. Why should deployment constraints be considered during model selection?
