
# Training and AutoML

This module explains how models are trained in Azure ML, what AutoML does in the backend,
and how to move from baseline experiments to reliable model selection.

## Learning goals

1. Understand manual training vs AutoML.
2. Configure an AutoML run with useful constraints.
3. Interpret run outputs and choose a production candidate.

![AutoML diagram](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_diagram.png)

> Image explanation: This visual shows automl diagram. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![AutoML process expectations](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/automl_process_what_to_expect.png)

> Image explanation: This visual shows automl process expectations. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Detailed ML-based time-series forecast steps](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/Detailed-steps-of-ML-based-time-series-forecast.png)

> Image explanation: This visual shows detailed ml-based time-series forecast steps. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

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

The `VotingEnsemble` or `StackEnsemble` at the end is AutoML's way of squeezing extra performance beyond single models â€” they are often the final winner.

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

## Deep dive: every concept, explained

This section explains what AutoML automates, what it does *not*, and why each control exists.

### What AutoML actually searches

AutoML is structured search over three coupled choices: **featurization** (how raw columns
become model inputs), **algorithm** (which model family), and **hyperparameters** (the settings
within that family). Conceptually it is solving an outer optimization:

$$
\min_{a \in \text{algorithms},\; h \in \text{hyperparams}(a)}\; \text{ValidationLoss}(a, h)
$$

It does not invent new algorithms — it intelligently *allocates a fixed budget* of trials across
known ones, using results so far to decide what to try next. This is why "search-space design"
matters more than raw iteration count: a good space contains the winning region; a bad one never
does.

### Featurization, demystified

When enabled, AutoML automatically handles missing-value imputation, categorical encoding,
text vectorization, and feature scaling — the same steps from the data-preparation module, applied
consistently inside cross-validation folds so they do not leak. The benefit is leakage-safe,
reproducible preprocessing; the cost is less manual control, which is why `featurization` settings
are explicit and logged for reproducibility.

### Cross-validation inside AutoML and why it ranks models fairly

`n_cross_validations=5` means every candidate is scored on 5 rotating validation folds and the
results averaged. This reduces the chance that one lucky split crowns the wrong model. For
**temporal** data, plain k-fold leaks the future, so **rolling-origin** validation is used
instead; for **grouped** entities (e.g. multiple rows per customer), group-aware splits prevent
the same entity appearing in both train and validation.

### Primary metric: aligning the optimizer with the business

AutoML optimizes exactly one **primary metric**, so choosing it *is* choosing what "best" means.
On imbalanced problems, `accuracy` is misleading (a model predicting "never fraud" scores 99%),
so `AUC_weighted` or `average_precision` are used instead. The lesson generalizes: the optimizer
will ruthlessly exploit whatever metric you give it, so the metric must encode the real cost
structure.

### Ensembles: why the winner is often a `VotingEnsemble`

After trying individual models, AutoML builds two meta-models:

- **Voting ensemble** — averages the predictions of the top runs. Diverse models make
  *uncorrelated* errors, so the average is more accurate and stable than any single member.
- **Stack ensemble** — trains a small meta-model on the base models' out-of-fold predictions to
  learn *how* to combine them.

These usually win because combining diverse learners reduces variance — the same bagging/stacking
principle from the model-types module, applied automatically.

### Concurrency, budget, and the cost/time trade-off

`max_concurrent_trials` controls how many candidates train in parallel; setting it to the
cluster's node count keeps compute busy and shortens wall-clock time, but does **not** reduce
total compute cost (you pay for the same number of trials, just faster). `timeout_minutes` and
iteration caps bound the search **budget** — the central knob trading off thoroughness against
time and money.

### The multi-objective selection score, explained

The candidate score $\text{Score}_{deploy}=w_1\text{Quality}-w_2\text{Latency}-w_3\text{Cost}+w_4\text{Stability}$
formalizes a real-world truth: the deployable model maximizes quality *and* stability while being
penalized for latency and cost. The weights $w_i$ encode business priorities — a real-time API
weights latency heavily; a nightly batch job weights it near zero. AutoML ranks by the primary
metric, but the *human* promotion decision should use this fuller objective, which is exactly why
the **promotion gates** check latency-under-SLO, security, and fairness, not just offline score.

### Why experiment tracking metadata is non-negotiable

The list of fields to persist (run ID, data version, environment version, feature hash,
hyperparameters, per-split metrics, model URI) is what makes a result **reproducible** and
**auditable**. If you cannot answer "which data, code, and environment produced this model?", you
cannot debug a regression, pass an audit, or safely retrain — so this metadata is the backbone of
MLOps, not optional bookkeeping.

