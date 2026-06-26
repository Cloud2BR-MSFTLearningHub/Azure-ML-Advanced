
# Performance Metrics

Choosing the right metric is one of the most important decisions in ML. A model can look
excellent on one metric and fail on the real business objective.

![How to choose a metric](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)

> **Tip - How to use this chart:** Pick the metric from the *cost of errors*, not habit. On imbalanced problems prefer F1, PR-AUC,
> or MCC over accuracy; weight recall when missed positives are costly, precision when false
> alarms are costly.

## Confusion matrix basics

Counts:

- $TP$: true positives
- $FP$: false positives
- $TN$: true negatives
- $FN$: false negatives

Rates:

$$
\text{TPR}=\frac{TP}{TP+FN},\quad
\text{FPR}=\frac{FP}{FP+TN},\quad
\text{TNR}=\frac{TN}{TN+FP}
$$

### Reading the confusion matrix

```
Actual \\ Predicted |  Positive  |  Negative
---------------------|------------|----------
       Positive      |   TP       |   FN
       Negative      |   FP       |   TN
```

| Cell | Meaning | Example (fraud model) |
|---|---|---|
| TP | Correctly flagged positive | Real fraud caught |
| FP | False alarm | Legitimate tx blocked |
| TN | Correctly cleared negative | Legit tx passes |
| FN | Missed positive | Fraud slips through |

For a fraud use case, **FN is the more dangerous error** (missed fraud). This means recall should be weighted heavily in metric choice.

## Classification metrics

- Precision: $\frac{TP}{TP+FP}$
- Recall: $\frac{TP}{TP+FN}$
- F1: $2\cdot\frac{PR}{P+R}$
- AUC

Additional formulas:

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

$$
\mathrm{AUC}=\int_0^1 \mathrm{TPR}(\mathrm{FPR})\,d\mathrm{FPR}
$$

When to use what:

| Scenario | Better metric choices |
|---|---|
| Class imbalance | F1, PR-AUC, MCC, balanced accuracy |
| High false-negative cost | Recall, F2 |
| High false-positive cost | Precision |
| Probability quality | Log loss, calibration metrics |

## Regression metrics

- MAE
- RMSE
- R2

Formulas:

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|,
\quad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
$$

$$
R^2=1-\frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{N}(y_i-\bar{y})^2}
$$

Interpretation tips:

- MAE is robust and easy to explain in original units.
- RMSE penalizes large errors more strongly.
- $R^2$ compares against a mean-prediction baseline.

## Forecasting metrics (practical)

- MAPE: intuitive percentage error, unstable near zero values.
- sMAPE: symmetric variant for better comparability.
- RMSE/MAE: still useful for forecast quality.

Formulas:

$$
\text{MAPE}=\frac{100}{N}\sum_{i=1}^{N}\left|\frac{y_i-\hat{y}_i}{y_i}\right|
$$

$$
\text{sMAPE}=\frac{100}{N}\sum_{i=1}^{N}\frac{2|y_i-\hat{y}_i|}{|y_i|+|\hat{y}_i|}
$$

Guidance: prefer RMSE/MAE for comparing models on the same scale. Use MAPE/sMAPE only when communicating errors as percentages to business stakeholders.

## Pitfalls to avoid

- Reporting one metric without confidence intervals.
- Comparing models on different validation splits.
- Ignoring threshold tuning in classification.
- Using accuracy as the primary metric for imbalanced data.
- Evaluating only globally when segment-level performance may diverge significantly.

## Threshold optimization (classification)

Production classification decisions require threshold policy, not default $0.5$.

Expected business cost at threshold $\tau$:

$$
\mathbb{E}[\text{Cost}(\tau)] = C_{FP}\cdot FP(\tau)+C_{FN}\cdot FN(\tau)
$$

Choose $\tau$ that minimizes expected cost under business constraints.

## Calibration and reliability

A classifier can rank well (high AUC) but produce poorly calibrated probabilities.

Calibration checks:

- Reliability curve
- Brier score
- Expected calibration error (ECE)

Use calibration methods (Platt scaling, isotonic regression) when decision systems rely
on probability values, not just rank ordering.

## SLI/SLO examples for model quality

| SLI | SLO example |
|---|---|
| Weekly macro-F1 | >= 0.82 |
| Weekly RMSE | <= 5.0 |
| Calibration error | <= 0.03 |
| Segment disparity ratio | <= 1.25 |

## Quick self-check

1. Which metric is safer than accuracy for imbalanced data?
2. Why can RMSE be much larger than MAE?
3. What does a negative $R^2$ imply?

## Deep dive: every concept, explained

This section explains *why* each metric is built the way it is and when it misleads.

### The confusion matrix is the source of every classification metric

All classification metrics are ratios of the four cells $TP, FP, TN, FN$. Memorizing the cells
is enough to reconstruct any metric:

- **Precision** $\tfrac{TP}{TP+FP}$ answers "when the model says positive, how often is it right?"
  : it is the metric you care about when **false positives are expensive** (blocking good
  customers, flagging healthy patients).
- **Recall / TPR** $\tfrac{TP}{TP+FN}$ answers "of all real positives, how many did we catch?" :
  the metric when **false negatives are expensive** (missed fraud, missed disease).
- **F1** $2\tfrac{PR}{P+R}$ is the **harmonic mean** of the two. The harmonic mean (not the
  ordinary average) is used because it stays low unless *both* precision and recall are high : it
  refuses to reward a model that sacrifices one for the other.

### Why accuracy fails on imbalanced data

Accuracy $\tfrac{TP+TN}{\text{all}}$ weights every example equally, so when 99% of cases are
negative, predicting "always negative" scores 99% while catching zero positives. This is why the
course repeatedly steers toward **F1, PR-AUC, MCC, or balanced accuracy** for skewed problems :
they all, in different ways, stop the majority class from dominating the score.

### ROC-AUC vs PR-AUC, and what "threshold-free" means

- **AUC** is the area under the ROC curve (TPR vs FPR as the threshold sweeps from 1 to 0). It
  equals the probability that the model ranks a random positive above a random negative : a pure
  measure of **ranking quality**, independent of any chosen threshold.
- On heavy imbalance, ROC-AUC can look deceptively high because the huge negative count keeps FPR
  low. **PR-AUC** (precision vs recall) focuses on the positive class and is the more honest
  summary when positives are rare.
- **MCC** (Matthews correlation coefficient) uses all four cells in a single balanced number from
  −1 to +1, which is why it is robust even under severe imbalance.

### Threshold optimization as cost minimization

A model outputs probabilities; the **threshold** $\tau$ converts them to decisions. Because false
positives and false negatives usually have *different* costs, the optimal threshold minimizes
expected cost $\mathbb{E}[\text{Cost}(\tau)] = C_{FP}\cdot FP(\tau) + C_{FN}\cdot FN(\tau)$ rather
than maximizing accuracy. Concretely: if a missed fraud costs 20 times as much as a false alarm, you lower $\tau$
to trade many false positives for fewer false negatives. The default 0.5 is almost never optimal
in production.

### Regression metrics: MAE vs RMSE vs $R^2$

- **MAE** averages absolute errors : it is in the target's units and treats all errors linearly,
  so it is **robust to outliers** and easy to explain ("off by \$5 on average").
- **RMSE** averages *squared* errors then square-roots, so large errors are penalized
  disproportionately. RMSE ≥ MAE always; a *large gap* between them signals a few big misses
  (heavy-tailed errors) worth investigating.
- $R^2 = 1 - \tfrac{SS_{res}}{SS_{tot}}$ compares the model against the trivial "predict the mean"
  baseline. $R^2=1$ is perfect, $0$ means no better than the mean, and **negative $R^2$ means the
  model is *worse* than predicting the mean** : a clear signal something is broken.

### Forecasting metrics and the zero-denominator trap

**MAPE** expresses error as a percentage, which stakeholders find intuitive, but it **divides by
the actual value**, so it explodes or is undefined near zero and over-penalizes under-forecasts.
**sMAPE** symmetrizes the denominator to bound the value and treat over/under-forecasts more
evenly. The practical rule from the module: optimize and compare models on RMSE/MAE (stable),
and translate to MAPE/sMAPE only for *communication*.

### Calibration: ranking well is not the same as being right

A model with high AUC ranks examples correctly but may still output probabilities that do not
match reality (e.g. its "90%" predictions are right only 70% of the time). When downstream
decisions use the probability *value* (expected-loss calculations, pricing, triage), you need
**calibration**:

- **Reliability curve** plots predicted vs observed frequency; the diagonal is perfect.
- **Brier score** is the mean squared error of the probabilities themselves.
- **ECE** (expected calibration error) summarizes the average gap between confidence and accuracy.
- **Platt scaling** (fit a logistic on the scores) and **isotonic regression** (fit a monotonic
  step function) are the standard post-hoc fixes.

### From metrics to SLIs/SLOs : closing the loop to operations

An offline metric becomes a production **SLI** (service level indicator) when it is measured
continuously, and an **SLO** (objective) when a threshold is attached (e.g. "weekly macro-F1 ≥
0.82"). This is how model quality joins latency and availability as a monitored, alertable
property : the bridge from this module to drift monitoring and deployment SLOs later in the
course.

## Quick self-check (deep dive)

1. Why is F1 the harmonic mean of precision and recall rather than the ordinary average?
2. On a 99%-negative dataset, why can ROC-AUC look great while PR-AUC is poor?
3. What does a negative $R^2$ tell you about the model?
4. Why is the default 0.5 threshold almost never optimal in production?
5. A model has high AUC but its "90%" predictions are right only 70% of the time: what is the problem and which fixes apply?

