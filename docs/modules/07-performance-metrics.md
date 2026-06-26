
# Performance Metrics

Choosing the right metric is one of the most important decisions in ML. A model can look
excellent on one metric and fail on the real business objective.

![How to choose a metric](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)

> Image explanation: This visual shows how to choose a metric. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

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

