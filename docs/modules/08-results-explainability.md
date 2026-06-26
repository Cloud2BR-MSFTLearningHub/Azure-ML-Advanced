---
title: 08 Results and Explainability
layout: default
nav_order: 10
---

# Results and Explainability

Model review artifacts:

- Confusion matrix
- ROC and precision-recall curves
- Feature importance
- SHAP and LIME explanations

## Monitoring and Drift

Covariate drift (input distribution shift):

$$
P_t(X)\neq P_{t+\Delta}(X)
$$

Concept drift (mapping shift):

$$
P_t(Y\mid X)\neq P_{t+\Delta}(Y\mid X)
$$

Population Stability Index (PSI):

$$
\mathrm{PSI}=\sum_{b=1}^{B}(a_b-e_b)\ln\frac{a_b}{e_b}
$$

where $a_b$ and $e_b$ are actual and expected bin proportions.

![ROC good vs bad](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/roc_good_bad.png)

![Precision-recall good vs bad](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/precisionrecall_good_bad.png)

Use both global and local explanations before production release.
