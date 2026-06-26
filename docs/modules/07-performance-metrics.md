---
title: 07 Performance Metrics
layout: default
nav_order: 9
---

# Performance Metrics

![How to choose a metric](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png)

Confusion matrix rates:

$$
	ext{TPR}=\frac{TP}{TP+FN},\quad
	ext{FPR}=\frac{FP}{FP+TN},\quad
	ext{TNR}=\frac{TN}{TN+FP}
$$

Classification metrics:

- Precision: $\frac{TP}{TP+FP}$
- Recall: $\frac{TP}{TP+FN}$
- F1: $2\cdot\frac{PR}{P+R}$
- AUC

Additional classification formulas:

$$
	ext{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\mathrm{MCC}=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

$$
\mathrm{AUC}=\int_0^1 \mathrm{TPR}(\mathrm{FPR})\,d\mathrm{FPR}
$$

Regression metrics:

- MAE
- RMSE
- R2

Regression formulas:

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|,
\quad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
$$

$$
R^2=1-\frac{\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{N}(y_i-\bar{y})^2}
$$
