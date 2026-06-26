---
title: 05 Model Types
layout: default
nav_order: 7
---

# Model Types

![Model implementation schema](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/logic_schema_model_implementation.png)

Common algorithms by task:

- Classification: logistic regression, random forest, gradient boosting, SVM
- Regression: linear regression, random forest regressor, XGBoost
- Forecasting: AutoARIMA, Prophet, gradient boosting variants

Regularized objective:

$$
\min_{\theta} \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i), y_i) + \lambda R(\theta)
$$

## Representative Mathematical Forms

Logistic regression probability:

$$
\hat{p}=\sigma(\theta^T x)=\frac{1}{1+e^{-\theta^T x}}
$$

Naive Bayes decision rule:

$$
P(y\mid x_1,\dots,x_n)\propto P(y)\prod_{i=1}^{n}P(x_i\mid y)
$$

Elastic Net objective:

$$
\min_{\theta}\frac{1}{2N}\|y-X\theta\|_2^2+\lambda\left(\alpha\|\theta\|_1+\frac{1-\alpha}{2}\|\theta\|_2^2\right)
$$

LightGBM and gradient boosting models build additive trees:

$$
F_m(x)=F_{m-1}(x)+\nu\,h_m(x)
$$

where $h_m(x)$ is the fitted weak learner at stage $m$ and $\nu$ is the learning rate.
