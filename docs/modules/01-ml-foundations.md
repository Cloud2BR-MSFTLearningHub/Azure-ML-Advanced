---
title: 01 ML Foundations
layout: default
nav_order: 3
---

# ML Foundations

Core learning families:

- Supervised learning: classification and regression
- Unsupervised learning: clustering and association
- Reinforcement learning: policy learning from rewards

![Machine learning types](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/machine_learning_types.png)

![Types of ML based on objective](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/types_of_ml_based_in_objective.png)

## Problem Types

- Supervised: classification and regression
- Unsupervised: clustering and association
- Reinforcement: control and policy optimization

Supervised objective:

$$
\min_{\theta} \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i), y_i)
$$

Common loss functions:

$$
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

$$
\mathcal{L}_{BCE}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{p}_i)+(1-y_i)\log(1-\hat{p}_i)\right]
$$

Gradient descent update:

$$
	heta_{t+1}=\theta_t-\eta\nabla_{\theta}\mathcal{L}
$$

Regularization:

$$
\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i),y_i)+\lambda R(\theta)
$$
