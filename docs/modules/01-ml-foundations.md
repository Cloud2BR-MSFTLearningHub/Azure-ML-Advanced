
# ML Foundations

This module builds the mathematical and conceptual base needed for all later modules.
It starts from first principles and then moves into model families and selection logic.

## Core learning families

- Supervised learning: classification and regression
- Unsupervised learning: clustering and association
- Reinforcement learning: policy learning from rewards

Additional modern families used in production:

- Semi-supervised learning: combine a small labeled set with a large unlabeled set.
- Self-supervised learning: create supervisory signals from the data itself.
- Online learning: update models continuously from streaming/new data.

What to remember:

- Supervised learns from known answers.
- Unsupervised discovers structure without labels.
- Reinforcement learns through interaction and reward.

![Machine learning types](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/machine_learning_types.png)

> Image explanation: This visual shows machine learning types. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

![Types of ML based on objective](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/types_of_ml_based_in_objective.png)

> Image explanation: This visual shows types of ml based on objective. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

## Problem Types

- Supervised: classification and regression
- Unsupervised: clustering and association
- Reinforcement: control and policy optimization

## Data and notation basics

- Dataset: $D = \{(x_i, y_i)\}_{i=1}^{N}$ for supervised learning.
- Feature vector: $x_i \in \mathbb{R}^{d}$.
- Target/label: $y_i$.
- Model: $f_{\theta}(x)$ with parameters $\theta$.

## Supervised learning categories

| Category | Output type | Examples | Typical metrics |
|---|---|---|---|
| Binary classification | 0/1 class | Fraud yes/no, churn yes/no | Precision, Recall, F1, AUC |
| Multi-class classification | One of $K$ classes | Product category, diagnosis class | Macro-F1, accuracy, log loss |
| Multi-label classification | Multiple classes per sample | Tagging documents/topics | Micro-F1, Hamming loss |
| Regression | Continuous value | Price, demand, latency | MAE, RMSE, $R^2$ |
| Time-series forecasting | Future values over time | Sales, energy, traffic | MAPE, RMSE, sMAPE |

### Classification vs regression intuition

- Classification predicts which class.
- Regression predicts how much.

The same feature set can support both depending on business objective.

## Unsupervised learning categories

| Category | Purpose | Typical methods |
|---|---|---|
| Clustering | Group similar observations | K-Means, DBSCAN, hierarchical clustering |
| Dimensionality reduction | Compress features while retaining structure | PCA, UMAP, autoencoders |
| Association mining | Find co-occurrence rules | Apriori, FP-growth |
| Anomaly detection | Detect rare/abnormal patterns | Isolation Forest, One-Class SVM |

## Semi-supervised and self-supervised

- Semi-supervised is useful when labels are expensive. Example: you have 1,000 labelled medical images and 50,000 unlabelled ones. A semi-supervised approach trains on both, propagating labels from confident predictions.
- Self-supervised is common in foundation models (GPT, BERT, CLIP) and pretraining pipelines. The model is trained on a proxy task whose labels come from the data itself â€” e.g., predict the next word, reconstruct a masked patch.
- Both reduce dependence on manual labeling, which is expensive and slow at scale.

| Approach | Label requirement | Common algorithms |
|---|---|---|
| Supervised | All samples labelled | Logistic regression, XGBoost, NN |
| Semi-supervised | Small fraction labelled | Label propagation, pseudo-labelling |
| Self-supervised | No labels needed | Masked autoencoders, contrastive learning |

## Reinforcement learning components

RL is usually modeled as a Markov Decision Process (MDP):

$$
(\mathcal{S},\mathcal{A},P,R,\gamma)
$$

where:

- $\mathcal{S}$: set of states
- $\mathcal{A}$: set of actions
- $P$: transition dynamics
- $R$: reward function
- $\gamma$: discount factor

Value function concept:

$$
V^{\pi}(s)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t\mid s_0=s\right]
$$

Goal:

$$
\max_{\pi}\;\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
$$

Supervised objective:

$$
\min_{\theta} \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i), y_i)
$$

This is empirical risk minimization: find parameters minimizing average training loss.

Common loss functions:

$$
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

$$
\mathcal{L}_{BCE}=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{p}_i)+(1-y_i)\log(1-\hat{p}_i)\right]
$$

Multiclass cross-entropy:

$$
\mathcal{L}_{CE}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(\hat{p}_{ik})
$$

Optimization objective with regularization:

$$
\min_{\theta}\;\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i),y_i)+\lambda R(\theta)
$$

Gradient descent update:

$$
\theta_{t+1}=\theta_t-\eta\nabla_{\theta}\mathcal{L}
$$

Regularization:

$$
\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i),y_i)+\lambda R(\theta)
$$

Common choices:

- L1 regularization: $R(\theta)=\lVert\theta\rVert_1$ (sparsity, feature selection)
- L2 regularization: $R(\theta)=\lVert\theta\rVert_2^2$ (weight shrinkage, stability)

## Overfitting and generalization

- Training error can decrease while test error increases (overfitting).
- Use train/validation/test separation and cross-validation.
- Prefer simpler models when performance is comparable.

Practical split sizes (rule of thumb):

| Split | Typical proportion | Purpose |
|---|---|---|
| Train | 60â€“80% | Fit model parameters |
| Validation | 10â€“20% | Tune hyperparameters and compare models |
| Test | 10â€“20% | Final unbiased evaluation before deployment |

The test set must **never** be used during model selection. Using it for selection is a form of data leakage that makes offline scores over-optimistic.

Cross-validation: when data is limited, k-fold CV uses all data for both training and validation by rotating folds. K=5 or K=10 is typical.

## Bias-variance intuition

- High bias: model too simple, underfits. Symptom: low training accuracy and low test accuracy.
- High variance: model too complex, overfits. Symptom: high training accuracy, much lower test accuracy.

The expected test error decomposes as:

$$
\mathbb{E}[(y-\hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2
$$

where $\sigma^2$ is irreducible noise.

The practical goal is to balance both:

| Technique | Addresses |
|---|---|
| More training data | Reduces variance |
| Regularization (L1/L2) | Reduces variance |
| Feature selection/engineering | Can reduce bias and variance |
| More complex model | Reduces bias (risk: more variance) |
| Ensemble methods | Reduces both (usually) |

## How to choose an ML type quickly

| If your question is... | Use... |
|---|---|
| Can I predict this known target? | Supervised learning |
| Can I group similar records without labels? | Unsupervised learning |
| Can an agent learn through interaction and reward? | Reinforcement learning |
| I have few labels but lots of unlabeled data | Semi-supervised learning |

## Typical mistakes to avoid

- Using accuracy alone on highly imbalanced datasets.
- Mixing train/test data during preprocessing (data leakage).
- Ignoring concept drift after deployment.
- Treating model score as the only KPI without business impact validation.

## Quick self-check

1. If your target is numeric, which family do you use?
2. Why can high training accuracy still fail in production?
3. What does the learning rate $\eta$ control in gradient descent?

