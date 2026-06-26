
# Model Types

![Model implementation schema](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/logic_schema_model_implementation.png)

> Image explanation: This visual shows model implementation schema. Use it to understand the concept in this section and connect it to practical Azure ML decisions.

This module connects algorithm families to problem types and deployment constraints.

## Common algorithms by task

- Classification: logistic regression, random forest, gradient boosting, SVM
- Regression: linear regression, random forest regressor, XGBoost
- Forecasting: AutoARIMA, Prophet, gradient boosting variants

## Model family quick guide

| Family | Strength | Weakness | Typical use |
|---|---|---|---|
| Linear models | Fast, interpretable | Limited nonlinear capacity | Baselines, tabular regression |
| Tree ensembles | Strong tabular performance | Larger memory/latency | Structured business data |
| Kernel methods | Good margin-based behavior | Poor scaling on very large data | Medium-size classification |
| Neural networks | High representational power | Data and tuning intensive | Vision, NLP, complex patterns |

Regularized objective:

$$
\min_{\theta} \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i), y_i) + \lambda R(\theta)
$$

## Representative Mathematical Forms

Logistic regression probability:

$$
\hat{p}=\sigma(\theta^T x)=\frac{1}{1+e^{-\theta^T x}}
$$

Decision boundary interpretation:

- If $\hat{p} > \tau$, predict positive class.
- Threshold $\tau$ should be tuned by business cost trade-off.

Naive Bayes decision rule:

$$
P(y\mid x_1,\dots,x_n)\propto P(y)\prod_{i=1}^{n}P(x_i\mid y)
$$

Assumption note: Naive Bayes assumes feature conditional independence.

Elastic Net objective:

$$
\min_{\theta}\frac{1}{2N}\|y-X\theta\|_2^2+\lambda\left(\alpha\|\theta\|_1+\frac{1-\alpha}{2}\|\theta\|_2^2\right)
$$

LightGBM and gradient boosting models build additive trees:

$$
F_m(x)=F_{m-1}(x)+\nu\,h_m(x)
$$

where $h_m(x)$ is the fitted weak learner at stage $m$ and $\nu$ is the learning rate.

## Practical model selection

| Constraint | Preference |
|---|---|
| Need explainability | Linear models, shallow trees |
| Best tabular accuracy | Gradient boosting (LightGBM/XGBoost/CatBoost) |
| Very low latency | Linear or optimized tree model |
| Limited training data | Simpler regularized models |
| High-dimensional sparse features | Sparse linear models (SGDClassifier, Elastic Net) |
| Mixed numeric + categorical | Tree ensembles or CatBoost (native cat handling) |

## Decision tree intuition

Decision trees split data by maximizing a purity measure at each node:

$$
\text{Gini impurity} = 1 - \sum_{k=1}^{K} p_k^2
$$

$$
\text{Information gain} = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)
$$

where $H(S) = -\sum_k p_k \log_2 p_k$ is the entropy of set $S$.

Deep trees overfit. Random forests average many trees trained on bootstrap samples and random feature subsets. This reduces variance without much increase in bias.

## Gradient boosting mechanics

Gradient boosting builds trees iteratively to correct residual errors:

| Iteration | What is learned |
|---|---|
| 0 | Base prediction (mean or class frequency) |
| 1 | Tree fitted to gradient of loss (first-order residuals) |
| 2 | Tree fitted to remaining residuals |
| ... | Each step shrinks residuals towards zero |

Key hyperparameters that matter most:

| Parameter | Effect |
|---|---|
| `n_estimators` | More trees = more capacity (risk: overfit without early stopping) |
| `learning_rate` (shrinkage $\nu$) | Smaller = more conservative, usually better with more trees |
| `max_depth` / `num_leaves` | Controls tree complexity (main overfit knob) |
| `min_child_samples` | Regularises leaf size |
| `subsample` / `colsample_bytree` | Stochastic column/row sampling, reduces variance |

## Bias, variance, and complexity

- Increasing model complexity usually reduces bias but increases variance.
- Regularization, pruning, and early stopping are practical controls.

## Advanced considerations

- Calibration: predicted probabilities should reflect real event frequency.
- Fairness: evaluate group-wise performance, not only global score.
- Robustness: test under noise, missingness, and shifted distributions.

## Ensemble methods in practice

Three main ensemble patterns beyond gradient boosting:

| Method | Idea | Benefit |
|---|---|---|
| Bagging | Train models on bootstrap samples, average predictions | Reduces variance |
| Boosting | Train models sequentially, each correcting the last | Reduces bias iteratively |
| Stacking | Train a meta-model on out-of-fold predictions of base models | Often best final accuracy |

Stacking example (2-layer):

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

estimators = [
    ("rf", RandomForestClassifier(n_estimators=100)),
    ("lgbm", LGBMClassifier(n_estimators=200)),
]
stacker = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacker.fit(X_train, y_train)
```

## Algorithm complexity and latency trade-off

| Model type | Inference latency | Memory footprint | Notes |
|---|---|---|---|
| Logistic regression | Very low (us) | Very small | Single matrix multiply |
| Shallow decision tree | Low (us) | Small | Tree traversal |
| Random forest (100 trees) | Medium (ms) | Medium | N tree traversals |
| LightGBM (1000 trees) | Low-medium | Medium | Leaf-wise, well optimised |
| Deep neural network | High (ms-s on CPU) | Large | Batch inference preferred |

## Quick self-check

1. Why might a linear model be preferred even if score is slightly lower?
2. What does learning rate $\nu$ control in boosting?
3. Which family is often strongest on structured tabular data?

