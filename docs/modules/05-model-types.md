
# Model Types

![Model implementation schema](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/logic_schema_model_implementation.png)

!!! note "What this shows"
    A logical schema for implementing a model — from problem type to algorithm family to deployment
    constraints. Use it to trace how a business question narrows down to a specific model choice.

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

## Deep dive: every concept, explained

This section connects the equations above to the intuition and the engineering trade-offs.

### Linear and logistic models — the interpretable baseline

A **linear model** predicts $\hat y = \theta^T x$: each feature contributes a weighted vote, and
the weight $\theta_j$ is directly readable as "effect of feature $j$". **Logistic regression**
wraps this in the **sigmoid** $\sigma(z) = \tfrac{1}{1+e^{-z}}$, which squashes any real number
into $(0,1)$ so the output is a valid probability. The model is linear in *log-odds*: a unit
change in $x_j$ multiplies the odds by $e^{\theta_j}$. This transparency is why linear models
remain the default baseline and the choice when regulators require explainable decisions.

### The decision threshold $\tau$ is a business lever, not a constant

A classifier outputs a probability; turning it into a yes/no needs a **threshold** $\tau$
(default 0.5). Moving $\tau$ trades precision against recall: a fraud team that fears missed
fraud lowers $\tau$ (catch more, accept more false alarms); a team that fears blocking good
customers raises it. The right $\tau$ is set by the *relative cost* of the two error types, not
by the algorithm — which is why thresholds are tuned after training, against business cost.

### Naive Bayes and the independence assumption

$P(y\mid x) \propto P(y)\prod_i P(x_i\mid y)$ comes straight from Bayes' rule, with one
simplifying ("naive") assumption: features are **conditionally independent given the class**.
This is almost never literally true, yet the model works surprisingly well for text/spam because
it needs very little data and training is just counting frequencies. Knowing the assumption tells
you its failure mode: strongly correlated features get their evidence double-counted.

### Trees, impurity, and why ensembles beat single trees

A **decision tree** repeatedly splits the data to make each resulting group more "pure":

- **Gini impurity** $1-\sum_k p_k^2$ and **entropy** $-\sum_k p_k\log_2 p_k$ both measure how
  mixed a node's labels are; a split is chosen to reduce this the most (**information gain**).
- A single deep tree memorizes the training data → **high variance / overfitting**.
- **Random forests** fix this by **bagging**: train many trees on bootstrap samples with random
  feature subsets and average them. Averaging decorrelated trees cancels their individual errors,
  cutting variance with little added bias.
- **Gradient boosting** takes the opposite route: build trees **sequentially**, each one fitted
  to the *residual errors* (the gradient of the loss) of the current ensemble. The update
  $F_m(x) = F_{m-1}(x) + \nu\,h_m(x)$ adds each new weak learner scaled by the **shrinkage**
  $\nu$. Small $\nu$ with many trees is the well-known recipe for top tabular accuracy.

### The boosting hyperparameters, and what they really control

- `n_estimators` is *capacity*: more trees fit finer structure but overfit without early
  stopping on a validation set.
- `learning_rate` ($\nu$) is *caution per step*: lower means each tree corrects less, so the
  ensemble generalizes better — but needs proportionally more trees.
- `max_depth` / `num_leaves` is the *main overfit knob*: it caps how complex any single tree can
  get.
- `subsample` / `colsample_bytree` inject **stochasticity** (row/column sampling) that
  decorrelates trees and reduces variance, much like a random forest does.

### Bagging vs boosting vs stacking — one sentence each

- **Bagging** reduces **variance** by averaging independent models (random forest).
- **Boosting** reduces **bias** by sequentially correcting mistakes (XGBoost/LightGBM).
- **Stacking** trains a **meta-model** on the out-of-fold predictions of diverse base models to
  exploit their complementary strengths — usually the highest accuracy, at the cost of complexity
  and latency.

### Calibration, fairness, robustness — the production-grade concerns

- **Calibration**: a model is calibrated if, among predictions of "70% probability", about 70%
  are actually positive. Boosted trees are often *mis-calibrated* and benefit from Platt scaling
  or isotonic regression before probabilities are used in decisions.
- **Fairness**: aggregate accuracy can hide that a model performs worse for a subgroup. Always
  evaluate metrics *per segment*, not just globally.
- **Robustness**: production data is noisier than training data; test the model under injected
  noise, missing fields, and shifted distributions before trusting it.

### Why latency and memory belong in model selection

The latency/footprint table above is a reminder that the "best" model is the one that meets
*all* constraints. A 1000-tree ensemble that adds 30 ms per call may break a real-time SLA, while
a single matrix-multiply logistic regression serves in microseconds. Accuracy is necessary but
never sufficient — cost, latency, interpretability, and maintainability are co-equal selection
criteria.

## Quick self-check
