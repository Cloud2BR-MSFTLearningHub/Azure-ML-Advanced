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

> **Note - What this shows:** The major learning families side by side. The distinguishing axis is the *feedback signal*:
> labeled answers (supervised), structure-only (unsupervised), or reward from interaction
> (reinforcement). Identifying which signal your data provides is the first step in choosing an
> approach.

![Types of ML based on objective](https://raw.githubusercontent.com/brown9804/ML_DS_path/main/_docs/img/types_of_ml_based_in_objective.png)

> **Note - What this shows:** ML tasks organized by *objective* (predict a class, predict a number, group records, reduce
> dimensions). Map your business question to one of these objectives before picking an algorithm —
> the objective constrains both the model family and the evaluation metric.

## Problem Types

- Supervised: classification and regression
- Unsupervised: clustering and association
- Reinforcement: control and policy optimization

## Data and notation basics

- Dataset: $D = (x_i, y_i)_{i=1}^{N}$ for supervised learning.
- Feature vector: $x_i \in \mathbb{R}^{d}$.
- Target/label: $y_i$.
- Model: $f_{\theta}(x)$ with parameters $\theta$.

## Supervised learning categories

| Category                   | Output type                 | Examples                          | Typical metrics              |
| -------------------------- | --------------------------- | --------------------------------- | ---------------------------- |
| Binary classification      | 0/1 class                   | Fraud yes/no, churn yes/no        | Precision, Recall, F1, AUC   |
| Multi-class classification | One of $K$ classes          | Product category, diagnosis class | Macro-F1, accuracy, log loss |
| Multi-label classification | Multiple classes per sample | Tagging documents/topics          | Micro-F1, Hamming loss       |
| Regression                 | Continuous value            | Price, demand, latency            | MAE, RMSE, $R^2$             |
| Time-series forecasting    | Future values over time     | Sales, energy, traffic            | MAPE, RMSE, sMAPE            |

### Classification vs regression intuition

- Classification predicts which class.
- Regression predicts how much.

The same feature set can support both depending on business objective.

## Unsupervised learning categories

| Category                 | Purpose                                     | Typical methods                          |
| ------------------------ | ------------------------------------------- | ---------------------------------------- |
| Clustering               | Group similar observations                  | K-Means, DBSCAN, hierarchical clustering |
| Dimensionality reduction | Compress features while retaining structure | PCA, UMAP, autoencoders                  |
| Association mining       | Find co-occurrence rules                    | Apriori, FP-growth                       |
| Anomaly detection        | Detect rare/abnormal patterns               | Isolation Forest, One-Class SVM          |

## Semi-supervised and self-supervised

- Semi-supervised is useful when labels are expensive. Example: you have 1,000 labelled medical images and 50,000 unlabelled ones. A semi-supervised approach trains on both, propagating labels from confident predictions.
- Self-supervised is common in foundation models (GPT, BERT, CLIP) and pretraining pipelines. The model is trained on a proxy task whose labels come from the data itself â€” e.g., predict the next word, reconstruct a masked patch.
- Both reduce dependence on manual labeling, which is expensive and slow at scale.

| Approach        | Label requirement       | Common algorithms                         |
| --------------- | ----------------------- | ----------------------------------------- |
| Supervised      | All samples labelled    | Logistic regression, XGBoost, NN          |
| Semi-supervised | Small fraction labelled | Label propagation, pseudo-labelling       |
| Self-supervised | No labels needed        | Masked autoencoders, contrastive learning |

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
V^{\pi}(s)=\mathbb{E}*{\pi}\left[\sum*{t=0}^{\infty}\gamma^t r_t\mid s_0=s\right]  
$$

Goal:

$$  
\max_{\pi}\mathbb{E}*{\pi}\left[\sum*{t=0}^{\infty}\gamma^t r_t\right]  
$$

Supervised objective:

$$  
\min_{\theta} \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i), y_i)  
$$

This is empirical risk minimization: find parameters minimizing average training loss.

Common loss functions:

$$  
\mathcal{L}*{MSE}=\frac{1}{N}\sum*{i=1}^{N}(y_i-\hat{y}_i)^2  
$$

$$  
\mathcal{L}*{BCE}=-\frac{1}{N}\sum*{i=1}^{N}\left[y_i\log(\hat{p}_i)+(1-y_i)\log(1-\hat{p}_i)\right]  
$$

Multiclass cross-entropy:

$$  
\mathcal{L}*{CE}=-\frac{1}{N}\sum*{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(\hat{p}_{ik})  
$$

Optimization objective with regularization:

$$  
\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\theta}(x_i),y_i)+\lambda R(\theta)  
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

| Split      | Typical proportion | Purpose                                     |
| ---------- | ------------------ | ------------------------------------------- |
| Train      | 60â€“80%           | Fit model parameters                        |
| Validation | 10â€“20%           | Tune hyperparameters and compare models     |
| Test       | 10â€“20%           | Final unbiased evaluation before deployment |

The test set must **never** be used during model selection. Using it for selection is a form of data leakage that makes offline scores over-optimistic.

Cross-validation: when data is limited, k-fold CV uses all data for both training and validation by rotating folds. K=5 or K=10 is typical.

## Bias-variance intuition

![Bias-variance trade-off](../assets/img/bias-variance-tradeoff.svg)

> **Note - How to read this chart:** As complexity grows, **bias squared** falls (the model can fit more) while **variance** rises
> (the model reacts more to the particular training sample). Their sum — total error — is a U-shape
> minimized at an intermediate complexity. Left of the minimum you underfit; right of it you
> overfit. The floor of the curve never reaches zero because of irreducible noise $\sigma^2$.

- High bias: model too simple, underfits. Symptom: low training accuracy and low test accuracy.
- High variance: model too complex, overfits. Symptom: high training accuracy, much lower test accuracy.

The expected test error decomposes as:

$$  
\mathbb{E}[(y-\hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2  
$$

where $\sigma^2$ is irreducible noise.

The practical goal is to balance both:

| Technique                     | Addresses                          |
| ----------------------------- | ---------------------------------- |
| More training data            | Reduces variance                   |
| Regularization (L1/L2)        | Reduces variance                   |
| Feature selection/engineering | Can reduce bias and variance       |
| More complex model            | Reduces bias (risk: more variance) |
| Ensemble methods              | Reduces both (usually)             |

## How to choose an ML type quickly

| If your question is...                             | Use...                   |
| -------------------------------------------------- | ------------------------ |
| Can I predict this known target?                   | Supervised learning      |
| Can I group similar records without labels?        | Unsupervised learning    |
| Can an agent learn through interaction and reward? | Reinforcement learning   |
| I have few labels but lots of unlabeled data       | Semi-supervised learning |

## Typical mistakes to avoid

- Using accuracy alone on highly imbalanced datasets.
- Mixing train/test data during preprocessing (data leakage).
- Ignoring concept drift after deployment.
- Treating model score as the only KPI without business impact validation.

## Deep dive: every concept, explained

This section unpacks the notation and objectives above so each symbol has a clear meaning  
and a reason for existing.

### Reading the supervised setup $D = (x_i, y_i)_{i=1}^{N}$

- $N$ is the number of training examples. More examples reduce **variance** (see below) because  
the empirical average is a tighter estimate of the true expectation.
- $x_i \in \mathbb{R}^{d}$ is a **feature vector**: a row of $d$ numbers describing one example.  
The dimension $d$ is the *feature space size*; high $d$ with small $N$ is the classic  
"curse of dimensionality", where data becomes sparse and distances lose meaning.
- $y_i$ is the **label**. Its type decides the task: discrete → classification, continuous →  
regression, ordered-over-time → forecasting.
- $f_\theta$ is the **model**: a parameterized function. $\theta$ are the **parameters**  
(weights) the optimizer adjusts. Anything you set *before* training (tree depth, learning  
rate, regularization strength) is a **hyperparameter**, tuned on validation data, not learned.

### Empirical risk minimization (ERM), step by step

The objective $\min_\theta \frac{1}{N}\sum_i \mathcal{L}(f_\theta(x_i), y_i)$ says:  
"choose parameters that make the average mistake on the training data as small as possible."

- $\mathcal{L}$ is the **loss function** — it scores how wrong a single prediction is.
- The $\frac{1}{N}\sum$ turns per-example losses into an **average** (the *empirical risk*),  
which is our computable stand-in for the true expected risk over $P(X,Y)$.
- ERM only works if the training sample resembles production data. When it does not, low  
training risk does not imply low real-world risk — this is exactly why we hold out test data.

### Why each loss function has the shape it does

- **MSE** $\frac{1}{N}\sum (y_i-\hat y_i)^2$ squares errors, so large mistakes dominate. It is  
the maximum-likelihood loss when noise is Gaussian, which is why it pairs with regression.
- **Binary cross-entropy (BCE)** $-\frac{1}{N}\sum [y\log\hat p + (1-y)\log(1-\hat p)]$  
measures the *surprise* of the true label under the predicted probability. It explodes when  
the model is confidently wrong ($\hat p \to 0$ while $y=1$), which strongly discourages  
overconfidence. It is the maximum-likelihood loss for a Bernoulli target.
- **Categorical cross-entropy** generalizes BCE to $K$ classes by summing surprise over the  
one-hot label $y_{ik}$.

### Gradient descent: what each symbol controls

The update $\theta_{t+1} = \theta_t - \eta\nabla_\theta\mathcal{L}$ is the workhorse of ML.

- $\nabla_\theta\mathcal{L}$ is the **gradient**: the direction of steepest *increase* of the  
loss. Moving in the *negative* gradient direction reduces loss.
- $\eta$ is the **learning rate** (step size). Too large → the optimizer overshoots and  
diverges; too small → training is slow and may stall in flat regions. In practice it is the  
single most important hyperparameter to tune.
- Variants matter operationally: **batch** GD uses all data per step (stable, slow);  
**stochastic** GD uses one example (noisy, fast); **mini-batch** GD (the standard) trades off  
both and is what deep-learning frameworks use.

### L1 vs L2 regularization: geometry and effect

Regularization adds a penalty $\lambda R(\theta)$ to discourage complex models:

- **L1** ($\lVert\theta\rVert_1$) has corners on its constraint region, so the optimum often  
lands exactly on an axis → some weights become **exactly zero** → automatic feature selection  
and sparse models.
- **L2** ($\lVert\theta\rVert_2^2$) shrinks all weights smoothly toward zero without forcing  
any to vanish → more stable, better-conditioned solutions.
- $\lambda$ is the **regularization strength**: higher $\lambda$ → simpler model → more bias,  
less variance. It is tuned on validation data, never on the test set.

### Bias: variance decomposition, in plain terms

$\mathbb{E}[(y-\hat f(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$ splits a model's  
expected error into three sources:

- **Bias** — error from wrong assumptions (model too simple to capture the pattern). High bias  
shows as *both* low train and low test accuracy (underfitting).
- **Variance** — error from sensitivity to the particular training sample. High variance shows  
as a large gap between high train accuracy and lower test accuracy (overfitting).
- $\sigma^2$ — **irreducible noise** in the labels themselves. No model can beat this floor; it  
is set by data quality, not algorithm choice.

The art of modeling is moving along this trade-off — adding capacity to cut bias, adding data  
or regularization to cut variance — until the *sum* is minimized.

### Cross-validation and why the test set is sacred

- **k-fold cross-validation** rotates which fold is held out, so every example is used for both  
training and validation across folds. The averaged score is a lower-variance estimate of  
generalization than a single split, essential when data is scarce.
- The **test set** is touched exactly once, at the very end. Any decision (model choice,  
threshold, hyperparameter) influenced by test performance leaks information and makes the  
reported score optimistically biased, a subtle but common form of **data leakage**.

## Quick self-check
