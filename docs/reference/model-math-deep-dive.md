
# Mathematical Deep Dive: Core ML Models

This page is an expanded mathematical deep dive of the model families used across
this training hub. It is designed as a practical reference for engineers who want
both formula-level understanding and production intuition.

## How to use this page

1. Start with objective function and assumptions.
2. Review optimization behavior and regularization knobs.
3. Check failure modes before deployment.
4. Connect to metric choice in the performance module.

---

## 1) Linear Regression

Model form:

$$
\hat{y} = X\theta + b
$$

Least-squares objective:

$$
\min_{\theta,b}\;\frac{1}{N}\|y-(X\theta+b)\|_2^2
$$

Closed-form (normal equation, centered form):

$$
\hat{\theta}=(X^TX)^{-1}X^Ty
$$

Assumptions and notes:

- Linear relationship between features and target.
- Sensitive to multicollinearity and outliers.
- Great baseline for tabular regression.

Regularized variants:

- Ridge: add $\lambda\|\theta\|_2^2$
- Lasso: add $\lambda\|\theta\|_1$
- Elastic Net: combine both for stability + sparsity.

---

## 2) Logistic Regression

Binary probability:

$$
P(y=1\mid x)=\sigma(\theta^Tx+b)=\frac{1}{1+e^{-(\theta^Tx+b)}}
$$

Binary cross-entropy loss:

$$
\min_{\theta,b}\;-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\right]
$$

Decision rule:

$$
\hat{y}=\mathbb{1}[\hat{p}>\tau]
$$

Practical points:

- Coefficients are interpretable in log-odds space.
- Class imbalance requires threshold tuning and often class weights.
- Use calibration checks when probabilities drive business decisions.

---

## 3) Naive Bayes

Bayes rule with conditional independence:

$$
P(y\mid x_1,\dots,x_d)\propto P(y)\prod_{j=1}^{d}P(x_j\mid y)
$$

Prediction:

$$
\hat{y}=\arg\max_y\left[\log P(y)+\sum_{j=1}^{d}\log P(x_j\mid y)\right]
$$

Why it works despite the assumption:

- Independence is often false, but ranking still can be strong.
- Performs well for sparse high-dimensional text tasks.

Failure mode:

- Correlated features can distort posterior estimates.

---

## 4) Support Vector Machines (SVM)

Hard-margin formulation:

$$
\min_{w,b}\;\frac{1}{2}\|w\|^2 \quad \text{s.t.}\; y_i(w^Tx_i+b)\ge 1
$$

Soft-margin (hinge loss):

$$
\min_{w,b,\xi}\;\frac{1}{2}\|w\|^2 + C\sum_i\xi_i
\quad \text{s.t.}\; y_i(w^Tx_i+b)\ge 1-\xi_i,\;\xi_i\ge 0
$$

Kernel trick:

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
$$

Practical notes:

- Strong on medium-size datasets.
- Kernel SVM scales poorly on very large $N$.
- Hyperparameters $C$ and kernel parameters dominate behavior.

---

## 5) Decision Trees

Split criterion examples:

$$
\text{Gini}(S)=1-\sum_{k=1}^{K}p_k^2
$$

$$
H(S)=-\sum_{k=1}^{K}p_k\log_2 p_k
$$

Information gain for split $v$:

$$
IG=H(S)-\sum_v\frac{|S_v|}{|S|}H(S_v)
$$

Pros and risks:

- Highly interpretable, handles mixed data types.
- Unpruned trees overfit quickly.
- Depth and min-samples controls are key regularizers.

---

## 6) Random Forest

Ensemble prediction (regression):

$$
\hat{y}=\frac{1}{T}\sum_{t=1}^{T}f_t(x)
$$

Classification via majority vote:

$$
\hat{y}=\text{mode}(f_1(x),\dots,f_T(x))
$$

Core idea:

- Bagging + random feature subsets reduce variance.
- Better generalization than a single deep tree.

Trade-offs:

- Higher memory and inference cost than linear models.
- Strong baseline for tabular data with minimal tuning.

---

## 7) Gradient Boosting (XGBoost / LightGBM)

Additive stage-wise model:

$$
F_m(x)=F_{m-1}(x)+\nu h_m(x)
$$

where $h_m$ is trained on loss gradients.

Generic objective:

$$
\mathcal{L}=\sum_{i=1}^{N}\ell(y_i,\hat{y}_i)+\sum_{m}\Omega(h_m)
$$

with regularization on tree complexity.

Why it wins often on tabular:

- Captures nonlinear interactions well.
- Handles heterogeneous feature scales.
- Rich regularization and shrinkage controls.

Most important knobs:

- `learning_rate`, `n_estimators`
- `max_depth` or `num_leaves`
- `subsample`, `colsample_bytree`
- regularization terms (`lambda_l1`, `lambda_l2`)

---

## 8) Neural Networks (MLP basics)

Layer mapping:

$$
a^{(l)}=\phi\left(W^{(l)}a^{(l-1)}+b^{(l)}\right)
$$

Empirical risk minimization:

$$
\min_{\Theta}\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_{\Theta}(x_i),y_i)
$$

Gradient descent update:

$$
\Theta_{t+1}=\Theta_t-\eta\nabla_{\Theta}\mathcal{L}
$$

Regularization options:

- Weight decay ($L_2$)
- Dropout
- Early stopping
- Data augmentation

Operational note:

- Capacity is high, so validation strategy and monitoring are mandatory.

---

## 9) Time-Series Forecasting Models

Autoregressive family (AR):

$$
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\epsilon_t
$$

ARIMA adds differencing and moving average terms:

$$
\phi(B)(1-B)^d y_t = c + \theta(B)\epsilon_t
$$

Where $B$ is backshift operator.

Practical forecasting constraints:

- Use chronological validation only.
- Evaluate both scale-dependent and percentage metrics.
- Refit cadence should follow drift and seasonality changes.

---

## 10) Model Comparison from a Mathematical Lens

| Family | Main objective | Typical optimization | Common risk |
|---|---|---|---|
| Linear/Logistic | Convex loss + optional regularization | Deterministic convex methods | Underfitting nonlinear patterns |
| SVM | Margin maximization + hinge loss | Quadratic optimization | Scaling on large datasets |
| Trees/Forests | Impurity minimization | Greedy recursive splitting | Overfit without constraints |
| Boosting | Additive loss reduction | Gradient-based stage updates | Overfit if too many deep trees |
| Neural Nets | Non-convex empirical risk minimization | SGD/Adam backprop | Instability, data hunger |

---

## 11) Choosing the Right Model in Azure ML

Suggested workflow:

1. Start with linear/logistic and tree baselines.
2. Move to boosting for tabular accuracy gains.
3. Use neural architectures for unstructured data.
4. Validate with business-aligned metrics and threshold policy.
5. Deploy with monitoring for drift, latency, and calibration.

A mathematically elegant model is not automatically the best production model. In
practice, the best model maximizes business value under latency, cost, governance,
and maintainability constraints.
