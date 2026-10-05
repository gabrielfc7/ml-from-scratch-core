# Mathematical Foundations: Linear Models & Optimization

This document details the mathematical derivation, loss formulations, and gradient updates implemented in `ml_from_scratch.linear` and `ml_from_scratch.optimizers`.

---

## 1. Binary Logistic Regression

### 1.1 Hypothesis and Sigmoid Activation

Given a feature matrix $X \in \mathbb{R}^{N \times D}$ and binary labels $y \in \{0, 1\}^N$, the linear predictor defines logits $z \in \mathbb{R}^N$:

$$z = X w + b$$

The predicted probability $p_i = P(y_i = 1 \mid x_i)$ is parameterized via the logistic sigmoid function $\sigma(z)$:

$$\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

To guarantee numerical stability against overflow when $z_i \ll 0$ or underflow when $z_i \gg 0$:

$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$

### 1.2 Binary Cross-Entropy Loss

Assuming independent and identically distributed Bernoulli trials, the negative log-likelihood (binary cross-entropy loss) over $N$ training samples is:

$$\mathcal{L}_{\text{BCE}}(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

### 1.3 Analytical Gradients

Differentiating $\mathcal{L}_{\text{BCE}}$ with respect to the weights $w$ and bias $b$:

$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

Applying the multivariate chain rule across all samples yields the vectorized gradients:

$$\nabla_w \mathcal{L}_{\text{BCE}} = \frac{1}{N} X^T (p - y)$$

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial b} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)$$

---

## 2. Multinomial Logistic Regression (Softmax)

For $K > 2$ discrete classes, let $W \in \mathbb{R}^{D \times K}$ and $b \in \mathbb{R}^K$. The linear activations produce a logit matrix $Z = X W + b \in \mathbb{R}^{N \times K}$.

### 2.1 Softmax Normalization

The predicted probability that sample $i$ belongs to class $k$ is given by the softmax function:

$$P_{ik} = \frac{\exp(Z_{ik} - \max_{j} Z_{ij})}{\sum_{c=1}^K \exp(Z_{ic} - \max_{j} Z_{ij})}$$

Subtracting the row-maximum $\max_j Z_{ij}$ prevents numerical overflow during exponentiation without altering relative probabilities.

### 2.2 Categorical Cross-Entropy Loss

Given one-hot encoded ground truth labels $Y \in \{0, 1\}^{N \times K}$:

$$\mathcal{L}_{\text{CCE}}(W, b) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K Y_{ik} \ln(P_{ik})$$

The vectorized gradient with respect to parameter matrix $W$ and bias vector $b$:

$$\nabla_W \mathcal{L}_{\text{CCE}} = \frac{1}{N} X^T (P - Y)$$

$$\nabla_b \mathcal{L}_{\text{CCE}} = \frac{1}{N} \sum_{i=1}^N (P_{i, :} - Y_{i, :})$$

---

## 3. Regularization Penalties

To control model complexity and prevent overfitting, penalty terms are incorporated with regularization strength $\lambda = \frac{1}{C \cdot N}$ (matching scikit-learn convention):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \Omega(w)$$

| Penalty Type | Objective Penalty $\Omega(w)$ | Subgradient $\nabla_w \Omega(w)$ |
| :--- | :--- | :--- |
| **L2 (Ridge)** | $\frac{\lambda}{2} \|w\|_2^2 = \frac{\lambda}{2} \sum_{j} w_j^2$ | $\lambda w$ |
| **L1 (Lasso)** | $\lambda \|w\|_1 = \lambda \sum_{j} \|w_j\|$ | $\lambda \cdot \text{sign}(w)$ |
| **ElasticNet** | $\lambda \left( \rho \|w\|_1 + \frac{1 - \rho}{2} \|w\|_2^2 \right)$ | $\lambda \left( \rho \cdot \text{sign}(w) + (1 - \rho) w \right)$ |

---

## 4. Optimization Engines

### 4.1 Stochastic Gradient Descent (SGD) with Momentum

For parameters $\theta \in \{w, b\}$ and gradient $g_t = \nabla_\theta \mathcal{L}$:

$$v_t = \beta v_{t-1} + g_t$$

$$\theta_{t+1} = \theta_t - \eta_t v_t$$

where $\beta \in [0, 1)$ is the momentum decay factor and $\eta_t$ is the learning rate.

### 4.2 Adam (Adaptive Moment Estimation)

Adam maintains running exponential moving averages of both the first raw moment (mean gradient) and the second uncentered moment (uncentered gradient variance):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Because $m_0 = 0$ and $v_0 = 0$, both moments are biased towards zero in early iterations. Analytical bias-corrected moments are computed as:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The parameter update rule is:

$$\theta_{t+1} = \theta_t - \frac{\eta_t}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default hyperparameter values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
