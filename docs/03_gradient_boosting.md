# Mathematical Foundations: Gradient Boosted Decision Trees (GBDT)

This document formalizes Jerome Friedman's Greedy Function Approximation framework implemented in `ml_from_scratch.ensemble.gbdt`.

---

## 1. Gradient Boosting as Functional Gradient Descent

Whereas standard gradient descent optimizes parameter vectors $\theta \in \mathbb{R}^p$, gradient boosting performs optimization directly in **function space** $\mathcal{F}$.

Given an empirical risk objective:

$$\min_{F \in \mathcal{F}} \sum_{i=1}^N L(y_i, F(x_i))$$

We construct an additive sequence of base hypothesis estimators (typically shallow regression trees $h_m(x)$):

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where $\eta \in (0, 1]$ represents the learning rate (shrinkage).

---

## 2. Gradient Boosted Regressor (Squared Loss)

For squared error loss:

$$L(y, F) = \frac{1}{2} (y - F)^2$$

### 2.1 Initialization

$$F_0(x) = \arg\min_c \sum_{i=1}^N \frac{1}{2} (y_i - c)^2 = \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$$

### 2.2 Pseudo-Residuals

The negative gradient of the loss with respect to current predictions evaluated at $F_{m-1}$:

$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}} = y_i - F_{m-1}(x_i)$$

In squared error regression, the pseudo-residuals are identical to the standard arithmetic residuals.

### 2.3 Iterative Update

1. Fit a regression tree $h_m(x)$ to predict residuals $\{ (x_i, r_{im}) \}_{i=1}^N$.
2. Update the ensemble predictor:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

---

## 3. Gradient Boosted Classifier (Binary Log-Loss)

For binary classification $y_i \in \{0, 1\}$, the loss is the Bernoulli deviance:

$$L(y, F) = - \left[ y \ln(p) + (1 - y) \ln(1 - p) \right], \quad \text{where } p = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$$

Here, $F(x) = \ln \left( \frac{p}{1 - p} \right)$ represents the predicted log-odds.

### 3.1 Initialization

$$F_0(x) = \ln \left( \frac{\bar{y}}{1 - \bar{y}} \right), \quad \text{where } \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$$

### 3.2 Pseudo-Residuals

Taking the negative gradient with respect to log-odds $F(x_i)$:

$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{m-1}} = y_i - p_{i, m-1}$$

The pseudo-residual represents the signed difference between the true label and current probability.

### 3.3 1-Step Newton-Raphson Terminal Leaf Update

Let $R_{jm}$ denote the $j$-th terminal leaf region of tree $h_m(x)$. Because logistic loss cannot be minimized in closed form over tree leaves, Friedman applies a single Newton-Raphson approximation step for each terminal node:

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

Expanding $L(y_i, F_{m-1}(x_i) + \gamma)$ as a second-order Taylor series around $\gamma = 0$:

$$\gamma_{jm} \approx \frac{\sum_{x_i \in R_{jm}} \left( -\frac{\partial L}{\partial F} \right)}{\sum_{x_i \in R_{jm}} \frac{\partial^2 L}{\partial F^2}} = \frac{\sum_{x_i \in R_{jm}} r_{im}}{\sum_{x_i \in R_{jm}} p_{i, m-1}(1 - p_{i, m-1})}$$

### 3.4 Terminal Ensemble Update

For each terminal leaf $j$, the leaf value is updated to $\gamma_{jm}$:

$$F_m(x) = F_{m-1}(x) + \eta \sum_{j} \gamma_{jm} \mathbf{1}(x \in R_{jm})$$

Final class probability predictions are retrieved by applying the logistic sigmoid:

$$\hat{P}(y = 1 \mid x) = \sigma(F_M(x)) = \frac{1}{1 + \exp(-F_M(x))}$$
