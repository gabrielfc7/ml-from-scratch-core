# Mathematical Foundations: CART Trees & Random Forests

This document presents the mathematical mechanics behind `ml_from_scratch.tree`.

---

## 1. Classification and Regression Trees (CART)

A decision tree recursively partitions the feature space $\mathcal{X} \subseteq \mathbb{R}^D$ into axis-aligned hyper-rectangles $R_1, R_2, \dots, R_M$.

### 1.1 Splitting Mechanism

At each internal node containing subset $Q$ with $|Q| = N_m$ samples, the algorithm evaluates all candidate features $j \in \{1, \dots, D\}$ and threshold splits $t \in \mathbb{R}$ dividing $Q$ into two disjoint children:

$$Q_{\text{left}}(j, t) = \{ x \in Q \mid x_j \le t \}$$

$$Q_{\text{right}}(j, t) = \{ x \in Q \mid x_j > t \}$$

The optimal split $(j^*, t^*)$ maximizes the impurity reduction (information gain):

$$\Delta I(Q, j, t) = I(Q) - \frac{|Q_{\text{left}}|}{N_m} I(Q_{\text{left}}) - \frac{|Q_{\text{right}}|}{N_m} I(Q_{\text{right}})$$

---

## 2. Impurity Criteria

### 2.1 Classification: Gini Impurity

Given empirical class proportions $p_{mk} = \frac{1}{N_m} \sum_{i \in Q_m} \mathbf{1}(y_i = k)$ for classes $k \in \{1, \dots, K\}$:

$$I_{\text{Gini}}(Q_m) = 1 - \sum_{k=1}^K p_{mk}^2$$

* Minimum value: $0.0$ (node contains samples from exactly one class; completely pure).
* Maximum value: $1 - \frac{1}{K}$ (uniform distribution across all classes).

### 2.2 Classification: Shannon Entropy

$$I_{\text{Entropy}}(Q_m) = -\sum_{k=1}^K p_{mk} \log_2(p_{mk}) \quad (\text{with } 0 \log_2 0 \equiv 0)$$

Measures the expected information content (in bits) required to describe class membership.

### 2.3 Regression: Variance Reduction (Mean Squared Error)

For continuous target values $y \in \mathbb{R}$:

$$I_{\text{MSE}}(Q_m) = \frac{1}{N_m} \sum_{i \in Q_m} (y_i - \bar{y}_m)^2, \quad \text{where } \bar{y}_m = \frac{1}{N_m} \sum_{i \in Q_m} y_i$$

The terminal leaf prediction is the empirical conditional expectation:

$$\hat{y}(x) = \bar{y}_m \quad \forall x \in R_m$$

---

## 3. Feature Importance (Mean Decrease in Impurity - MDI)

The importance of feature $j$ across a tree $T$ is the weighted sum of impurity decreases over all internal nodes $v$ where feature $j$ was selected for the split:

$$\text{Imp}(j) = \sum_{v \in T : \text{split}(v) = j} N_v \cdot \Delta I(Q_v, j, t_v)$$

The raw importances are normalized across all $D$ features so they sum to unity:

$$\text{MDI}_j = \frac{\text{Imp}(j)}{\sum_{k=1}^D \text{Imp}(k)}$$

---

## 4. Random Forests (Bagging & Subspace Sampling)

Random Forests reduce the variance of individual high-variance CART trees through Bootstrap Aggregation (Bagging) combined with random feature subspace projection:

1. **Bootstrap Resampling:** For each tree $b \in \{1, \dots, B\}$, draw a training sample $D_b$ of size $N$ with replacement from the original dataset $D$. Approximately $1 - \frac{1}{e} \approx 63.2\%$ of original samples are selected, leaving $36.8\%$ as Out-of-Bag (OOB) samples.
2. **Random Subspace Selection:** At each node split, only a random subset of $m \le D$ candidate features is evaluated:
   * Classification: $m = \lfloor \sqrt{D} \rfloor$
   * Regression: $m = \lfloor D / 3 \rfloor$ or $m = D$

### 4.1 Ensemble Aggregation

* **Classification (Soft Voting):**
  $$\hat{P}(y = k \mid x) = \frac{1}{B} \sum_{b=1}^B P_b(y = k \mid x)$$
  $$\hat{y}(x) = \arg\max_{k} \hat{P}(y = k \mid x)$$

* **Regression:**
  $$\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B \hat{y}_b(x)$$

### 4.2 Out-of-Bag (OOB) Generalization Score

For each training sample $i$, aggregate predictions only from the subset of trees $\mathcal{B}_i = \{ b \mid (x_i, y_i) \notin D_b \}$ where sample $i$ was omitted during bootstrap training:

$$\hat{y}_{\text{OOB}}(x_i) = \text{aggregate}_{b \in \mathcal{B}_i} \left( \hat{y}_b(x_i) \right)$$

The OOB score provides an unbiased estimate of generalization error without requiring an explicit cross-validation partition.
