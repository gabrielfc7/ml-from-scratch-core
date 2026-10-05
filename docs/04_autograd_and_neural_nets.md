# Mathematical Foundations: Reverse-Mode Automatic Differentiation & Neural Networks

This document details the mechanics of reverse-mode automatic differentiation (autodiff) and Multi-Layer Perceptrons implemented in `ml_from_scratch.autograd`.

---

## 1. Reverse-Mode Automatic Differentiation

Unlike numerical differentiation (which incurs truncation and round-off errors and requires $O(D)$ evaluations for $D$ parameters) or symbolic differentiation (which can suffer from expression swell), **reverse-mode automatic differentiation** computes exact analytical gradients in a single backward pass with $O(1)$ constant computational overhead relative to the forward pass.

### 1.1 Computational Graph Representation

Any mathematical expression is compiled into a Directed Acyclic Graph (DAG) $G = (V, E)$:
* **Nodes ($V$):** Intermediate scalar variables $v_i$.
* **Edges ($E$):** Functional dependencies $v_j = f_j(\text{Parents}(v_j))$.

### 1.2 Multivariate Chain Rule

Let $L$ denote the scalar loss function at the terminal sink node. For each intermediate node $v_i$, we define the adjoint (gradient):

$$\bar{v}_i \equiv \frac{\partial L}{\partial v_i}$$

By the multivariate chain rule across all child nodes $\text{Children}(v_i)$ that consume $v_i$:

$$\bar{v}_i = \sum_{j \in \text{Children}(v_i)} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_i}$$

### 1.3 Reverse Topological Sort

To guarantee that all child adjoints $\bar{v}_j$ have been completely accumulated before backpropagating to parent $v_i$, the evaluation order is determined by a **reverse topological sort** of the computation graph:

1. Perform Depth-First Search (DFS) post-order traversal from the root node $L$.
2. Initialize root gradient $\bar{L} = 1.0$.
3. Process nodes in reverse topological order, calling each node's local backward closure $\_backward()$.

---

## 2. Primitive Operation Adjoints

| Operation | Forward Evaluation | Local Partial Derivatives | Adjoint Accumulation |
| :--- | :--- | :--- | :--- |
| **Addition** | $z = x + y$ | $\frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1$ | $\bar{x} \mathrel{+}= \bar{z}, \quad \bar{y} \mathrel{+}= \bar{z}$ |
| **Multiplication** | $z = x \cdot y$ | $\frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x$ | $\bar{x} \mathrel{+}= y \cdot \bar{z}, \quad \bar{y} \mathrel{+}= x \cdot \bar{z}$ |
| **Power** | $z = x^k$ | $\frac{\partial z}{\partial x} = k x^{k-1}$ | $\bar{x} \mathrel{+}= (k x^{k-1}) \cdot \bar{z}$ |
| **Exponential** | $z = e^x$ | $\frac{\partial z}{\partial x} = e^x = z$ | $\bar{x} \mathrel{+}= z \cdot \bar{z}$ |
| **Logarithm** | $z = \ln(x)$ | $\frac{\partial z}{\partial x} = \frac{1}{x}$ | $\bar{x} \mathrel{+}= \frac{1}{x} \cdot \bar{z}$ |
| **ReLU** | $z = \max(0, x)$ | $\frac{\partial z}{\partial x} = \mathbf{1}(x > 0)$ | $\bar{x} \mathrel{+}= \mathbf{1}(x > 0) \cdot \bar{z}$ |
| **Sigmoid** | $z = \sigma(x) = \frac{1}{1 + e^{-x}}$ | $\frac{\partial z}{\partial x} = z (1 - z)$ | $\bar{x} \mathrel{+}= z(1 - z) \cdot \bar{z}$ |
| **Tanh** | $z = \tanh(x)$ | $\frac{\partial z}{\partial x} = 1 - z^2$ | $\bar{x} \mathrel{+}= (1 - z^2) \cdot \bar{z}$ |

---

## 3. Multi-Layer Perceptron (MLP)

### 3.1 Artificial Neuron

A single neuron $j$ in layer $l$ performs an affine transformation followed by non-linear activation:

$$a_j^{(l)} = \sum_{i=1}^{n_{l-1}} w_{ji}^{(l)} h_i^{(l-1)} + b_j^{(l)}$$

$$h_j^{(l)} = \phi\left(a_j^{(l)}\right)$$

Weights are initialized via Xavier / Glorot uniform initialization:

$$w \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

### 3.2 Gradient Backpropagation & Parameter Updates

For loss function $L(y, \hat{y})$:

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta} \quad \forall \theta \in \{ w_{ji}^{(l)}, b_j^{(l)} \}$$
