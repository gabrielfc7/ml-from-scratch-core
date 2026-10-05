"""Logistic Regression with SGD/Adam optimizers, L1/L2 penalties, and Softmax."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ml_from_scratch.base import BaseEstimator, ClassifierMixin, check_array, check_is_fitted, check_X_y
from ml_from_scratch.optimizers import Adam, Optimizer, SGD


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation function."""
    z = np.asarray(z, dtype=np.float64)
    # For z >= 0: 1 / (1 + exp(-z))
    # For z < 0:  exp(z) / (1 + exp(z))
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result = np.empty_like(z)
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_neg = np.exp(z[neg_mask])
    result[neg_mask] = exp_neg / (1.0 + exp_neg)
    return result


def _stable_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation over 2D array along axis 1."""
    # Shift logits by row-max for numerical stability
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic Regression classifier supporting binary and multinomial targets.

    Supports custom first-order optimization (SGD, Adam) and L1/L2/ElasticNet penalties.

    Parameters
    ----------
    penalty : {'l2', 'l1', 'elasticnet', None}, default='l2'
        Regularization penalty.
    C : float, default=1.0
        Inverse of regularization strength (alpha = 1.0 / C). Smaller values specify stronger regularization.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. Used only if penalty='elasticnet'.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    max_iter : int, default=500
        Maximum number of iterations/epochs.
    tol : float, default=1e-5
        Tolerance for stopping criteria based on change in loss.
    optimizer : {'adam', 'sgd'} or Optimizer instance, default='adam'
        Optimization algorithm.
    learning_rate : float, default=0.01
        Initial learning rate for the built-in optimizer.
    batch_size : int, optional, default=None
        Mini-batch size. If None or >= n_samples, standard full-batch gradient descent is performed.
    random_state : int, optional, default=None
        Seed for shuffle and parameter initialization.
    """

    def __init__(
        self,
        penalty: Optional[str] = "l2",
        C: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 500,
        tol: float = 1e-5,
        optimizer: Union[str, Optimizer] = "adam",
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state

        # Attributes set during fit
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.loss_history_: list[float] = []

    def _get_optimizer(self) -> Optimizer:
        if isinstance(self.optimizer, Optimizer):
            return self.optimizer
        if self.optimizer == "adam":
            return Adam(lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return SGD(lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}. Expected 'adam', 'sgd', or Optimizer instance.")

    def _compute_loss_and_grad_binary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: float,
        reg_weight: float,
    ) -> tuple[float, np.ndarray, float]:
        """Compute binary cross-entropy loss and gradients."""
        n_samples = len(X)
        z = X @ w + b
        p = _stable_sigmoid(z)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)

        loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        # Error residual: p - y
        diff = p - y
        grad_w = (X.T @ diff) / n_samples
        grad_b = float(np.mean(diff)) if self.fit_intercept else 0.0

        # Regularization
        if self.penalty == "l2":
            loss += 0.5 * reg_weight * float(np.sum(w ** 2))
            grad_w += reg_weight * w
        elif self.penalty == "l1":
            loss += reg_weight * float(np.sum(np.abs(w)))
            grad_w += reg_weight * np.sign(w)
        elif self.penalty == "elasticnet":
            l1_w = self.l1_ratio * reg_weight
            l2_w = (1.0 - self.l1_ratio) * reg_weight
            loss += l1_w * float(np.sum(np.abs(w))) + 0.5 * l2_w * float(np.sum(w ** 2))
            grad_w += l1_w * np.sign(w) + l2_w * w

        return loss, grad_w, grad_b

    def _compute_loss_and_grad_multinomial(
        self,
        X: np.ndarray,
        Y_one_hot: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        reg_weight: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute multinomial categorical cross-entropy loss and gradients."""
        n_samples = len(X)
        z = X @ W + b
        p = _stable_softmax(z)
        p = np.clip(p, 1e-15, 1.0 - 1e-15)

        loss = -float(np.sum(Y_one_hot * np.log(p))) / n_samples

        diff = p - Y_one_hot
        grad_W = (X.T @ diff) / n_samples
        grad_b = np.mean(diff, axis=0) if self.fit_intercept else np.zeros_like(b)

        # Regularization
        if self.penalty == "l2":
            loss += 0.5 * reg_weight * float(np.sum(W ** 2))
            grad_W += reg_weight * W
        elif self.penalty == "l1":
            loss += reg_weight * float(np.sum(np.abs(W)))
            grad_W += reg_weight * np.sign(W)
        elif self.penalty == "elasticnet":
            l1_w = self.l1_ratio * reg_weight
            l2_w = (1.0 - self.l1_ratio) * reg_weight
            loss += l1_w * float(np.sum(np.abs(W))) + 0.5 * l2_w * float(np.sum(W ** 2))
            grad_W += l1_w * np.sign(W) + l2_w * W

        return loss, grad_W, grad_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Fit the logistic regression model according to the given training data."""
        X_arr, y_arr = check_X_y(X, y)
        n_samples, n_features = X_arr.shape

        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError(f"Classifier can't train when only 1 class is present: {self.classes_}")

        reg_weight = 0.0 if self.penalty is None or self.C <= 0 else (1.0 / (self.C * n_samples))
        optimizer = self._get_optimizer()
        optimizer.reset()

        self.loss_history_ = []

        if n_classes == 2:
            # Binary classification
            # Map classes to 0 and 1
            y_binary = (y_arr == self.classes_[1]).astype(np.float64)
            w = rng.randn(n_features) * 0.01
            b = 0.0

            prev_loss = float("inf")
            batch_sz = self.batch_size if (self.batch_size and self.batch_size < n_samples) else n_samples
            min_iter = min(20, self.max_iter)

            for epoch in range(self.max_iter):
                indices = rng.permutation(n_samples) if batch_sz < n_samples else np.arange(n_samples)

                for start_idx in range(0, n_samples, batch_sz):
                    batch_idx = indices[start_idx : start_idx + batch_sz]
                    X_b, y_b = X_arr[batch_idx], y_binary[batch_idx]

                    _, grad_w, grad_b = self._compute_loss_and_grad_binary(X_b, y_b, w, b, reg_weight)
                    w = optimizer.update(w, grad_w, "w")
                    if self.fit_intercept:
                        b = float(optimizer.update(np.array([b]), np.array([grad_b]), "b")[0])

                optimizer.step_epoch()

                # Evaluate full loss for convergence
                epoch_loss, _, _ = self._compute_loss_and_grad_binary(X_arr, y_binary, w, b, reg_weight)
                self.loss_history_.append(epoch_loss)

                if epoch >= min_iter and abs(prev_loss - epoch_loss) < self.tol:
                    self.n_iter_ = epoch + 1
                    break
                prev_loss = epoch_loss
            else:
                self.n_iter_ = self.max_iter

            self.coef_ = w.reshape(1, -1)
            self.intercept_ = np.array([b])

        else:
            # Multinomial classification
            class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
            Y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
            for i, label in enumerate(y_arr):
                Y_one_hot[i, class_to_idx[label]] = 1.0

            W = rng.randn(n_features, n_classes) * 0.01
            b = np.zeros(n_classes, dtype=np.float64)

            prev_loss = float("inf")
            batch_sz = self.batch_size if (self.batch_size and self.batch_size < n_samples) else n_samples
            min_iter = min(20, self.max_iter)

            for epoch in range(self.max_iter):
                indices = rng.permutation(n_samples) if batch_sz < n_samples else np.arange(n_samples)

                for start_idx in range(0, n_samples, batch_sz):
                    batch_idx = indices[start_idx : start_idx + batch_sz]
                    X_b, Y_b = X_arr[batch_idx], Y_one_hot[batch_idx]

                    _, grad_W, grad_b = self._compute_loss_and_grad_multinomial(X_b, Y_b, W, b, reg_weight)
                    W = optimizer.update(W, grad_W, "W")
                    if self.fit_intercept:
                        b = optimizer.update(b, grad_b, "b")

                optimizer.step_epoch()

                epoch_loss, _, _ = self._compute_loss_and_grad_multinomial(X_arr, Y_one_hot, W, b, reg_weight)
                self.loss_history_.append(epoch_loss)

                if epoch >= min_iter and abs(prev_loss - epoch_loss) < self.tol:
                    self.n_iter_ = epoch + 1
                    break
                prev_loss = epoch_loss
            else:
                self.n_iter_ = self.max_iter

            self.coef_ = W.T  # Shape: (n_classes, n_features) to match scikit-learn convention
            self.intercept_ = b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimation for each class."""
        check_is_fitted(self, ["coef_", "intercept_", "classes_"])
        X_arr = check_array(X)

        if len(self.classes_) == 2:
            # Binary
            z = X_arr @ self.coef_[0] + self.intercept_[0]
            p1 = _stable_sigmoid(z).reshape(-1, 1)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])
        else:
            # Multinomial: coef_ is (n_classes, n_features) -> transpose to (n_features, n_classes)
            z = X_arr @ self.coef_.T + self.intercept_
            return _stable_softmax(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]
