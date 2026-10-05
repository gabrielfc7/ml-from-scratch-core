"""Gradient Boosted Decision Tree (GBDT) engine showing exact residual fitting."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ml_from_scratch.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    check_array,
    check_is_fitted,
    check_X_y,
)
from ml_from_scratch.tree.decision_tree import DecisionTreeNode, DecisionTreeRegressor


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _find_leaf(node: DecisionTreeNode, row: np.ndarray) -> DecisionTreeNode:
    """Traverse tree to locate leaf node for a single sample."""
    curr = node
    while not curr.is_leaf:
        if row[curr.feature] <= curr.threshold:  # type: ignore[index]
            curr = curr.left  # type: ignore[assignment]
        else:
            curr = curr.right  # type: ignore[assignment]
    return curr


class GradientBoostedRegressor(BaseEstimator, RegressorMixin):
    """Gradient Boosted Regressor using Friedman's residual fitting under squared error loss.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Shrinkage parameter scaling the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of individual regression trees (typically shallow: 3-5).
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    random_state : int, optional, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.init_val_: float = 0.0
        self.trees_: list[DecisionTreeRegressor] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.train_loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostedRegressor:
        """Fit boosting ensemble by sequentially fitting regression trees to residuals."""
        X_arr, y_arr = check_X_y(X, y)
        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.random_state)

        # Base prediction F_0(x) = mean(y)
        self.init_val_ = float(np.mean(y_arr))
        f_pred = np.full(n_samples, self.init_val_, dtype=np.float64)

        self.trees_ = []
        self.train_loss_history_ = []
        importances_sum = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Compute negative gradient of 0.5 * (y - F)^2 -> pseudo-residuals: y - F
            residuals = y_arr - f_pred

            tree_seed = rng.randint(0, 100000000)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=tree_seed,
            )
            tree.fit(X_arr, residuals)

            # Update predictions: F_m(x) = F_{m-1}(x) + eta * h_m(x)
            f_pred += self.learning_rate * tree.predict(X_arr)
            self.trees_.append(tree)

            if tree.feature_importances_ is not None:
                importances_sum += tree.feature_importances_

            mse = float(np.mean((y_arr - f_pred) ** 2))
            self.train_loss_history_.append(mse)

        total_imp = np.sum(importances_sum)
        self.feature_importances_ = importances_sum / total_imp if total_imp > 0 else importances_sum
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X."""
        check_is_fitted(self, ["trees_"])
        X_arr = check_array(X)
        y_pred = np.full(len(X_arr), self.init_val_, dtype=np.float64)
        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X_arr)
        return y_pred


class GradientBoostedClassifier(BaseEstimator, ClassifierMixin):
    """Gradient Boosted Classifier using negative gradients of cross-entropy loss.

    For binary classification, fits trees to pseudo-residuals r_i = y_i - p_i
    and optimizes leaf values using Friedman's 1-step Newton-Raphson update.
    For multi-class classification, fits independent ensembles via One-vs-Rest.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Step size shrinkage.
    max_depth : int, default=3
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    random_state : int, optional, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.init_logits_: Union[float, list[float]] = 0.0
        self.trees_: Union[list[DecisionTreeRegressor], list[list[DecisionTreeRegressor]]] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def _fit_binary(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
    ) -> tuple[float, list[DecisionTreeRegressor], np.ndarray]:
        n_samples, n_features = X.shape

        # Initial log-odds: F_0 = ln(p_0 / (1 - p_0))
        p0 = float(np.mean(y))
        p0 = np.clip(p0, 1e-6, 1.0 - 1e-6)
        init_logit = float(np.log(p0 / (1.0 - p0)))
        f_pred = np.full(n_samples, init_logit, dtype=np.float64)

        trees: list[DecisionTreeRegressor] = []
        importances_sum = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_estimators):
            p = _sigmoid(f_pred)
            residuals = y - p

            tree_seed = rng.randint(0, 100000000)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=tree_seed,
            )
            tree.fit(X, residuals)

            # Newton-Raphson leaf value optimization: gamma_j = sum(r_i) / sum(p_i * (1 - p_i))
            leaf_map: dict[int, list[int]] = {}
            leaves: dict[int, DecisionTreeNode] = {}

            for i in range(n_samples):
                leaf = _find_leaf(tree.root_, X[i])  # type: ignore[arg-type]
                leaf_id = id(leaf)
                if leaf_id not in leaf_map:
                    leaf_map[leaf_id] = []
                    leaves[leaf_id] = leaf
                leaf_map[leaf_id].append(i)

            for leaf_id, sample_indices in leaf_map.items():
                idx_arr = np.array(sample_indices, dtype=np.int64)
                r_sum = np.sum(residuals[idx_arr])
                p_denom = np.sum(p[idx_arr] * (1.0 - p[idx_arr]))
                leaves[leaf_id].value = float(r_sum / (p_denom + 1e-10))

            f_pred += self.learning_rate * tree.predict(X)
            trees.append(tree)

            if tree.feature_importances_ is not None:
                importances_sum += tree.feature_importances_

        return init_logit, trees, importances_sum

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostedClassifier:
        """Fit gradient boosted classifier."""
        X_arr, y_arr = check_X_y(X, y)
        n_samples, n_features = X_arr.shape
        self.classes_ = np.unique(y_arr)
        rng = np.random.RandomState(self.random_state)

        if len(self.classes_) == 2:
            y_binary = (y_arr == self.classes_[1]).astype(np.float64)
            init_logit, trees, imp = self._fit_binary(X_arr, y_binary, rng)
            self.init_logits_ = init_logit
            self.trees_ = trees
            total_imp = np.sum(imp)
            self.feature_importances_ = imp / total_imp if total_imp > 0 else imp
        else:
            # One-vs-Rest for multiclass
            self.init_logits_ = []
            self.trees_ = []
            total_imp = np.zeros(n_features, dtype=np.float64)

            for cls in self.classes_:
                y_cls = (y_arr == cls).astype(np.float64)
                init_logit, trees, imp = self._fit_binary(X_arr, y_cls, rng)
                self.init_logits_.append(init_logit)
                self.trees_.append(trees)
                total_imp += imp

            imp_norm = np.sum(total_imp)
            self.feature_importances_ = total_imp / imp_norm if imp_norm > 0 else total_imp

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        check_is_fitted(self, ["classes_", "trees_"])
        X_arr = check_array(X)

        if len(self.classes_) == 2:
            logits = np.full(len(X_arr), self.init_logits_, dtype=np.float64)
            for tree in self.trees_:  # type: ignore[union-attr]
                logits += self.learning_rate * tree.predict(X_arr)
            p1 = _sigmoid(logits).reshape(-1, 1)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])
        else:
            all_probs = []
            for init_logit, tree_list in zip(self.init_logits_, self.trees_):  # type: ignore[arg-type]
                logits = np.full(len(X_arr), init_logit, dtype=np.float64)
                for tree in tree_list:
                    logits += self.learning_rate * tree.predict(X_arr)
                all_probs.append(_sigmoid(logits))

            probs = np.column_stack(all_probs)
            # Normalize to sum to 1
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
