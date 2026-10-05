"""CART Decision Tree Classifier and Regressor implemented in pure NumPy."""

from __future__ import annotations

from dataclasses import dataclass
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
from ml_from_scratch.tree.criterion import entropy, gini_impurity, variance_reduction


@dataclass
class DecisionTreeNode:
    """A node inside the Decision Tree."""

    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[DecisionTreeNode] = None
    right: Optional[DecisionTreeNode] = None
    value: Optional[Union[np.ndarray, float]] = None
    impurity: float = 0.0
    n_samples: int = 0
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """Classification and Regression Trees (CART) for Classification.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        Function to measure the quality of a split.
    max_depth : int, optional, default=None
        Maximum depth of the tree. If None, nodes expand until all leaves are pure or contain less than min_samples_split.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {int, float, 'sqrt', 'log2', None}, default=None
        Number of features to consider when looking for the best split.
    random_state : int, optional, default=None
        Controls feature subsampling randomness when max_features < n_features.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[DecisionTreeNode] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            return gini_impurity(y, np.arange(self.n_classes_))
        elif self.criterion == "entropy":
            return entropy(y, np.arange(self.n_classes_))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _determine_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Unknown max_features: {self.max_features}")

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
    ) -> tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        current_impurity = self._impurity(y)
        if current_impurity == 0.0:
            return None, None, 0.0

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        m_features = self._determine_max_features(n_features)
        feature_indices = (
            rng.choice(n_features, size=m_features, replace=False)
            if m_features < n_features
            else np.arange(n_features)
        )

        for feat in feature_indices:
            x_col = X[:, feat]
            sort_idx = np.argsort(x_col)
            x_sorted, y_sorted = x_col[sort_idx], y[sort_idx]

            # Find split boundaries where feature values change
            split_mask = x_sorted[:-1] != x_sorted[1:]
            split_indices = np.where(split_mask)[0]

            for idx in split_indices:
                n_left = idx + 1
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                threshold = (x_sorted[idx] + x_sorted[idx + 1]) / 2.0
                left_y = y_sorted[:n_left]
                right_y = y_sorted[n_left:]

                left_imp = self._impurity(left_y)
                right_imp = self._impurity(right_y)

                gain = current_impurity - (n_left / n_samples) * left_imp - (n_right / n_samples) * right_imp

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = threshold

        return best_feat, best_thresh, best_gain

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.RandomState
    ) -> DecisionTreeNode:
        n_samples = len(y)
        counts = np.bincount(y, minlength=self.n_classes_)
        node_probabilities = counts / n_samples
        current_impurity = self._impurity(y)

        # Check stopping conditions
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or current_impurity == 0.0
        ):
            return DecisionTreeNode(
                value=node_probabilities,
                impurity=current_impurity,
                n_samples=n_samples,
            )

        feat, thresh, gain = self._best_split(X, y, rng)

        if feat is None or gain <= 1e-12:
            return DecisionTreeNode(
                value=node_probabilities,
                impurity=current_impurity,
                n_samples=n_samples,
            )

        # Record feature importance (weighted impurity decrease)
        self.feature_importances_[feat] += n_samples * gain

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)

        return DecisionTreeNode(
            feature=feat,
            threshold=thresh,
            left=left_child,
            right=right_child,
            impurity=current_impurity,
            n_samples=n_samples,
            gain=gain,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        """Build a decision tree classifier from training data."""
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_arr.shape[1]

        # Map y to consecutive integers 0..K-1
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([label_to_idx[val] for val in y_arr], dtype=np.int64)

        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)

        self.root_ = self._build_tree(X_arr, y_encoded, depth=0, rng=rng)

        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _predict_row_proba(self, node: DecisionTreeNode, row: np.ndarray) -> np.ndarray:
        if node.is_leaf:
            return node.value  # type: ignore[return-value]
        if row[node.feature] <= node.threshold:
            return self._predict_row_proba(node.left, row)  # type: ignore[arg-type]
        else:
            return self._predict_row_proba(node.right, row)  # type: ignore[arg-type]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        check_is_fitted(self, ["root_", "classes_"])
        X_arr = check_array(X)
        probs = [self._predict_row_proba(self.root_, row) for row in X_arr]  # type: ignore[arg-type]
        return np.asarray(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """Classification and Regression Trees (CART) for Regression.

    Parameters
    ----------
    max_depth : int, optional, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {int, float, 'sqrt', 'log2', None}, default=None
        Number of features to consider when looking for the best split.
    random_state : int, optional, default=None
        Random state for feature selection.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[DecisionTreeNode] = None
        self.n_features_in_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    def _determine_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Unknown max_features: {self.max_features}")

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
    ) -> tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0.0

        current_var = variance_reduction(y)
        if current_var <= 1e-12:
            return None, None, 0.0

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        m_features = self._determine_max_features(n_features)
        feature_indices = (
            rng.choice(n_features, size=m_features, replace=False)
            if m_features < n_features
            else np.arange(n_features)
        )

        for feat in feature_indices:
            x_col = X[:, feat]
            sort_idx = np.argsort(x_col)
            x_sorted, y_sorted = x_col[sort_idx], y[sort_idx]

            split_mask = x_sorted[:-1] != x_sorted[1:]
            split_indices = np.where(split_mask)[0]

            for idx in split_indices:
                n_left = idx + 1
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                threshold = (x_sorted[idx] + x_sorted[idx + 1]) / 2.0
                left_y = y_sorted[:n_left]
                right_y = y_sorted[n_left:]

                left_var = variance_reduction(left_y)
                right_var = variance_reduction(right_y)

                gain = current_var - (n_left / n_samples) * left_var - (n_right / n_samples) * right_var

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = threshold

        return best_feat, best_thresh, best_gain

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.RandomState
    ) -> DecisionTreeNode:
        n_samples = len(y)
        mean_val = float(np.mean(y))
        current_var = variance_reduction(y)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or current_var <= 1e-12
        ):
            return DecisionTreeNode(
                value=mean_val,
                impurity=current_var,
                n_samples=n_samples,
            )

        feat, thresh, gain = self._best_split(X, y, rng)

        if feat is None or gain <= 1e-12:
            return DecisionTreeNode(
                value=mean_val,
                impurity=current_var,
                n_samples=n_samples,
            )

        self.feature_importances_[feat] += n_samples * gain

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)

        return DecisionTreeNode(
            feature=feat,
            threshold=thresh,
            left=left_child,
            right=right_child,
            impurity=current_var,
            n_samples=n_samples,
            gain=gain,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from training data."""
        X_arr, y_arr = check_X_y(X, y)
        self.n_features_in_ = X_arr.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)

        self.root_ = self._build_tree(X_arr, y_arr.astype(np.float64), depth=0, rng=rng)

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _predict_row(self, node: DecisionTreeNode, row: np.ndarray) -> float:
        if node.is_leaf:
            return float(node.value)  # type: ignore[arg-type]
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)  # type: ignore[arg-type]
        else:
            return self._predict_row(node.right, row)  # type: ignore[arg-type]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression targets for X."""
        check_is_fitted(self, ["root_"])
        X_arr = check_array(X)
        return np.array([self._predict_row(self.root_, row) for row in X_arr], dtype=np.float64)  # type: ignore[arg-type]
