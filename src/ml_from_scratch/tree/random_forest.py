"""Random Forest Classifier and Regressor built via bootstrap aggregation."""

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
from ml_from_scratch.metrics import accuracy_score, r2_score
from ml_from_scratch.tree.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """A Random Forest classifier.

    A random forest is a meta-estimator that fits multiple CART decision trees on
    sub-samples of the dataset and uses averaging/majority voting to improve predictive
    accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {'gini', 'entropy'}, default='gini'
        Function to measure split quality.
    max_depth : int, optional, default=None
        Maximum depth of individual trees.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum samples required to be at a leaf node.
    max_features : {int, float, 'sqrt', 'log2', None}, default='sqrt'
        Number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate generalization score.
    random_state : int, optional, default=None
        Controls randomness of bootstrap sampling and feature selection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Build a forest of trees from the training set."""
        X_arr, y_arr = check_X_y(X, y)
        n_samples, n_features = X_arr.shape

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        importances_sum = np.zeros(n_features, dtype=np.float64)

        # For OOB score calculation
        oob_probs = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int64)

        for i in range(self.n_estimators):
            tree_seed = rng.randint(0, 100000000)
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            if self.bootstrap:
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                tree.fit(X_arr[boot_idx], y_arr[boot_idx])

                if self.oob_score:
                    oob_mask = np.ones(n_samples, dtype=bool)
                    oob_mask[boot_idx] = False
                    oob_indices = np.where(oob_mask)[0]
                    if len(oob_indices) > 0:
                        probs = tree.predict_proba(X_arr[oob_indices])
                        oob_probs[oob_indices] += probs
                        oob_counts[oob_indices] += 1
            else:
                tree.fit(X_arr, y_arr)

            self.estimators_.append(tree)
            if tree.feature_importances_ is not None:
                importances_sum += tree.feature_importances_

        self.feature_importances_ = importances_sum / self.n_estimators

        if self.oob_score and self.bootstrap:
            valid_oob = oob_counts > 0
            if np.any(valid_oob):
                oob_probs_valid = oob_probs[valid_oob] / oob_counts[valid_oob, None]
                oob_preds = self.classes_[np.argmax(oob_probs_valid, axis=1)]
                self.oob_score_ = accuracy_score(y_arr[valid_oob], oob_preds)
            else:
                self.oob_score_ = 0.0

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X as the average over all trees."""
        check_is_fitted(self, ["estimators_", "classes_"])
        X_arr = check_array(X)
        all_probs = np.zeros((len(X_arr), self.n_classes_), dtype=np.float64)
        for tree in self.estimators_:
            all_probs += tree.predict_proba(X_arr)
        return all_probs / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """A Random Forest regressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, optional, default=None
        Maximum depth of individual trees.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum samples required to be at a leaf node.
    max_features : {int, float, 'sqrt', 'log2', None}, default=1.0
        Number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to calculate out-of-bag R^2 score.
    random_state : int, optional, default=None
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        self.estimators_: list[DecisionTreeRegressor] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """Build a forest of trees from the training set."""
        X_arr, y_arr = check_X_y(X, y)
        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        importances_sum = np.zeros(n_features, dtype=np.float64)

        oob_preds_sum = np.zeros(n_samples, dtype=np.float64)
        oob_counts = np.zeros(n_samples, dtype=np.int64)

        for i in range(self.n_estimators):
            tree_seed = rng.randint(0, 100000000)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            if self.bootstrap:
                boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
                tree.fit(X_arr[boot_idx], y_arr[boot_idx])

                if self.oob_score:
                    oob_mask = np.ones(n_samples, dtype=bool)
                    oob_mask[boot_idx] = False
                    oob_indices = np.where(oob_mask)[0]
                    if len(oob_indices) > 0:
                        preds = tree.predict(X_arr[oob_indices])
                        oob_preds_sum[oob_indices] += preds
                        oob_counts[oob_indices] += 1
            else:
                tree.fit(X_arr, y_arr)

            self.estimators_.append(tree)
            if tree.feature_importances_ is not None:
                importances_sum += tree.feature_importances_

        self.feature_importances_ = importances_sum / self.n_estimators

        if self.oob_score and self.bootstrap:
            valid_oob = oob_counts > 0
            if np.any(valid_oob):
                oob_mean_pred = oob_preds_sum[valid_oob] / oob_counts[valid_oob]
                self.oob_score_ = r2_score(y_arr[valid_oob], oob_mean_pred)
            else:
                self.oob_score_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X as average of tree predictions."""
        check_is_fitted(self, ["estimators_"])
        X_arr = check_array(X)
        all_preds = np.zeros(len(X_arr), dtype=np.float64)
        for tree in self.estimators_:
            all_preds += tree.predict(X_arr)
        return all_preds / len(self.estimators_)
