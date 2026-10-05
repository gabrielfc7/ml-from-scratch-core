"""Base classes and utility functions for estimators in ml_from_scratch."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEstimator(ABC):
    """Abstract base class for all estimators in ml_from_scratch."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        params: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if not key.endswith("_") and not key.startswith("_"):
                params[key] = copy.deepcopy(value) if deep else value
        return params

    def set_params(self, **params: Any) -> BaseEstimator:
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
        return self

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Fit estimator to training data."""
        pass


class ClassifierMixin:
    """Mixin class for all classifiers in ml_from_scratch."""

    _estimator_type = "classifier"

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        from ml_from_scratch.metrics import accuracy_score

        predictions = self.predict(X)  # type: ignore[attr-defined]
        return accuracy_score(y, predictions)


class RegressorMixin:
    """Mixin class for all regressors in ml_from_scratch."""

    _estimator_type = "regressor"

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        from ml_from_scratch.metrics import r2_score

        predictions = self.predict(X)  # type: ignore[attr-defined]
        return r2_score(y, predictions)


def check_array(
    X: Any,
    ensure_2d: bool = True,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Validate and convert an input into a contiguous NumPy array."""
    arr = np.asarray(X, dtype=dtype)
    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    if ensure_2d:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got array with shape {arr.shape}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Input array contains NaN or infinity values.")
    return np.ascontiguousarray(arr)


def check_X_y(
    X: Any,
    y: Any,
    ensure_2d: bool = True,
    dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate input feature matrix X and target vector y."""
    X_arr = check_array(X, ensure_2d=ensure_2d, dtype=dtype)
    y_arr = np.asarray(y)

    if y_arr.size == 0:
        raise ValueError("Target array y must not be empty.")
    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: X has {len(X_arr)}, y has {len(y_arr)}"
        )
    if np.isnan(y_arr).any() or np.isinf(y_arr).any():
        raise ValueError("Target array contains NaN or infinity values.")

    return X_arr, y_arr


def check_is_fitted(estimator: Any, attributes: list[str]) -> None:
    """Verify that an estimator has been fitted by checking internal attributes."""
    for attr in attributes:
        if not hasattr(estimator, attr) or getattr(estimator, attr) is None:
            raise RuntimeError(
                f"This {estimator.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator."
            )
