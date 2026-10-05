"""Split criteria functions for classification and regression decision trees."""

from __future__ import annotations

import numpy as np


def gini_impurity(y: np.ndarray, classes: np.ndarray) -> float:
    """Compute Gini impurity for classification labels."""
    n_samples = len(y)
    if n_samples == 0:
        return 0.0

    counts = np.bincount(y, minlength=len(classes))
    probs = counts / n_samples
    return float(1.0 - np.sum(probs ** 2))


def entropy(y: np.ndarray, classes: np.ndarray) -> float:
    """Compute Shannon Entropy (in bits) for classification labels."""
    n_samples = len(y)
    if n_samples == 0:
        return 0.0

    counts = np.bincount(y, minlength=len(classes))
    probs = counts / n_samples
    non_zero_probs = probs[probs > 0]
    return float(-np.sum(non_zero_probs * np.log2(non_zero_probs)))


def variance_reduction(y: np.ndarray) -> float:
    """Compute variance (Mean Squared Error from mean) for regression targets."""
    n_samples = len(y)
    if n_samples <= 1:
        return 0.0
    mean_val = np.mean(y)
    return float(np.mean((y - mean_val) ** 2))


def mean_absolute_deviation(y: np.ndarray) -> float:
    """Compute Mean Absolute Error from median for regression targets."""
    n_samples = len(y)
    if n_samples <= 1:
        return 0.0
    med_val = np.median(y)
    return float(np.mean(np.abs(y - med_val)))
