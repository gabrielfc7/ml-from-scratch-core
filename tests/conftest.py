"""Test fixtures and synthetic datasets for ml_from_scratch test suites."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, make_classification, make_regression


@pytest.fixture
def binary_classification_data():
    """Linearly separable binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        class_sep=1.8,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_data():
    """Multi-class classification dataset (Iris)."""
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def breast_cancer_data():
    """Standard real-world binary classification dataset."""
    data = load_breast_cancer()
    return data.data, data.target


@pytest.fixture
def regression_data():
    """Synthetic regression dataset."""
    X, y = make_regression(
        n_samples=250,
        n_features=5,
        noise=0.1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def diabetes_data():
    """Standard real-world regression dataset."""
    data = load_diabetes()
    return data.data, data.target
