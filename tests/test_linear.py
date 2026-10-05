"""Tests for LogisticRegression and optimization algorithms."""

import numpy as np
import pytest
from ml_from_scratch.linear import LogisticRegression
from ml_from_scratch.optimizers import Adam, CosineAnnealingLR, SGD, StepLR


def test_logistic_regression_binary(binary_classification_data):
    X, y = binary_classification_data
    model = LogisticRegression(optimizer="adam", learning_rate=0.05, max_iter=300, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    acc = model.score(X, y)
    probs = model.predict_proba(X)

    assert acc >= 0.85
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert model.coef_.shape == (1, X.shape[1])
    assert model.intercept_.shape == (1,)


def test_logistic_regression_multinomial(multiclass_data):
    X, y = multiclass_data
    # Standardize features for logistic regression
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    model = LogisticRegression(optimizer="adam", learning_rate=0.05, max_iter=400, random_state=42)
    model.fit(X_scaled, y)

    acc = model.score(X_scaled, y)
    probs = model.predict_proba(X_scaled)

    assert acc >= 0.90
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)
    assert model.coef_.shape == (3, X.shape[1])
    assert model.intercept_.shape == (3,)


def test_logistic_regression_sgd_optimizer(binary_classification_data):
    X, y = binary_classification_data
    model = LogisticRegression(optimizer="sgd", learning_rate=0.05, max_iter=300, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) >= 0.80


def test_logistic_regression_regularization_shrinkage(binary_classification_data):
    X, y = binary_classification_data

    # Weak regularization (C = 1000) vs strong regularization (C = 0.01)
    weak_reg = LogisticRegression(penalty="l2", C=1000.0, max_iter=300, random_state=42)
    strong_reg = LogisticRegression(penalty="l2", C=0.01, max_iter=300, random_state=42)

    weak_reg.fit(X, y)
    strong_reg.fit(X, y)

    weak_norm = np.linalg.norm(weak_reg.coef_)
    strong_norm = np.linalg.norm(strong_reg.coef_)

    # Strong regularization must shrink coefficients towards zero
    assert strong_norm < weak_norm


def test_logistic_regression_elasticnet_penalty(binary_classification_data):
    X, y = binary_classification_data
    model = LogisticRegression(penalty="elasticnet", C=1.0, l1_ratio=0.5, max_iter=200, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) >= 0.80


def test_lr_schedulers():
    step_sched = StepLR(step_size=10, gamma=0.5)
    assert step_sched.get_lr(0, 0.1) == 0.1
    assert step_sched.get_lr(10, 0.1) == 0.05
    assert step_sched.get_lr(20, 0.1) == 0.025

    cos_sched = CosineAnnealingLR(t_max=50, eta_min=0.01)
    assert pytest.approx(cos_sched.get_lr(0, 0.1)) == 0.1
    assert pytest.approx(cos_sched.get_lr(50, 0.1)) == 0.01


def test_not_fitted_error():
    model = LogisticRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0, 2.0]]))
