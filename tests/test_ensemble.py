"""Tests for Gradient Boosted Decision Tree (GBDT) engine."""

import numpy as np
import pytest
from ml_from_scratch.ensemble import GradientBoostedClassifier, GradientBoostedRegressor


def test_gradient_boosted_regressor_loss_reduction(regression_data):
    X, y = regression_data
    gbr = GradientBoostedRegressor(n_estimators=40, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X, y)

    # Initial loss must be greater than final loss
    assert gbr.train_loss_history_[0] > gbr.train_loss_history_[-1]

    # R2 score must be high
    r2 = gbr.score(X, y)
    assert r2 > 0.85
    assert len(gbr.trees_) == 40
    assert gbr.feature_importances_ is not None
    assert pytest.approx(np.sum(gbr.feature_importances_)) == 1.0


def test_gradient_boosted_classifier_binary(binary_classification_data):
    X, y = binary_classification_data
    gbc = GradientBoostedClassifier(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(X, y)

    acc = gbc.score(X, y)
    assert acc > 0.90

    probs = gbc.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_gradient_boosted_classifier_multiclass(multiclass_data):
    X, y = multiclass_data
    gbc = GradientBoostedClassifier(n_estimators=25, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(X, y)

    acc = gbc.score(X, y)
    assert acc > 0.92

    probs = gbc.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)
