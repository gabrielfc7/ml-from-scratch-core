"""Tests for Decision Trees and Random Forests."""

import numpy as np
import pytest
from ml_from_scratch.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    entropy,
    gini_impurity,
)


def test_gini_and_entropy_calculations():
    # Pure distribution [10, 0] -> Gini = 0.0, Entropy = 0.0
    y_pure = np.zeros(10, dtype=int)
    classes = np.array([0, 1])
    assert gini_impurity(y_pure, classes) == 0.0
    assert entropy(y_pure, classes) == 0.0

    # Even distribution [5, 5] -> Gini = 0.5, Entropy = 1.0 bit
    y_even = np.array([0] * 5 + [1] * 5, dtype=int)
    assert pytest.approx(gini_impurity(y_even, classes)) == 0.5
    assert pytest.approx(entropy(y_even, classes)) == 1.0


def test_decision_tree_classifier_overfits_perfectly_on_clean_data(binary_classification_data):
    X, y = binary_classification_data
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.mean(preds == y) >= 0.98
    assert clf.feature_importances_ is not None
    assert pytest.approx(np.sum(clf.feature_importances_)) == 1.0


def test_decision_tree_classifier_max_depth():
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, size=100)
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)

    def get_actual_depth(node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(get_actual_depth(node.left), get_actual_depth(node.right))

    assert get_actual_depth(clf.root_) <= 2


def test_decision_tree_regressor(regression_data):
    X, y = regression_data
    reg = DecisionTreeRegressor(max_depth=6, random_state=42)
    reg.fit(X, y)
    score = reg.score(X, y)
    assert score > 0.85
    assert reg.feature_importances_ is not None
    assert pytest.approx(np.sum(reg.feature_importances_)) == 1.0


def test_random_forest_classifier(multiclass_data):
    X, y = multiclass_data
    rf = RandomForestClassifier(n_estimators=30, max_depth=5, oob_score=True, random_state=42)
    rf.fit(X, y)

    train_score = rf.score(X, y)
    assert train_score > 0.95
    assert rf.oob_score_ is not None
    assert rf.oob_score_ > 0.85
    assert len(rf.estimators_) == 30

    probs = rf.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)


def test_random_forest_regressor(regression_data):
    X, y = regression_data
    rf = RandomForestRegressor(n_estimators=30, max_depth=6, oob_score=True, random_state=42)
    rf.fit(X, y)

    train_score = rf.score(X, y)
    assert train_score > 0.85
    assert rf.oob_score_ is not None
    assert rf.oob_score_ > 0.70
