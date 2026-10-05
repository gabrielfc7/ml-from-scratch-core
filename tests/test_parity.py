"""Parity tests asserting mathematical equivalence and metric convergence with scikit-learn."""

import numpy as np
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGBC,
    GradientBoostingRegressor as SklearnGBR,
    RandomForestClassifier as SklearnRFC,
    RandomForestRegressor as SklearnRFR,
)
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.metrics import (
    accuracy_score as sk_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    log_loss as sk_log_loss,
    mean_squared_error as sk_mean_squared_error,
    r2_score as sk_r2_score,
)
from sklearn.tree import (
    DecisionTreeClassifier as SklearnDTC,
    DecisionTreeRegressor as SklearnDTR,
)

from ml_from_scratch.ensemble import GradientBoostedClassifier, GradientBoostedRegressor
from ml_from_scratch.linear import LogisticRegression
from ml_from_scratch.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_squared_error,
    r2_score,
)
from ml_from_scratch.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


def test_metrics_strict_numerical_parity_with_sklearn(regression_data, binary_classification_data):
    """Assert our scratch metric implementations match scikit-learn to within 1e-7."""
    X_reg, y_reg = regression_data
    y_reg_pred = y_reg + np.random.RandomState(42).normal(0, 0.5, size=len(y_reg))

    # Mean Squared Error parity
    mse_scratch = mean_squared_error(y_reg, y_reg_pred)
    mse_sklearn = sk_mean_squared_error(y_reg, y_reg_pred)
    assert abs(mse_scratch - mse_sklearn) < 1e-7

    # R2 Score parity
    r2_scratch = r2_score(y_reg, y_reg_pred)
    r2_sklearn = sk_r2_score(y_reg, y_reg_pred)
    assert abs(r2_scratch - r2_sklearn) < 1e-7

    # Classification accuracy & confusion matrix parity
    X_cls, y_cls = binary_classification_data
    y_cls_pred = y_cls.copy()
    y_cls_pred[:20] = 1 - y_cls_pred[:20]  # Introduce some errors

    acc_scratch = accuracy_score(y_cls, y_cls_pred)
    acc_sklearn = sk_accuracy_score(y_cls, y_cls_pred)
    assert abs(acc_scratch - acc_sklearn) < 1e-7

    cm_scratch = confusion_matrix(y_cls, y_cls_pred)
    cm_sklearn = sk_confusion_matrix(y_cls, y_cls_pred)
    np.testing.assert_array_equal(cm_scratch, cm_sklearn)

    # Log Loss parity within 1e-4
    rng = np.random.RandomState(42)
    probs_scratch = rng.uniform(0.01, 0.99, size=len(y_cls))
    loss_scratch = log_loss(y_cls, probs_scratch)
    loss_sklearn = sk_log_loss(y_cls, probs_scratch)
    assert abs(loss_scratch - loss_sklearn) < 1e-4


def test_logistic_regression_parity_with_sklearn(breast_cancer_data):
    """Verify LogisticRegression accuracy is on par with scikit-learn on Breast Cancer."""
    X, y = breast_cancer_data
    # Standardize data
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std

    # Fit scikit-learn LogisticRegression
    sk_model = SklearnLogReg(penalty="l2", C=1.0, solver="lbfgs", max_iter=500, random_state=42)
    sk_model.fit(X_scaled, y)
    sk_acc = sk_model.score(X_scaled, y)

    # Fit our scratch LogisticRegression with Adam
    scratch_model = LogisticRegression(penalty="l2", C=1.0, learning_rate=0.05, max_iter=500, random_state=42)
    scratch_model.fit(X_scaled, y)
    scratch_acc = scratch_model.score(X_scaled, y)

    # Both models should achieve high accuracy (> 95%) and be within 3% of each other
    assert sk_acc > 0.95
    assert scratch_acc > 0.95
    assert abs(scratch_acc - sk_acc) < 0.03


def test_decision_tree_classifier_parity_with_sklearn(multiclass_data):
    """Verify DecisionTreeClassifier achieves comparable performance to scikit-learn on Iris."""
    X, y = multiclass_data

    sk_dt = SklearnDTC(criterion="gini", max_depth=3, random_state=42)
    sk_dt.fit(X, y)
    sk_acc = sk_dt.score(X, y)

    scratch_dt = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    scratch_dt.fit(X, y)
    scratch_acc = scratch_dt.score(X, y)

    assert sk_acc > 0.94
    assert scratch_acc > 0.94
    assert abs(scratch_acc - sk_acc) <= 0.03


def test_decision_tree_regressor_parity_with_sklearn(regression_data):
    """Verify DecisionTreeRegressor achieves comparable R2 to scikit-learn."""
    X, y = regression_data

    sk_reg = SklearnDTR(max_depth=4, random_state=42)
    sk_reg.fit(X, y)
    sk_r2 = sk_reg.score(X, y)

    scratch_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
    scratch_reg.fit(X, y)
    scratch_r2 = scratch_reg.score(X, y)

    assert sk_r2 > 0.70
    assert scratch_r2 > 0.70
    assert abs(scratch_r2 - sk_r2) < 0.05


def test_random_forest_parity_with_sklearn(multiclass_data, regression_data):
    """Verify Random Forest performance parity with scikit-learn."""
    X_cls, y_cls = multiclass_data
    sk_rf = SklearnRFC(n_estimators=30, max_depth=4, random_state=42)
    sk_rf.fit(X_cls, y_cls)

    scratch_rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
    scratch_rf.fit(X_cls, y_cls)

    assert sk_rf.score(X_cls, y_cls) > 0.95
    assert scratch_rf.score(X_cls, y_cls) > 0.95

    # Regressor
    X_reg, y_reg = regression_data
    sk_rfr = SklearnRFR(n_estimators=30, max_depth=5, random_state=42)
    sk_rfr.fit(X_reg, y_reg)

    scratch_rfr = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
    scratch_rfr.fit(X_reg, y_reg)

    assert sk_rfr.score(X_reg, y_reg) > 0.85
    assert scratch_rfr.score(X_reg, y_reg) > 0.85


def test_gradient_boosting_parity_with_sklearn(regression_data, binary_classification_data):
    """Verify Gradient Boosting performance parity with scikit-learn."""
    X_reg, y_reg = regression_data
    sk_gbr = SklearnGBR(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42)
    sk_gbr.fit(X_reg, y_reg)

    scratch_gbr = GradientBoostedRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42)
    scratch_gbr.fit(X_reg, y_reg)

    assert sk_gbr.score(X_reg, y_reg) > 0.85
    assert scratch_gbr.score(X_reg, y_reg) > 0.85
    assert abs(scratch_gbr.score(X_reg, y_reg) - sk_gbr.score(X_reg, y_reg)) < 0.06

    # Classifier
    X_cls, y_cls = binary_classification_data
    sk_gbc = SklearnGBC(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42)
    sk_gbc.fit(X_cls, y_cls)

    scratch_gbc = GradientBoostedClassifier(n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42)
    scratch_gbc.fit(X_cls, y_cls)

    assert sk_gbc.score(X_cls, y_cls) > 0.90
    assert scratch_gbc.score(X_cls, y_cls) > 0.90
    assert abs(scratch_gbc.score(X_cls, y_cls) - sk_gbc.score(X_cls, y_cls)) < 0.05
