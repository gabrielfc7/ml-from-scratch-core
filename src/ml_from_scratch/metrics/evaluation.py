"""Evaluation metrics implemented in vectorized NumPy."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if y_t.shape != y_p.shape:
        raise ValueError(f"Shape mismatch: y_true {y_t.shape} vs y_pred {y_p.shape}")
    return float(np.mean(y_t == y_p))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute confusion matrix to evaluate accuracy of classification."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_t, y_p]))
    else:
        labels = np.asarray(labels)

    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)

    for true_val, pred_val in zip(y_t, y_p):
        if true_val in label_to_idx and pred_val in label_to_idx:
            cm[label_to_idx[true_val], label_to_idx[pred_val]] += 1

    return cm


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    average: str = "binary",
) -> float:
    """Compute precision score."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((y_t == pos_label) & (y_p == pos_label))
        fp = np.sum((y_t != pos_label) & (y_p == pos_label))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else 0.0

    classes = np.unique(np.concatenate([y_t, y_p]))
    precisions = []
    for cls in classes:
        tp = np.sum((y_t == cls) & (y_p == cls))
        fp = np.sum((y_t != cls) & (y_p == cls))
        precisions.append(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0)

    return float(np.mean(precisions))


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    average: str = "binary",
) -> float:
    """Compute recall score."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((y_t == pos_label) & (y_p == pos_label))
        fn = np.sum((y_t == pos_label) & (y_p != pos_label))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else 0.0

    classes = np.unique(np.concatenate([y_t, y_p]))
    recalls = []
    for cls in classes:
        tp = np.sum((y_t == cls) & (y_p == cls))
        fn = np.sum((y_t == cls) & (y_p != cls))
        recalls.append(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)

    return float(np.mean(recalls))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Union[int, str] = 1,
    average: str = "binary",
) -> float:
    """Compute F1 score."""
    prec = precision_score(y_true, y_pred, pos_label=pos_label, average=average)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, average=average)
    if (prec + rec) == 0.0:
        return 0.0
    return float(2.0 * (prec * rec) / (prec + rec))


def log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Log loss, aka logistic loss or cross-entropy loss."""
    y_t = np.asarray(y_true)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)

    if p.ndim == 1 or p.shape[1] == 1:
        # Binary classification
        p = p.ravel()
        y_t = y_t.ravel()
        loss = -np.mean(y_t * np.log(p) + (1.0 - y_t) * np.log(1.0 - p))
        return float(loss)

    # Multi-class
    n_samples, n_classes = p.shape
    if y_t.ndim == 1:
        # Integer labels
        one_hot = np.zeros_like(p)
        one_hot[np.arange(n_samples), y_t.astype(int)] = 1.0
        y_t = one_hot

    # Renormalize rows
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.sum(y_t * np.log(p)) / n_samples
    return float(loss)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error regression loss."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_t - y_p) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error regression loss."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))
