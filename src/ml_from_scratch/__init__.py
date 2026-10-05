"""ml-from-scratch-core: Clean, zero-dependency NumPy reference implementations of ML algorithms."""

__version__ = "0.1.0"

from ml_from_scratch.autograd import MLP, Layer, Neuron, Value
from ml_from_scratch.ensemble import GradientBoostedClassifier, GradientBoostedRegressor
from ml_from_scratch.linear import LogisticRegression
from ml_from_scratch.optimizers import SGD, Adam, ConstantLR, CosineAnnealingLR, StepLR
from ml_from_scratch.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

__all__ = [
    "LogisticRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostedClassifier",
    "GradientBoostedRegressor",
    "Value",
    "Neuron",
    "Layer",
    "MLP",
    "SGD",
    "Adam",
    "ConstantLR",
    "StepLR",
    "CosineAnnealingLR",
]
