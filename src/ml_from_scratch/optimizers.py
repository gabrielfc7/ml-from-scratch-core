"""Vectorized first-order optimization algorithms for parameter updates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    @abstractmethod
    def get_lr(self, epoch: int, base_lr: float) -> float:
        """Calculate learning rate at a given epoch."""
        pass


class ConstantLR(LRScheduler):
    """Constant learning rate schedule."""

    def get_lr(self, epoch: int, base_lr: float) -> float:
        return base_lr


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs."""

    def __init__(self, step_size: int = 30, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int, base_lr: float) -> float:
        return base_lr * (self.gamma ** (epoch // self.step_size))


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule."""

    def __init__(self, t_max: int = 100, eta_min: float = 0.0):
        self.t_max = t_max
        self.eta_min = eta_min

    def get_lr(self, epoch: int, base_lr: float) -> float:
        epoch = min(epoch, self.t_max)
        return self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + np.cos(np.pi * epoch / self.t_max))


class Optimizer(ABC):
    """Abstract base class for numerical optimizers."""

    def __init__(self, lr: float = 0.01, scheduler: Optional[LRScheduler] = None):
        self.lr = lr
        self.scheduler = scheduler or ConstantLR()
        self.t = 0

    def step_epoch(self) -> None:
        """Signal completion of an epoch to update scheduler."""
        self.t += 1

    def current_lr(self) -> float:
        """Return the current scheduled learning rate."""
        return self.scheduler.get_lr(self.t, self.lr)

    @abstractmethod
    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        """Compute the updated parameter array given its gradient."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal accumulator states."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional Momentum, Nesterov, and Weight Decay.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    momentum : float, default=0.0
        Momentum factor (0.0 means vanilla SGD).
    nesterov : bool, default=False
        Whether to enable Nesterov accelerated gradient.
    weight_decay : float, default=0.0
        L2 regularization coefficient applied to parameters during gradient step.
    scheduler : LRScheduler, optional
        Learning rate schedule.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(lr=lr, scheduler=scheduler)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocities: dict[str, np.ndarray] = {}

    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        g = grad.copy()
        if self.weight_decay > 0.0:
            g += self.weight_decay * param

        lr = self.current_lr()

        if self.momentum > 0.0:
            if param_id not in self.velocities:
                self.velocities[param_id] = np.zeros_like(param)
            v = self.velocities[param_id]
            v = self.momentum * v + g
            self.velocities[param_id] = v

            if self.nesterov:
                step = self.momentum * v + g
            else:
                step = v
            return param - lr * step

        return param - lr * g

    def reset(self) -> None:
        self.velocities.clear()
        self.t = 0


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Maintains exponentially decaying averages of past gradients (first moment m_t)
    and past squared gradients (second uncentered moment v_t).

    Parameters
    ----------
    lr : float, default=0.001
        Learning rate (alpha).
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimates.
    eps : float, default=1e-8
        Small constant for numerical stability to prevent division by zero.
    weight_decay : float, default=0.0
        L2 weight decay penalty.
    scheduler : LRScheduler, optional
        Learning rate schedule.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(lr=lr, scheduler=scheduler)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.step_counts: dict[str, int] = {}

    def update(self, param: np.ndarray, grad: np.ndarray, param_id: str) -> np.ndarray:
        g = grad.copy()
        if self.weight_decay > 0.0:
            g += self.weight_decay * param

        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)
            self.step_counts[param_id] = 0

        self.step_counts[param_id] += 1
        t = self.step_counts[param_id]

        m = self.m[param_id]
        v = self.v[param_id]

        # Update biased first and second moments
        m = self.beta1 * m + (1.0 - self.beta1) * g
        v = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

        self.m[param_id] = m
        self.v[param_id] = v

        # Bias corrections
        m_hat = m / (1.0 - (self.beta1 ** t))
        v_hat = v / (1.0 - (self.beta2 ** t))

        lr = self.current_lr()
        param_new = param - lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param_new

    def reset(self) -> None:
        self.m.clear()
        self.v.clear()
        self.step_counts.clear()
        self.t = 0
