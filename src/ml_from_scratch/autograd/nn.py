"""Neural network modules, layers, and MLP constructed on the scalar autograd engine."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Sequence, Union

from ml_from_scratch.autograd.engine import Value


class Module(ABC):
    """Base neural network module."""

    def zero_grad(self) -> None:
        """Reset gradients for all parameters to 0.0."""
        for p in self.parameters():
            p.grad = 0.0

    @abstractmethod
    def parameters(self) -> list[Value]:
        """Return list of trainable parameter Value objects."""
        pass


class Neuron(Module):
    """A single artificial neuron with configurable activation function.

    Parameters
    ----------
    nin : int
        Number of input connections.
    act : {'relu', 'sigmoid', 'tanh', 'linear'}, default='relu'
        Activation function applied to the affine transformation.
    """

    def __init__(self, nin: int, act: str = "relu"):
        # Xavier / Glorot uniform initialization: bound = sqrt(6 / (nin + 1))
        bound = (6.0 / (nin + 1.0)) ** 0.5
        self.w = [Value(random.uniform(-bound, bound)) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x: Sequence[Union[Value, float]]) -> Value:
        # Affine combination: sum(w_i * x_i) + b
        act_val = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.act == "relu":
            return act_val.relu()
        elif self.act == "sigmoid":
            return act_val.sigmoid()
        elif self.act == "tanh":
            return act_val.tanh()
        elif self.act == "linear":
            return act_val
        else:
            raise ValueError(f"Unknown activation function '{self.act}'")

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    """A dense / fully connected feedforward layer of neurons.

    Parameters
    ----------
    nin : int
        Number of input dimensions.
    nout : int
        Number of output neurons.
    act : {'relu', 'sigmoid', 'tanh', 'linear'}, default='relu'
        Activation applied by neurons in this layer.
    """

    def __init__(self, nin: int, nout: int, act: str = "relu"):
        self.neurons = [Neuron(nin, act=act) for _ in range(nout)]

    def __call__(self, x: Sequence[Union[Value, float]]) -> Union[Value, list[Value]]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class MLP(Module):
    """Multi-Layer Perceptron (feedforward neural network).

    Parameters
    ----------
    nin : int
        Input dimensionality.
    nouts : list of int
        List of layer output sizes, including hidden layers and output layer.
    activations : list of str, optional
        Activation functions for each layer. If not provided, hidden layers use 'relu' and output uses 'linear'.
    """

    def __init__(
        self,
        nin: int,
        nouts: list[int],
        activations: Union[str, list[str], None] = None,
    ):
        sz = [nin] + nouts
        n_layers = len(nouts)

        if activations is None:
            acts = ["relu"] * (n_layers - 1) + ["linear"]
        elif isinstance(activations, str):
            acts = [activations] * n_layers
        else:
            acts = list(activations)

        self.layers = [Layer(sz[i], sz[i + 1], act=acts[i]) for i in range(n_layers)]

    def __call__(self, x: Sequence[Union[Value, float]]) -> Union[Value, list[Value]]:
        out = x
        for layer in self.layers:
            out = layer(out) if isinstance(out, list) else layer([out])
        return out

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def mse_loss(y_preds: Sequence[Value], y_trues: Sequence[Union[float, int]]) -> Value:
    """Mean squared error loss over a collection of Value predictions."""
    total = sum(((yp - yt) ** 2 for yp, yt in zip(y_preds, y_trues)), Value(0.0))
    return total * (1.0 / len(y_preds))


def binary_cross_entropy_loss(
    y_preds: Sequence[Value],
    y_trues: Sequence[Union[float, int]],
    eps: float = 1e-12,
) -> Value:
    """Binary cross-entropy loss over a collection of Value probability predictions."""
    total = Value(0.0)
    for yp, yt in zip(y_preds, y_trues):
        # Loss = - [y * log(p) + (1 - y) * log(1 - p)]
        p_clipped = yp + eps
        one_minus_p = (1.0 - yp) + eps
        loss_i = -1.0 * (yt * p_clipped.log() + (1.0 - yt) * one_minus_p.log())
        total = total + loss_i
    return total * (1.0 / len(y_preds))
