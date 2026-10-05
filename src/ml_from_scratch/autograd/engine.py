"""Minimal reverse-mode scalar automatic differentiation engine."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Union


class Value:
    """A scalar value that tracks operations and automatically computes gradients.

    Attributes
    ----------
    data : float
        Scalar numerical value.
    grad : float
        Accumulated gradient d(Loss) / d(self).
    _backward : Callable[[], None]
        Closure applying the local chain rule for the operation that created this Value.
    _prev : set[Value]
        Parent operand nodes in the computational graph.
    _op : str
        String description of the generating operation (e.g. '+', '*', 'relu').
    """

    def __init__(
        self,
        data: float | int,
        _children: Iterable[Value] = (),
        _op: str = "",
    ):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Value, float, int]) -> Value:
        return self + other

    def __mul__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Value, float, int]) -> Value:
        return self * other

    def __pow__(self, other: Union[int, float]) -> Value:
        if not isinstance(other, (int, float)):
            raise TypeError("Only float/int powers are supported.")
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Union[Value, float, int]) -> Value:
        return self + (-other)

    def __rsub__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __truediv__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def exp(self) -> Value:
        """Exponential function e^x."""
        x = max(min(self.data, 700.0), -700.0)  # Bound to avoid float overflow
        val = math.exp(x)
        out = Value(val, (self,), "exp")

        def _backward() -> None:
            self.grad += val * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Natural logarithm ln(x)."""
        if self.data <= 0.0:
            raise ValueError(f"Cannot take logarithm of non-positive value {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        """Rectified Linear Unit activation."""
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """Numerically stable Sigmoid activation 1 / (1 + e^-x)."""
        if self.data >= 0:
            s = 1.0 / (1.0 + math.exp(-self.data))
        else:
            exp_x = math.exp(self.data)
            s = exp_x / (1.0 + exp_x)
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Hyperbolic tangent activation."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Compute reverse-mode automatic differentiation backpropagation over the computation graph."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
