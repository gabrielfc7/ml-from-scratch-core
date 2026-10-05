"""Tests for scalar autograd engine and neural network foundations."""

import math
import random
import pytest
from ml_from_scratch.autograd import MLP, Value, mse_loss


def test_autograd_basic_arithmetic_and_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f
    L.backward()

    # Analytical gradients:
    # L = f * (a*b + c)
    # dL/df = d = 2 * (-3) + 10 = 4.0
    # dL/dd = f = -2.0
    # dL/de = dL/dd * 1 = -2.0
    # dL/dc = dL/dd * 1 = -2.0
    # dL/da = dL/de * b = -2.0 * -3.0 = 6.0
    # dL/db = dL/de * a = -2.0 * 2.0 = -4.0

    assert pytest.approx(L.data) == -8.0
    assert pytest.approx(f.grad) == 4.0
    assert pytest.approx(c.grad) == -2.0
    assert pytest.approx(a.grad) == 6.0
    assert pytest.approx(b.grad) == -4.0


def test_autograd_numerical_gradient_check():
    """Verify reverse-mode gradients against numerical two-sided finite differences."""
    eps = 1e-6

    def func(x_val, y_val):
        x = Value(x_val)
        y = Value(y_val)
        out = (x * 2.0 + y.relu()) ** 2 + x.sigmoid() + (x / y)
        out.backward()
        return out.data, x.grad, y.grad

    x0, y0 = 1.5, 2.5
    f0, gx_auto, gy_auto = func(x0, y0)

    # Numerical grad for x
    f_x_plus, _, _ = func(x0 + eps, y0)
    f_x_minus, _, _ = func(x0 - eps, y0)
    gx_num = (f_x_plus - f_x_minus) / (2.0 * eps)

    # Numerical grad for y
    f_y_plus, _, _ = func(x0, y0 + eps)
    f_y_minus, _, _ = func(x0, y0 - eps)
    gy_num = (f_y_plus - f_y_minus) / (2.0 * eps)

    assert pytest.approx(gx_auto, abs=1e-4) == gx_num
    assert pytest.approx(gy_auto, abs=1e-4) == gy_num


def test_autograd_activations():
    x = Value(0.5)
    # Test tanh
    t = x.tanh()
    t.backward()
    # d/dx tanh(x) = 1 - tanh(x)^2
    expected_tanh_grad = 1.0 - math.tanh(0.5) ** 2
    assert pytest.approx(x.grad) == expected_tanh_grad

    # Test exp and log
    a = Value(2.0)
    b = a.exp()
    c = b.log()
    c.backward()
    # c = ln(exp(a)) = a -> dc/da = 1.0
    assert pytest.approx(a.grad) == 1.0


def test_mlp_xor_convergence():
    """Train an MLP on XOR dataset from scratch to verify end-to-end backprop."""
    random.seed(42)

    # XOR truth table
    xs = [
        [2.0, 2.0],  # 0
        [2.0, -2.0], # 1
        [-2.0, 2.0], # 1
        [-2.0, -2.0],# 0
    ]
    ys = [0.0, 1.0, 1.0, 0.0]

    # 2 inputs -> hidden layer of 4 neurons (tanh) -> 1 output (sigmoid)
    mlp = MLP(2, [4, 1], activations=["tanh", "sigmoid"])

    learning_rate = 0.5
    for epoch in range(150):
        # Forward pass
        ypred = [mlp(x) for x in xs]
        loss = mse_loss(ypred, ys)

        # Backward pass
        mlp.zero_grad()
        loss.backward()

        # SGD step
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad

    # Verify predictions after training
    final_preds = [mlp(x).data for x in xs]
    binary_preds = [1.0 if p >= 0.5 else 0.0 for p in final_preds]
    assert binary_preds == ys
