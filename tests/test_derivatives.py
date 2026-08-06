"""Accuracy tests for the hand-rolled finite-difference Jacobians.

These guard step-size regressions. A central difference has two competing
error terms: truncation ~ h^2 * f''' and cancellation ~ eps/h. The optimum
for a clean function is h ~ eps^(1/3) ~ 6e-6; for a function computed by an
iterative solver the floor is set by the solver tolerance, not by eps, so
anything below ~1e-8 is noise.
"""
import numpy as np

from fit_pcsaft._binary._utils import _make_binary_jac_fn


def test_binary_jac_matches_exact_derivative_smooth_fn():
    """On an analytic function the FD Jacobian must be accurate to ~1e-8."""
    def fun(x):
        return np.array([np.sin(3.0 * x[0]) + x[0] ** 2, np.exp(0.5 * x[0])])

    x = np.array([0.05])
    exact = np.array([[3.0 * np.cos(3.0 * 0.05) + 2.0 * 0.05],
                      [0.5 * np.exp(0.5 * 0.05)]])

    J = _make_binary_jac_fn(fun, 1)(x)

    assert J.shape == exact.shape
    np.testing.assert_allclose(J, exact, rtol=1e-8)


def test_binary_jac_multi_param_shape_and_values():
    """Two-parameter case: columns must map to the right parameter."""
    def fun(x):
        return np.array([2.0 * x[0] + 5.0 * x[1], x[0] * x[1]])

    x = np.array([0.3, 0.7])
    exact = np.array([[2.0, 5.0], [0.7, 0.3]])

    J = _make_binary_jac_fn(fun, 2)(x)

    assert J.shape == (2, 2)
    np.testing.assert_allclose(J, exact, rtol=1e-7)


def test_default_step_is_above_the_noise_floor():
    """Regression guard on the literal default.

    h=1e-12 was the historical value and gave ~69% error on a real
    bubble-point residual. Anything below 1e-9 is unusable.
    """
    import inspect

    default_h = inspect.signature(_make_binary_jac_fn).parameters["h"].default
    assert 1e-9 <= default_h <= 1e-3
