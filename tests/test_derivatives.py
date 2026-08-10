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


def test_kij_polynomial_jacobian_formula():
    """The weighted Vandermonde must equal the numerical Jacobian of _poly_resid.

    Guards column order (increasing powers of dT) and orientation. Getting
    either wrong silently produces a wrong k_ij(T) polynomial.
    """
    from scipy.optimize._numdiff import approx_derivative

    t_ref = 298.15
    T = np.linspace(280.0, 360.0, 7)
    dT = T - t_ref
    kij_arr = -0.048 + 3.0e-4 * dT
    w_sqrt = np.linspace(0.4, 1.0, 7)
    order = 2

    def poly_resid(coeffs):
        pred = sum(c * dT**j for j, c in enumerate(coeffs))
        return w_sqrt * (pred - kij_arr)

    analytic = w_sqrt[:, None] * np.vander(dT, order + 1, increasing=True)
    x = np.array([-0.05, 2.0e-4, 1.0e-7])

    np.testing.assert_allclose(
        analytic, approx_derivative(poly_resid, x), rtol=1e-5, atol=1e-9
    )


def test_kij_polynomial_recovers_trend_despite_outlier():
    """Equivalence guard: the Cauchy polish must still reject a gross outlier."""
    from fit_pcsaft._binary._utils import _fit_kij_polynomial

    t_ref = 298.15
    T = np.linspace(280.0, 360.0, 9)
    true_c = np.array([-0.048, 3.0e-4])
    kij = true_c[0] + true_c[1] * (T - t_ref)
    kij[4] += 0.25                      # gross outlier
    ard = np.full(len(T), 1.0)
    ard[4] = 50.0                       # and it is flagged as unreliable

    coeffs, _ = _fit_kij_polynomial(T, kij, ard, kij_order=1, kij_t_ref=t_ref)

    np.testing.assert_allclose(coeffs, true_c, rtol=5e-2, atol=1e-4)


def test_make_core_caches_compute_between_fun_and_jac(monkeypatch):
    """scipy calls fun(x) then jac(x) at the same x; that must cost one AD call."""
    import feos

    import si_units as si

    from fit_pcsaft._pure.jacobian import _make_core
    from fit_pcsaft._types import Compound, FitConfig, PureData, Units

    calls = {"n": 0}
    real = feos.Property.vapor_pressure_derivatives

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(feos.Property, "vapor_pressure_derivatives", counting)

    data = PureData(
        T_psat=np.array([300.0, 320.0, 340.0]),
        p_psat=np.array([8.86, 30.0, 70.0]),
        T_rho=np.array([]), rho=np.array([]),
        T_hvap=np.array([]), hvap=np.array([]),
    )
    compound = Compound(identifier=feos.Identifier(cas="64-17-5"), mw=46.07)
    config = FitConfig(
        w_psat=3.0, w_rho=2.0, w_hvap=1.0, extrapolate_psat=False,
        loss="linear", f_scale={"psat": 1.0, "rho": 1.0},
    )
    units = Units(
        temperature=si.KELVIN,
        pressure=si.KILO * si.PASCAL,
        density=si.KILOGRAM / si.METER**3,
        enthalpy=si.KILO * si.JOULE / si.MOL,
    )

    fun, jac = _make_core(
        data, compound, units, config,
        feos.EquationOfStateAD.PcSaftNonAssoc,
        ["m", "sigma", "epsilon_k"],
        lambda p: np.array([p[0], p[1], p[2], 0.0]),
    )

    x = np.sqrt(np.array([2.3827, 3.1771, 198.24]))
    f = fun(x)
    J = jac(x)

    assert calls["n"] == 1, f"expected 1 feos AD call, got {calls['n']}"
    assert f.shape == (3,)
    assert J.shape == (3, 3)


# --------------------------------------------------------------------------
# feos AD contract tests
#
# These document feos behaviour, not ours. They exist so that a feos upgrade
# which changes any of these assumptions fails loudly instead of silently
# corrupting a Jacobian.
# --------------------------------------------------------------------------

_ETHANOL = dict(m=2.3827, sigma=3.1771, epsilon_k=198.24,
                kappa_ab=0.032384, epsilon_k_ab=2653.4)
_ETHANOL_ROW = [_ETHANOL["m"], _ETHANOL["sigma"], _ETHANOL["epsilon_k"], 0.0,
                _ETHANOL["kappa_ab"], _ETHANOL["epsilon_k_ab"], 1.0, 1.0]
_ASSOC_NAMES = ["m", "sigma", "epsilon_k", "kappa_ab", "epsilon_k_ab"]


def test_feos_exposes_hvap_derivatives():
    """Contract check: the AD call feos gives us, in the units it gives us."""
    import feos

    vals, grad, conv = feos.Property.enthalpy_of_vaporization_derivatives(
        feos.EquationOfStateAD.PcSaftFull, _ASSOC_NAMES,
        np.array(_ETHANOL_ROW), np.array([[350.0]]),
    )

    assert bool(conv[0])
    assert grad.shape == (1, 5)
    # J/mol, not kJ/mol
    assert 3.0e4 < float(vals[0]) < 5.0e4


def test_hvap_ad_matches_finite_differences():
    """AD gradient must match a converged central difference."""
    import feos

    col = {"m": 0, "sigma": 1, "epsilon_k": 2, "kappa_ab": 4, "epsilon_k_ab": 5}

    def hvap(row):
        v, _, _ = feos.Property.enthalpy_of_vaporization_derivatives(
            feos.EquationOfStateAD.PcSaftFull, _ASSOC_NAMES,
            np.array(row), np.array([[350.0]]),
        )
        return float(v[0])

    _, grad, _ = feos.Property.enthalpy_of_vaporization_derivatives(
        feos.EquationOfStateAD.PcSaftFull, _ASSOC_NAMES,
        np.array(_ETHANOL_ROW), np.array([[350.0]]),
    )

    for k, name in enumerate(_ASSOC_NAMES):
        if name == "sigma":
            continue  # structurally zero, see test_hvap_is_invariant_in_sigma
        i = col[name]
        h = abs(_ETHANOL_ROW[i]) * 1e-6
        up, dn = list(_ETHANOL_ROW), list(_ETHANOL_ROW)
        up[i] += h
        dn[i] -= h
        fd = (hvap(up) - hvap(dn)) / (2 * h)
        np.testing.assert_allclose(grad[0][k], fd, rtol=1e-6)


def test_hvap_is_invariant_in_sigma():
    """Documents a real property, not a bug.

    sigma sets only the length scale, so molar residual enthalpy at VLE is
    invariant under corresponding-states scaling. psat moves, Hvap does not.
    A future change that makes this column nonzero is a regression.
    """
    import feos

    shifted = list(_ETHANOL_ROW)
    shifted[1] += 0.05

    def hvap(row):
        v, _, _ = feos.Property.enthalpy_of_vaporization_derivatives(
            feos.EquationOfStateAD.PcSaftFull, ["m"],
            np.array(row), np.array([[350.0]]),
        )
        return float(v[0])

    np.testing.assert_allclose(hvap(_ETHANOL_ROW), hvap(shifted), rtol=1e-12)


def test_quadrupole_is_not_differentiable():
    """Guard the constraint that keeps the q != 0 numerical fallback alive."""
    import feos

    _, grad, _ = feos.Property.vapor_pressure_derivatives(
        feos.EquationOfStateAD.PcSaftFull, ["q"],
        np.array(_ETHANOL_ROW), np.array([[350.0]]),
    )

    assert float(grad[0][0]) == 0.0, "if q became differentiable, drop the fallback"


# --------------------------------------------------------------------------
# End-to-end: the AD path against the numerical path it replaces
# --------------------------------------------------------------------------


def _ethanol_compound():
    import feos

    from fit_pcsaft._types import Compound

    return Compound(identifier=feos.Identifier(cas="64-17-5"), mw=46.07)


def _units(pressure=None):
    import si_units as si

    from fit_pcsaft._types import Units

    return Units(
        temperature=si.KELVIN,
        pressure=pressure if pressure is not None else si.KILO * si.PASCAL,
        density=si.KILOGRAM / si.METER**3,
        enthalpy=si.KILO * si.JOULE / si.MOL,
    )


_X_ETHANOL = np.sqrt(np.array([
    _ETHANOL["m"], _ETHANOL["sigma"], _ETHANOL["epsilon_k"],
    _ETHANOL["kappa_ab"], _ETHANOL["epsilon_k_ab"],
]))


def test_analytical_and_numerical_jacobians_agree_with_hvap():
    """The new AD hvap path must match the numerical path it replaces."""
    from fit_pcsaft._fit_utils import _make_f_and_df_numerical
    from fit_pcsaft._pure.jacobian import _make_f_and_df
    from fit_pcsaft._types import FitConfig, ModelSpec, PureData

    data = PureData(
        T_psat=np.array([300.0, 330.0, 360.0]),
        p_psat=np.array([8.86, 30.0, 90.0]),
        T_rho=np.array([300.0, 330.0]),
        rho=np.array([780.0, 750.0]),
        T_hvap=np.array([300.0, 350.0]),
        hvap=np.array([42.3, 38.4]),
    )
    spec = ModelSpec(mu=0.0, na=1, nb=1, q=0.0)
    config = FitConfig(
        w_psat=3.0, w_rho=2.0, w_hvap=1.0, extrapolate_psat=False,
        loss="linear", f_scale={"psat": 1.0, "rho": 1.0, "hvap": 1.0},
    )

    compound, units = _ethanol_compound(), _units()
    f_ad, J_ad, _ = _make_f_and_df(data, compound, spec, units, config)
    f_num, J_num, _ = _make_f_and_df_numerical(data, compound, spec, units, config)

    np.testing.assert_allclose(f_ad(_X_ETHANOL), f_num(_X_ETHANOL), rtol=1e-8)
    # Forward differences on an iterative solver: loose but decisive.
    np.testing.assert_allclose(
        J_ad(_X_ETHANOL), J_num(_X_ETHANOL), rtol=2e-3, atol=1e-6
    )


def test_analytical_path_respects_pressure_units():
    """bar vs kPa must change the residuals by exactly the unit ratio."""
    import si_units as si

    from fit_pcsaft._pure.jacobian import _make_f_and_df
    from fit_pcsaft._types import FitConfig, ModelSpec, PureData

    T = np.array([300.0, 330.0])
    p_kpa = np.array([8.86, 30.0])

    def build(p_unit, p_vals):
        data = PureData(
            T_psat=T, p_psat=p_vals,
            T_rho=np.array([]), rho=np.array([]),
            T_hvap=np.array([]), hvap=np.array([]),
        )
        config = FitConfig(
            w_psat=3.0, w_rho=2.0, w_hvap=1.0, extrapolate_psat=False,
            loss="linear", f_scale={"psat": 1.0, "rho": 1.0},
        )
        return _make_f_and_df(
            data, _ethanol_compound(),
            ModelSpec(mu=0.0, na=1, nb=1, q=0.0),
            _units(pressure=p_unit), config,
        )

    f_kpa, _, _ = build(si.KILO * si.PASCAL, p_kpa)
    f_bar, _, _ = build(si.BAR, p_kpa / 100.0)

    # Same physical data expressed in two units -> identical relative residuals.
    np.testing.assert_allclose(f_kpa(_X_ETHANOL), f_bar(_X_ETHANOL), rtol=1e-9)
