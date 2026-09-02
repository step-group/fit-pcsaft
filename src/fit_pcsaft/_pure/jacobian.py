"""Combined cost + Jacobian for pure component PC-SAFT fitting.

feos.Property.{vapor_pressure,equilibrium_liquid_density,enthalpy_of_vaporization}
_derivatives each return (values, jacobian, mask) in a single vectorized AD call.

Structure
---------
_make_f_and_df derives the feos AD model, the differentiable parameter names
and the AD row from one place -- ``_fit_utils.ad_model`` / ``param_names`` /
``ad_rows`` -- and hands them to _make_core, which builds the shared
_compute/fun/jac closures. fun and jac cache the result of _compute so scipy's
sequential fun(x) → jac(x) calls per optimizer step trigger only one feos
evaluation. feos accepts a shared 1-D row for every input point, so no
per-property tiling is needed.

feos AD returns SI (Pa, kmol/m³, J/mol); ``units`` converts to the user's CSV
units. Quadrupole (q) has no AD support — feos silently returns a zero gradient
column for 'q' — so q != 0 stays on the numerical path in fit.py.
"""

import feos
import numpy as np
import si_units as si

from fit_pcsaft._fit_utils import _PENALTY, ad_model, ad_rows, param_names
from fit_pcsaft._types import Compound, FitConfig, ModelSpec, PureData, Units

# ---------------------------------------------------------------------------
# Shared core — _compute / fun / jac closures
# ---------------------------------------------------------------------------


def _make_core(
    data: PureData,
    compound: Compound,
    units: Units,
    config: FitConfig,
    eos_ad,
    fit_params,
    make_row,
    errors: "list | None" = None,
):
    """Build (fun, jac) given the case-specific eos_ad, fit_params, make_row.

    On failure this returns a *zero* Jacobian, which flattens the gradient and
    makes the optimizer stop immediately reporting success. ``errors`` records
    why, so the caller's stall guard can explain the bogus "convergence".
    """
    T_psat, d_psat = data.T_psat, data.p_psat
    T_rho, d_rho = data.T_rho, data.rho
    T_hvap, d_hvap = data.T_hvap, data.hvap
    mw = compound.mw

    n_psat, n_rho, n_hvap = len(T_psat), len(T_rho), len(T_hvap)
    n_total = n_psat + n_rho + n_hvap
    n_params = len(fit_params)

    # feos AD returns SI: Pa, kmol/m^3, J/mol. Convert to the user's CSV units.
    # Previously hardcoded to kPa and kg/m^3, which silently corrupted both the
    # residuals and the Jacobian for any other unit.
    to_p = 1.0 / (units.pressure / si.PASCAL)
    to_rho = mw / (units.density / (si.KILOGRAM / si.METER**3))
    to_h = 1.0 / (units.enthalpy / (si.JOULE / si.MOL))

    inv_T_psat = 1.0 / T_psat

    def _scales(w, n, key, unit_factor, d):
        if n == 0:
            return 0.0, np.zeros(1)
        s = np.sqrt(w / n) / config.f_scale[key]
        return s, (np.sqrt(w / n) * unit_factor) / d / config.f_scale[key]

    psat_cost_scale, psat_jac_scale = _scales(
        config.w_psat, n_psat, "psat", to_p, d_psat)
    rho_cost_scale, rho_jac_scale = _scales(
        config.w_rho, n_rho, "rho", to_rho, d_rho)
    hvap_cost_scale, hvap_jac_scale = _scales(
        config.w_hvap, n_hvap, "hvap", to_h, d_hvap)

    in_psat = np.expand_dims(np.asarray(T_psat, dtype=float), 1)
    in_rho = np.expand_dims(np.asarray(T_rho, dtype=float), 1) if n_rho else None
    in_hvap = np.expand_dims(np.asarray(T_hvap, dtype=float), 1) if n_hvap else None

    _penalty_f = np.full(n_total, _PENALTY)
    _penalty_J = np.zeros((n_total, n_params))
    _FAIL = (_penalty_f, _penalty_J, None, None)

    def _block(ad_fn, row, inp, unit_factor, cost_scale, jac_scale, d, chain):
        """Return (residuals, jacobian, predictions) or None on failure."""
        try:
            vals, grad, mask = ad_fn(eos_ad, fit_params, row, inp)
        except Exception as exc:
            if errors is not None:
                errors.append(exc)
            return None
        if not mask.all():
            return None
        pred = vals * unit_factor
        f = cost_scale * (pred / d - 1.0)
        J = grad * jac_scale[:, np.newaxis] * chain[np.newaxis, :]
        return f, J, pred

    def _compute(params_vec):
        p = params_vec**2
        chain = 2.0 * params_vec
        row = np.ascontiguousarray(make_row(p))

        # --- vapor pressure: the only block allowed to extrapolate ---
        try:
            vals, grad, mask = feos.Property.vapor_pressure_derivatives(
                eos_ad, fit_params, row, in_psat
            )
        except Exception as exc:
            if errors is not None:
                errors.append(exc)
            return _FAIL

        p_pred = np.full(n_psat, np.nan)
        p_pred[mask] = vals[mask] * to_p

        if not mask.all():
            if config.extrapolate_psat and mask.sum() >= 2:
                X = np.column_stack([np.ones(mask.sum()), inv_T_psat[mask]])
                coeffs = np.linalg.lstsq(X, np.log(p_pred[mask]), rcond=None)[0]
                p_pred[~mask] = np.exp(coeffs[0] + coeffs[1] * inv_T_psat[~mask])
            else:
                return _FAIL

        f_parts = [psat_cost_scale * (p_pred / d_psat - 1.0)]
        jac_psat = np.zeros((n_psat, n_params))
        jac_psat[mask] = grad[mask]
        J_parts = [jac_psat * psat_jac_scale[:, np.newaxis] * chain[np.newaxis, :]]

        rho_pred = None
        if n_rho:
            got = _block(
                feos.Property.equilibrium_liquid_density_derivatives,
                row, in_rho, to_rho, rho_cost_scale, rho_jac_scale,
                d_rho, chain,
            )
            if got is None:
                return _FAIL
            f_rho, J_rho, rho_pred = got
            f_parts.append(f_rho)
            J_parts.append(J_rho)

        if n_hvap:
            got = _block(
                feos.Property.enthalpy_of_vaporization_derivatives,
                row, in_hvap, to_h, hvap_cost_scale, hvap_jac_scale,
                d_hvap, chain,
            )
            if got is None:
                return _FAIL
            f_hvap, J_hvap, _ = got
            f_parts.append(f_hvap)
            J_parts.append(J_hvap)

        return (
            np.concatenate(f_parts) if len(f_parts) > 1 else f_parts[0],
            np.vstack(J_parts) if len(J_parts) > 1 else J_parts[0],
            p_pred,
            rho_pred,
        )

    # scipy calls fun(x) then jac(x) at the same x on every iteration.
    # Caching the last (x, result) halves the feos AD evaluations.
    _cache_x = None
    _cache_out = None

    def _cached(params_vec):
        nonlocal _cache_x, _cache_out
        if _cache_x is not None and np.array_equal(params_vec, _cache_x):
            return _cache_out
        _cache_out = _compute(params_vec)
        _cache_x = np.array(params_vec, copy=True)
        return _cache_out

    def fun(params_vec):
        return _cached(params_vec)[0]

    def jac(params_vec):
        return _cached(params_vec)[1]

    return fun, jac


def _make_f_and_df(
    data: PureData,
    compound: Compound,
    spec: ModelSpec,
    units: Units,
    config: FitConfig,
):
    """Return (fun, jac, errors) for the appropriate PC-SAFT case."""
    if len(data.T_sft) > 0:
        raise ValueError(
            "The analytical Jacobian does not support surface tension — feos exposes "
            "no AD path for it (feos.Property has no surface_tension_derivatives). "
            "Use the numerical Jacobian."
        )

    errors: list = []
    fun, jac = _make_core(
        data, compound, units, config,
        ad_model(spec), param_names(spec), lambda p: ad_rows(p, spec)[0], errors,
    )
    return fun, jac, errors
