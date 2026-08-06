"""Combined cost + Jacobian for pure component PC-SAFT fitting.

feos.Property.vapor_pressure_derivatives / equilibrium_liquid_density_derivatives
return (values, jacobian, mask) in a single vectorized AD call.

Structure
---------
4 private setup functions handle the case-specific differences:
  _setup_nonassoc      — non-associating, mu fixed
  _setup_nonassoc_mu   — non-associating, mu fitted
  _setup_assoc         — associating, mu fixed
  _setup_assoc_mu      — associating, mu fitted

Each returns (eos_ad, fit_params, build_arrays) where build_arrays(p) → (pa_psat, pa_rho).

_make_core takes those three plus data/compound/config and builds the shared
_compute/fun/jac closures. fun and jac cache the result of _compute so scipy's
sequential fun(x) → jac(x) calls per optimizer step trigger only one feos evaluation.

make_f_and_df dispatches to the right setup function and calls _make_core.

Unit assumptions (hardcoded to match feos AD output):
  psat: derivs[0] in Pa → /1000 → kPa
  density: derivs[0] in kmol/m³ → ×mw → kg/m³
"""

import feos
import numpy as np

from fit_pcsaft._fit_utils import _PENALTY
from fit_pcsaft._types import Compound, FitConfig, ModelSpec, PureData, Units

# ---------------------------------------------------------------------------
# Setup functions — return (eos_ad, fit_params, build_arrays)
# ---------------------------------------------------------------------------


def _setup_nonassoc(spec: ModelSpec, n_psat: int, n_rho: int):
    """Non-associating, mu fixed."""
    mu = spec.mu
    mu_col_psat = np.full((n_psat, 1), mu)
    mu_col_rho = np.full((n_rho, 1), mu) if n_rho > 0 else None

    def build_arrays(p):
        pa_psat = np.column_stack([np.tile(p, (n_psat, 1)), mu_col_psat])
        pa_rho = (
            np.column_stack([np.tile(p, (n_rho, 1)), mu_col_rho]) if n_rho > 0 else None
        )
        return pa_psat, pa_rho

    return (
        feos.EquationOfStateAD.PcSaftNonAssoc,
        ["m", "sigma", "epsilon_k"],
        build_arrays,
    )


def _setup_nonassoc_mu(spec: ModelSpec, n_psat: int, n_rho: int):
    """Non-associating, mu fitted."""

    def build_arrays(p):
        pa_psat = np.tile(p, (n_psat, 1))
        pa_rho = np.tile(p, (n_rho, 1)) if n_rho > 0 else None
        return pa_psat, pa_rho

    return (
        feos.EquationOfStateAD.PcSaftNonAssoc,
        ["m", "sigma", "epsilon_k", "mu"],
        build_arrays,
    )


def _setup_assoc(spec: ModelSpec, n_psat: int, n_rho: int):
    """Associating, mu fixed."""
    mu = spec.mu
    na, nb = spec.na, spec.nb
    mu_col_psat = np.full((n_psat, 1), mu)
    mu_col_rho = np.full((n_rho, 1), mu) if n_rho > 0 else None
    na_col_psat = np.full((n_psat, 1), float(na))
    nb_col_psat = np.full((n_psat, 1), float(nb))
    na_col_rho = np.full((n_rho, 1), float(na)) if n_rho > 0 else None
    nb_col_rho = np.full((n_rho, 1), float(nb)) if n_rho > 0 else None

    def build_arrays(p):
        pa_psat = np.column_stack(
            [
                np.tile(p[:3], (n_psat, 1)),
                mu_col_psat,
                np.tile(p[3:5], (n_psat, 1)),
                na_col_psat,
                nb_col_psat,
            ]
        )
        pa_rho = (
            np.column_stack(
                [
                    np.tile(p[:3], (n_rho, 1)),
                    mu_col_rho,
                    np.tile(p[3:5], (n_rho, 1)),
                    na_col_rho,
                    nb_col_rho,
                ]
            )
            if n_rho > 0
            else None
        )
        return pa_psat, pa_rho

    return (
        feos.EquationOfStateAD.PcSaftFull,
        ["m", "sigma", "epsilon_k", "kappa_ab", "epsilon_k_ab"],
        build_arrays,
    )


def _setup_assoc_mu(spec: ModelSpec, n_psat: int, n_rho: int):
    """Associating, mu fitted."""
    na, nb = spec.na, spec.nb
    na_col_psat = np.full((n_psat, 1), float(na))
    nb_col_psat = np.full((n_psat, 1), float(nb))
    na_col_rho = np.full((n_rho, 1), float(na)) if n_rho > 0 else None
    nb_col_rho = np.full((n_rho, 1), float(nb)) if n_rho > 0 else None

    def build_arrays(p):
        pa_psat = np.column_stack(
            [
                np.tile(p[:3], (n_psat, 1)),
                np.tile(p[3:4], (n_psat, 1)),
                np.tile(p[4:6], (n_psat, 1)),
                na_col_psat,
                nb_col_psat,
            ]
        )
        pa_rho = (
            np.column_stack(
                [
                    np.tile(p[:3], (n_rho, 1)),
                    np.tile(p[3:4], (n_rho, 1)),
                    np.tile(p[4:6], (n_rho, 1)),
                    na_col_rho,
                    nb_col_rho,
                ]
            )
            if n_rho > 0
            else None
        )
        return pa_psat, pa_rho

    return (
        feos.EquationOfStateAD.PcSaftFull,
        ["m", "sigma", "epsilon_k", "mu", "kappa_ab", "epsilon_k_ab"],
        build_arrays,
    )


# ---------------------------------------------------------------------------
# Shared core — _compute / fun / jac closures
# ---------------------------------------------------------------------------


def _make_core(
    data: PureData,
    compound: Compound,
    config: FitConfig,
    eos_ad,
    fit_params,
    build_arrays,
    errors: "list | None" = None,
):
    """Build (fun, jac) given the case-specific eos_ad, fit_params, build_arrays.

    On failure this returns a *zero* Jacobian, which flattens the gradient and
    makes the optimizer stop immediately reporting success. ``errors`` records
    why, so the caller's stall guard can explain the bogus "convergence".
    """
    T_psat = data.T_psat
    d_psat = data.p_psat
    T_rho = data.T_rho
    d_rho = data.rho
    mw = compound.mw

    n_psat = len(T_psat)
    n_rho = len(T_rho)
    n_total = n_psat + n_rho
    n_params = len(fit_params)

    inv_d_psat = 1.0 / d_psat
    inv_d_rho = 1.0 / d_rho if n_rho > 0 else None
    inv_T_psat = 1.0 / T_psat
    psat_cost_scale = np.sqrt(config.w_psat / n_psat) / config.f_scale["psat"]
    rho_cost_scale = np.sqrt(config.w_rho / n_rho) / config.f_scale["rho"] if n_rho > 0 else 0.0
    # Pa → kPa for psat; kmol/m³ → kg/m³ for rho (via mw)
    psat_jac_scale = (np.sqrt(config.w_psat / n_psat) / 1000.0) / d_psat / config.f_scale["psat"]
    rho_jac_scale = (
        (np.sqrt(config.w_rho / n_rho) * mw) / d_rho / config.f_scale["rho"]
        if n_rho > 0 else np.zeros(1)
    )

    temps_array_psat = np.expand_dims(np.array(T_psat), 1)
    temps_array_rho = np.expand_dims(np.array(T_rho), 1) if n_rho > 0 else None

    _penalty_f = np.full(n_total, _PENALTY)
    _penalty_J = np.zeros((n_total, n_params))

    def _compute(params_vec):
        p = params_vec**2
        chain = 2.0 * params_vec
        pa_psat, pa_rho = build_arrays(p)

        try:
            derivs_psat = feos.Property.vapor_pressure_derivatives(
                eos_ad, fit_params, pa_psat, temps_array_psat
            )
        except Exception as exc:
            if errors is not None:
                errors.append(exc)
            return _penalty_f, _penalty_J, None, None

        mask_psat = derivs_psat[2]
        p_pred = np.full(n_psat, np.nan)
        # feos >=0.10 returns full-length arrays; select the converged entries.
        p_pred[mask_psat] = derivs_psat[0][mask_psat] / 1000.0  # Pa → kPa

        if not mask_psat.all():
            if config.extrapolate_psat and mask_psat.sum() >= 2:
                X = np.column_stack([np.ones(mask_psat.sum()), inv_T_psat[mask_psat]])
                coeffs = np.linalg.lstsq(X, np.log(p_pred[mask_psat]), rcond=None)[0]
                p_pred[~mask_psat] = np.exp(
                    coeffs[0] + coeffs[1] * inv_T_psat[~mask_psat]
                )
            else:
                return _penalty_f, _penalty_J, None, None

        f_psat = psat_cost_scale * (p_pred * inv_d_psat - 1.0)
        jac_full_psat = np.zeros((n_psat, n_params))
        jac_full_psat[mask_psat] = derivs_psat[1][mask_psat]
        J_psat = jac_full_psat * psat_jac_scale[:, np.newaxis] * chain[np.newaxis, :]

        rho_pred = None
        if n_rho > 0:
            try:
                derivs_rho = feos.Property.equilibrium_liquid_density_derivatives(
                    eos_ad, fit_params, pa_rho, temps_array_rho
                )
            except Exception as exc:
                if errors is not None:
                    errors.append(exc)
                return _penalty_f, _penalty_J, None, None

            mask_rho = derivs_rho[2]
            rho_pred = np.full(n_rho, np.nan)
            rho_pred[mask_rho] = derivs_rho[0][mask_rho] * mw  # kmol/m³ → kg/m³

            if not mask_rho.all():
                return _penalty_f, _penalty_J, None, None

            f_rho = rho_cost_scale * (rho_pred * inv_d_rho - 1.0)
            jac_full_rho = np.zeros((n_rho, n_params))
            jac_full_rho[mask_rho] = derivs_rho[1][mask_rho]
            J_rho = jac_full_rho * rho_jac_scale[:, np.newaxis] * chain[np.newaxis, :]

            return (
                np.concatenate([f_psat, f_rho]),
                np.vstack([J_psat, J_rho]),
                p_pred,
                rho_pred,
            )

        return f_psat, J_psat, p_pred, rho_pred

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
    n_psat = len(data.T_psat)
    n_rho = len(data.T_rho)
    assoc = spec.na is not None and spec.nb is not None and spec.na > 0 and spec.nb > 0
    fit_mu = spec.mu is None

    if assoc and fit_mu:
        eos_ad, fit_params, build_arrays = _setup_assoc_mu(spec, n_psat, n_rho)
    elif assoc:
        eos_ad, fit_params, build_arrays = _setup_assoc(spec, n_psat, n_rho)
    elif fit_mu:
        eos_ad, fit_params, build_arrays = _setup_nonassoc_mu(spec, n_psat, n_rho)
    else:
        eos_ad, fit_params, build_arrays = _setup_nonassoc(spec, n_psat, n_rho)

    errors: list = []
    fun, jac = _make_core(
        data, compound, config, eos_ad, fit_params, build_arrays, errors
    )
    return fun, jac, errors
