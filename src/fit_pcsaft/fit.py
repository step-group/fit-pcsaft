import time
from pathlib import Path
from typing import Optional

import feos
import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._fit_utils import (
    _build_eos,
    _fetch_compound,
    _load_csv,
    _make_cost_fn,
)
from fit_pcsaft._pure.jacobian import _make_f_and_df
from fit_pcsaft._types import Compound, FitConfig, ModelSpec, PureData, Units
from fit_pcsaft.result import FitResult

_X0_NONASSOC: list[list[float]] = [
    [4.0, 3.7, 300],
    [4.0, 3.7, 250],
    [4.0, 3.7, 200],
    [2.0, 3.7, 300],
    [2.0, 3.7, 250],
    [2.0, 3.0, 200],
]
_X0_ASSOC: list[list[float]] = [
    [2.5, 3.2, 220, 0.03, 2500],
    [2.0, 3.1, 200, 0.02, 2600],
    [3.0, 3.2, 200, 0.04, 2400],
    [2.5, 3.1, 240, 0.03, 2700],
    [2.0, 3.2, 220, 0.04, 2500],
    [3.0, 3.1, 200, 0.02, 2600],
]
_MU_NONASSOC: list[float] = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
_MU_ASSOC: list[float] = [1.0, 1.5, 2.0, 1.0, 1.5, 2.0]


def _get_initial_sets(fit_mu: bool, assoc: bool = False):
    """Get initial parameter sets for the sequential multi-start optimization."""
    base, mus = (_X0_ASSOC, _MU_ASSOC) if assoc else (_X0_NONASSOC, _MU_NONASSOC)
    if fit_mu:
        return [b[:3] + [mu] + b[3:] for b, mu in zip(base, mus)]
    return base


def _compute_ard_metrics(
    params_fitted: np.ndarray,
    data: PureData,
    compound: Compound,
    spec: ModelSpec,
    units: Units,
    eos=None,
):
    """Compute ARD% metrics for fitted parameters."""
    if eos is None:
        eos = _build_eos(params_fitted, compound, spec)

    T_psat = data.T_psat
    p_psat = data.p_psat
    T_rho = data.T_rho
    rho_data = data.rho
    temperature_unit = units.temperature
    pressure_unit = units.pressure
    density_unit = units.density

    try:
        p_pred = np.array(
            [
                feos.PhaseEquilibrium.vapor_pressure(eos, T * temperature_unit)[0]
                / pressure_unit
                for T in T_psat
            ]
        )
    except Exception:
        p_pred = None

    if p_pred is not None:
        ard_psat = 100.0 * np.mean(np.abs((p_pred - p_psat) / p_psat))
    else:
        ard_psat = np.nan

    if len(T_rho) > 0:
        try:
            rho_pred = np.array(
                [
                    feos.PhaseEquilibrium.pure(
                        eos, T * temperature_unit
                    ).liquid.mass_density()
                    / density_unit
                    for T in T_rho
                ]
            )
        except Exception:
            rho_pred = None
    else:
        rho_pred = None

    if rho_pred is not None:
        ard_rho = 100.0 * np.mean(np.abs((rho_pred - rho_data) / rho_data))
    else:
        ard_rho = np.nan

    return eos, ard_psat, ard_rho


def _extract_params_dict(
    params_fitted: np.ndarray, mu: Optional[float], assoc: bool = False
):
    """Extract fitted parameters into output dict."""
    idx = 3
    d = {
        "m": float(params_fitted[0]),
        "sigma": float(params_fitted[1]),
        "epsilon_k": float(params_fitted[2]),
    }
    if mu is None:
        d["mu"] = float(params_fitted[idx])
        idx += 1
    if assoc:
        d["kappa_ab"] = float(params_fitted[idx])
        idx += 1
        d["epsilon_k_ab"] = float(params_fitted[idx])
        idx += 1
    return d


def fit_pure(
    id: str,
    psat_path: Path | str,
    density_path: Path | str,
    mu: Optional[float] = 0.0,
    q: float = 0.0,
    na: Optional[int] = None,
    nb: Optional[int] = None,
    psat_weight: float = 3.0,
    density_weight: float = 2.0,
    extrapolate_psat: bool = False,
    loss: str = "linear",
    f_scale: float = 1.0,
    pressure_unit: si.SIObject = si.KILO * si.PASCAL,
    temperature_unit: si.SIObject = si.KELVIN,
    density_unit: si.SIObject = si.KILOGRAM / (si.METER**3),
    scipy_kwargs: Optional[dict] = None,
) -> FitResult:
    """Fit PC-SAFT parameters to vapor pressure and liquid density data.

    Uses sequential multi-start optimization with 6 initial parameter sets.

    Arguments
    ---------
        id : str
            Compound identifier (name, SMILES, or InChI) for PubChem lookup
        psat_path : Path | str
            Path to vapor pressure CSV (T, Psat)
        density_path : Path | str
            Path to liquid density CSV (T, rhoL)
        mu : float or None
            Dipole moment. If None, mu is fitted. If float, fixed (default: 0.0).
        q : float
            Quadrupole moment, fixed (default: 0.0). Not fitted.
        na : int or None
            Number of association sites A. If provided (with nb), enables associating mode.
        nb : int or None
            Number of association sites B. Must be provided together with na.
        psat_weight : float
            Weight for vapor pressure in cost function (default: 3.0)
        density_weight : float
            Weight for liquid density in cost function (default: 2.0)
        extrapolate_psat : bool
            If True, Clausius-Clapeyron extrapolation fills in psat for temperatures
            where the EOS fails (e.g. above Tc). Useful for datasets that include
            near-critical or supercritical points (default: False).
        loss : str
            Loss function for scipy least_squares: 'linear', 'huber', 'soft_l1',
            'cauchy', 'arctan' (default: 'linear'). Non-linear losses automatically
            switch the solver from LM to TRF.
        f_scale : float
            Soft margin for robust loss functions (default: 1.0). Only used when
            loss != 'linear'.
        pressure_unit : si.SIObject
            Unit for pressure in CSV (default: kPa)
        temperature_unit : si.SIObject
            Unit for temperature in CSV (default: K)
        density_unit : si.SIObject
            Unit for density in CSV (default: kg/m³)
        scipy_kwargs : Optional[dict]
            Optional dict to override scipy least_squares defaults

    Returns
    -------
        FitResult: Fitted parameters, EOS, and fitting quality metrics
    """
    fit_mu = mu is None

    # Normalize: if only one of na/nb given, default the other to 0
    if na is not None and nb is None:
        nb = 0
    elif nb is not None and na is None:
        na = 0

    is_associative = na is not None and nb is not None and na > 0 and nb > 0

    identifier, mw = _fetch_compound(id)

    _t_psat, _p_psat = _load_csv(psat_path)
    _t_rhoL, _d_rhoLsat = _load_csv(density_path)

    data = PureData(T_psat=_t_psat, p_psat=_p_psat, T_rho=_t_rhoL, rho=_d_rhoLsat)

    compound = Compound(identifier=identifier, mw=float(mw))

    spec = ModelSpec(mu=mu, na=na, nb=nb, q=q)

    units = Units(
        temperature=temperature_unit, pressure=pressure_unit, density=density_unit
    )

    config = FitConfig(
        w_psat=psat_weight, w_rho=density_weight, extrapolate_psat=extrapolate_psat
    )

    if q != 0.0:
        # Analytical Jacobian does not include the quadrupole term — fall back to
        # numerical differentiation. Cost function uses individual feos calls.
        print(
            "Note: q != 0 — analytical Jacobian does not include quadrupole term. "
            "Falling back to numerical Jacobian (jac='2-point').\n"
        )

        cost_fn = _make_cost_fn(data, compound, spec, units, config)
        jac_fn = "2-point"

    else:
        cost_fn, jac_fn = _make_f_and_df(data, compound, spec, units, config)

    # LM does not support robust loss functions. Soln: fallback to trf.
    if loss != "linear":
        print(
            f"Note: loss='{loss}' requires method='trf' (LM does not support robust losses). "
            "Falling back to trust region reflective (method='trf').\n"
        )

    use_trf = loss != "linear"
    ls_kwargs = {
        "method": "trf" if use_trf else "lm",
        "ftol": 1e-08,
        "xtol": 1e-08,
        "gtol": 1e-08,
        "max_nfev": 10_000,
    }
    if use_trf:
        ls_kwargs["loss"] = loss
        ls_kwargs["f_scale"] = f_scale
    if scipy_kwargs:
        ls_kwargs.update(scipy_kwargs)

    t0 = time.perf_counter()
    results = []
    for x0i in _get_initial_sets(fit_mu, assoc=is_associative):
        try:
            res = least_squares(cost_fn, np.sqrt(x0i), jac=jac_fn, **ls_kwargs)
            results.append(res)
        except Exception:
            continue

    result = min(results, key=lambda r: r.cost)
    time_elapsed = time.perf_counter() - t0

    params_fitted = result.x**2

    eos_final, ard_psat, ard_rho = _compute_ard_metrics(
        params_fitted,
        data,
        compound,
        spec,
        units,
    )

    params_dict = _extract_params_dict(params_fitted, mu, assoc=is_associative)

    return FitResult(
        params=params_dict,
        eos=eos_final,
        data=data,
        compound=compound,
        spec=spec,
        units=units,
        ard_psat=ard_psat,
        ard_rho=ard_rho,
        scipy_result=result,
        time_elapsed=time_elapsed,
    )


#! TODO def fit_binary() -> FitResult: ...
