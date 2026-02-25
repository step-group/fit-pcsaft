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
    _make_f_and_df_numerical,
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
    # Small alcohols (methanol, ethanol)
    [2.0, 3.1, 200, 0.03, 2600],
    [2.5, 3.2, 220, 0.02, 2500],
    # Mid-size alcohols (propanol–hexanol)
    [3.0, 3.4, 230, 0.03, 2500],
    [3.5, 3.5, 240, 0.02, 2600],
    # Large alcohols (octanol–decanol and beyond)
    [4.5, 3.7, 240, 0.01, 2700],
    [5.0, 3.8, 250, 0.01, 2600],
    [5.5, 3.9, 260, 0.01, 2700],
    [6.0, 3.8, 250, 0.02, 2500],
]
_MU_NONASSOC: list[float] = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
_MU_ASSOC: list[float] = [1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]


def _get_initial_sets(fit_mu: bool, assoc: bool = False):
    """Get initial parameter sets for the sequential multi-start optimization."""
    base, mus = (_X0_ASSOC, _MU_ASSOC) if assoc else (_X0_NONASSOC, _MU_NONASSOC)
    if fit_mu:
        return [b[:3] + [mu] + b[3:] for b, mu in zip(base, mus)]
    return base


def _predict_psat(eos, T_psat, temperature_unit, pressure_unit) -> Optional[np.ndarray]:
    try:
        return np.array(
            [
                feos.PhaseEquilibrium.vapor_pressure(eos, T * temperature_unit)[0]
                / pressure_unit
                for T in T_psat
            ]
        )
    except Exception:
        return None


def _predict_rho(eos, T_rho, temperature_unit, density_unit) -> Optional[np.ndarray]:
    if len(T_rho) == 0:
        return None
    try:
        return np.array(
            [
                feos.PhaseEquilibrium.pure(eos, T * temperature_unit).liquid.mass_density()
                / density_unit
                for T in T_rho
            ]
        )
    except Exception:
        return None


def _predict_hvap(eos, T_hvap, temperature_unit, enthalpy_unit) -> Optional[np.ndarray]:
    if len(T_hvap) == 0:
        return None
    try:
        return np.array(
            [
                (
                    vle.vapor.molar_enthalpy(feos.Contributions.Residual)
                    - vle.liquid.molar_enthalpy(feos.Contributions.Residual)
                )
                / enthalpy_unit
                for T in T_hvap
                for vle in [feos.PhaseEquilibrium.pure(eos, T * temperature_unit)]
            ]
        )
    except Exception:
        return None


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

    def ard(pred, ref):
        return 100.0 * np.mean(np.abs((pred - ref) / ref)) if pred is not None else np.nan

    p_pred = _predict_psat(eos, data.T_psat, units.temperature, units.pressure)
    rho_pred = _predict_rho(eos, data.T_rho, units.temperature, units.density)
    hvap_pred = _predict_hvap(eos, data.T_hvap, units.temperature, units.enthalpy)

    return eos, ard(p_pred, data.p_psat), ard(rho_pred, data.rho), ard(hvap_pred, data.hvap)


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
    hvap_path: Optional[Path | str] = None,
    mu: Optional[float] = 0.0,
    q: float = 0.0,
    na: Optional[int] = None,
    nb: Optional[int] = None,
    psat_weight: float = 3.0,
    density_weight: float = 2.0,
    hvap_weight: float = 1.0,
    extrapolate_psat: bool = False,
    loss: str = "linear",
    f_scale: float = 1.0,
    pressure_unit: si.SIObject = si.KILO * si.PASCAL,
    temperature_unit: si.SIObject = si.KELVIN,
    density_unit: si.SIObject = si.KILOGRAM / (si.METER**3),
    enthalpy_unit: si.SIObject = si.KILO * si.JOULE / si.MOL,
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
        hvap_path : Path | str or None
            Path to enthalpy of vaporization CSV (T, Hvap). Optional. When provided,
            forces numerical Jacobian (no AD available for Hvap).
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
        hvap_weight : float
            Weight for enthalpy of vaporization in cost function (default: 1.0)
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
        enthalpy_unit : si.SIObject
            Unit for enthalpy of vaporization in CSV (default: kJ/mol)
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

    if hvap_path is not None:
        _t_hvap, _d_hvap = _load_csv(hvap_path)
    else:
        _t_hvap, _d_hvap = np.array([]), np.array([])

    data = PureData(
        T_psat=_t_psat,
        p_psat=_p_psat,
        T_rho=_t_rhoL,
        rho=_d_rhoLsat,
        T_hvap=_t_hvap,
        hvap=_d_hvap,
    )

    compound = Compound(identifier=identifier, mw=float(mw))

    spec = ModelSpec(mu=mu, na=na, nb=nb, q=q)

    units = Units(
        temperature=temperature_unit,
        pressure=pressure_unit,
        density=density_unit,
        enthalpy=enthalpy_unit,
    )

    config = FitConfig(
        w_psat=psat_weight,
        w_rho=density_weight,
        w_hvap=hvap_weight,
        extrapolate_psat=extrapolate_psat,
    )

    use_numerical_jac = q != 0.0 or hvap_path is not None

    if use_numerical_jac:
        reasons = []
        if q != 0.0:
            reasons.append("q != 0 (quadrupole term not in analytical Jacobian)")
        if hvap_path is not None:
            reasons.append("hvap data provided (no AD available for Hvap)")
        print(
            f"Note: {'; '.join(reasons)} — "
            "falling back to numerical Jacobian (2-point).\n"
        )
        cost_fn, jac_fn = _make_f_and_df_numerical(data, compound, spec, units, config)
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

    if not results:
        raise RuntimeError(
            "All initial parameter sets failed. Try different starting values or "
            "check that the data covers a valid temperature range."
        )

    result = min(results, key=lambda r: r.cost)
    time_elapsed = time.perf_counter() - t0

    params_fitted = result.x**2

    eos_final, ard_psat, ard_rho, ard_hvap = _compute_ard_metrics(
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
        ard_hvap=ard_hvap,
        scipy_result=result,
        time_elapsed=time_elapsed,
        input_name=id,
    )


#! TODO def fit_binary() -> FitResult: ...
