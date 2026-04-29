"""LLE k_ij fitting from liquid-liquid equilibrium data."""

import time
from pathlib import Path
from types import SimpleNamespace

import feos
import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _apply_induced_association,
    _build_binary_eos,
    _fit_kij_polynomial,
    _kij_at_T,
    _load_pure_records,
)
from fit_pcsaft._csv import SCHEMA_LLE, load_csv
from fit_pcsaft._binary.result import BinaryFitResult

# 51 feed compositions, sigmoid-spaced to sample densely near x1=0 and x1=1.
# s(i) = 0.05*i + 6e-5*i^3,  r(i) = exp(s(i)),  x1(i) = 1/(1+r(i))
# i in {-50, -48, ..., 48, 50}  →  x1 from ~0.9999 down to ~0.0001
_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
_LLE_FEEDS: list[float] = (1.0 / (1.0 + np.exp(_s))).tolist()
del _i, _s

# Number of coarse k_ij scan points used to pick the initial guess for each temperature
_N_KIJ_SCAN = 13


def fit_kij_lle(
    id1: str,
    id2: str,
    lle_path: "Path | str",
    params_path: "Path | str | list[Path | str]",
    kij_order: int = 0,
    kij_t_ref: float = 298.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    t_min: "si.SIObject | None" = None,
    t_max: "si.SIObject | None" = None,
    pressure: "si.SIObject" = 1.01325 * si.BAR,
    require_both_phases: bool = True,
    kij_per_point: bool = False,
    induced_assoc: bool = False,
    ucst_target: bool = False,
    relative_residuals: bool = True,
) -> BinaryFitResult:
    """Fit binary interaction parameter k_ij from LLE tieline data.

    Uses a two-stage approach:
    1. For each experimental temperature (or each individual data point when
       kij_per_point=True), solve an independent 1D least-squares problem to
       find the k_ij that best reproduces the tieline compositions.
    2. Fit a polynomial k_ij(T) to the collected (T, k_ij) pairs.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    lle_path : Path | str
        CSV file with columns: T, x1_I (phase I mole fraction); x1_II optional.
    params_path : Path | str | list
        Feos-compatible JSON parameter file(s).
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, 2=quadratic.
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K].
    kij_bounds : tuple
        (lower, upper) bounds for k_ij at each temperature.
    temperature_unit : si.SIObject
        Unit of T column in CSV (default: K).
    t_min : si.SIObject | None
        Lower temperature bound for fitting.
    t_max : si.SIObject | None
        Upper temperature bound for fitting.
    pressure : si.SIObject
        Pressure for tp_flash calculations (default: 1 bar).
    require_both_phases : bool
        If True (default), skip temperatures where only one phase composition
        is available. Single-phase-only points produce off-trend k_ij values
        because the 1D problem is under-constrained.
    kij_per_point : bool
        If False (default), aggregate rows at the same temperature and fit one
        k_ij per unique temperature (per tie line). If True, fit one k_ij per
        individual CSV row without any averaging across rows at the same T.
    induced_assoc : bool
        If True, apply the induced-association mixing rule. Requires exactly one
        self-associating component (with epsilon_k_ab > 0). The non-associating
        component is assigned epsilon_k_ab = 0 and kappa_ab copied from the
        self-associating component, with na = nb = 1 (2B scheme).
    ucst_target : bool
        If True, fit k_ij using only the highest temperature data point (closest
        to the UCST). Useful when the primary goal is to reproduce the critical
        solution temperature rather than the full phase envelope shape. The warm-
        start anchor is still computed from the lowest available temperature.

    Returns
    -------
    BinaryFitResult
    """
    record1, record2 = _load_pure_records(params_path, id1, id2)
    if induced_assoc:
        record1, record2 = _apply_induced_association(record1, record2)
    _lle_raw = load_csv(lle_path, SCHEMA_LLE)
    data: dict[str, np.ndarray] = {"T": _lle_raw["T"]}
    if "x1_I" in _lle_raw:
        data["x1_I"] = _lle_raw["x1_I"]
    if "x1_II" in _lle_raw:
        data["x1_II"] = _lle_raw["x1_II"]
    data_full = {k: v.copy() for k, v in data.items()}

    # Temperature filter
    if t_min is not None or t_max is not None:
        mask = np.ones(len(data["T"]), dtype=bool)
        if t_min is not None:
            mask &= data["T"] >= float(t_min / temperature_unit)
        if t_max is not None:
            mask &= data["T"] <= float(t_max / temperature_unit)
        data = {k: v[mask] for k, v in data.items()}

    T_arr = data["T"].astype(float)
    t_scale = float(temperature_unit / si.KELVIN)
    has_I = "x1_I" in data
    has_II = "x1_II" in data

    x1_I_arr = data["x1_I"].astype(float) if has_I else None
    x1_II_arr = data["x1_II"].astype(float) if has_II else None

    # Stage 1: point-wise k_ij fitting
    t0 = time.perf_counter()
    if kij_per_point:
        aggregated = _individual_lle_points(T_arr, x1_I_arr, x1_II_arr, t_scale)
    else:
        aggregated = _aggregate_lle_data(T_arr, x1_I_arr, x1_II_arr, t_scale)

    if require_both_phases:
        aggregated = [(T, xi, xii) for T, xi, xii in aggregated
                      if xi is not None and xii is not None]

    T_anchor_K = min(t for t, _, __ in aggregated)  # lowest T — warm-start seed, computed before ucst filter

    if ucst_target:
        T_ucst = max(t for t, _, __ in aggregated)
        aggregated = [(T, xi, xii) for T, xi, xii in aggregated if abs(T - T_ucst) < 1e-3]

    T_fitted = []
    kij_fitted = []
    cost_fitted = []
    fitted_point_meta = []  # (exp_I, exp_II, feeds) for post-poly ARD re-evaluation
    total_nfev = 0

    for T_K, exp_I, exp_II in aggregated:
        feeds = _exp_feeds(exp_I, exp_II) + _LLE_FEEDS
        n_phases = (1 if exp_I is not None else 0) + (1 if exp_II is not None else 0)
        # Penalty cost = 0.5 * n_phases * 1.0^2; accept anything below that
        penalty_cost = 0.5 * n_phases * 0.99

        def residuals(kij_arr, T_K=T_K, exp_I=exp_I, exp_II=exp_II, feeds=feeds):
            return _residuals_at_T(
                kij_arr,
                T_K,
                exp_I,
                exp_II,
                record1,
                record2,
                pressure,
                feeds,
                T_anchor_K=T_anchor_K,
                relative_residuals=relative_residuals,
            )

        # Coarse scan to find best initial k_ij guess (avoids getting trapped in
        # the flat penalty region when the EOS only shows LLE at large k_ij values).
        kij_scan = np.linspace(kij_bounds[0], kij_bounds[1], _N_KIJ_SCAN)
        best_x0 = 0.0
        best_scan_cost = np.inf
        for kij_val in kij_scan:
            try:
                c = 0.5 * float(np.sum(residuals([kij_val]) ** 2))
                if c < best_scan_cost:
                    best_scan_cost = c
                    best_x0 = kij_val
            except Exception:
                pass

        try:
            res = least_squares(
                residuals,
                x0=[best_x0],
                bounds=([kij_bounds[0]], [kij_bounds[1]]),
                method="trf",
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=1000,
            )
            total_nfev += res.nfev
            # Accept if optimizer found a real two-phase solution (cost below penalty)
            if res.cost < penalty_cost:
                T_fitted.append(T_K)
                kij_fitted.append(float(res.x[0]))
                # ARD% = 100 * mean(|relative residuals|)
                cost_fitted.append(100.0 * float(np.mean(np.abs(res.fun))))
                fitted_point_meta.append((exp_I, exp_II, feeds))
        except Exception:
            continue

    if len(T_fitted) == 0:
        raise RuntimeError("No temperatures converged. Try relaxing kij_bounds.")
    effective_order = min(kij_order, len(T_fitted) - 1)

    # Stage 2: ARD-weighted polynomial fit to k_ij(T) trend
    T_fitted_arr = np.array(T_fitted)
    kij_fitted_arr = np.array(kij_fitted)
    kij_coeffs, _ = _fit_kij_polynomial(
        T_fitted_arr, kij_fitted_arr, np.array(cost_fitted), kij_order, kij_t_ref
    )

    # Post-poly ARD: re-evaluate residuals at the polynomial k_ij for each point
    ard_poly = []
    for T_K, (exp_I, exp_II, feeds) in zip(T_fitted_arr, fitted_point_meta):
        kij_poly = _kij_at_T(kij_coeffs, float(T_K), kij_t_ref)
        try:
            r = _residuals_at_T(
                [kij_poly],
                float(T_K),
                exp_I,
                exp_II,
                record1,
                record2,
                pressure,
                feeds,
                T_anchor_K=T_anchor_K,
                relative_residuals=relative_residuals,
            )
            ard_poly.append(100.0 * float(np.mean(np.abs(r))))
        except Exception:
            pass
    ard_poly_arr = np.array(ard_poly)

    # Store pointwise data for diagnostic plotting
    data["T_kij"] = T_fitted_arr
    data["kij_pointwise"] = kij_fitted_arr
    data["ard_pointwise"] = np.array(cost_fitted)       # per-point optimal (stage 1)
    data["ard_pointwise_poly"] = ard_poly_arr            # at polynomial k_ij (stage 2)

    # Reported ARD: post-polynomial, excluding machine-precision near-zeros
    meaningful = ard_poly_arr[ard_poly_arr > 0.01]
    ard = float(meaningful.mean()) if len(meaningful) > 0 else float(np.mean(ard_poly_arr))

    # Residuals for the polynomial fit (k_ij poly vs point-wise k_ij values)
    poly_resid_vals = kij_fitted_arr - np.array(
        [_kij_at_T(kij_coeffs, T, kij_t_ref) for T in T_fitted_arr]
    )
    final_residuals = poly_resid_vals

    poly_result = SimpleNamespace(
        x=kij_coeffs,
        fun=final_residuals,
        cost=float(np.sum(final_residuals**2)) / 2.0,
        success=True,
        nfev=total_nfev,
        message="Point-wise LLE fitting completed",
    )

    eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))

    return BinaryFitResult(
        kij_coeffs=kij_coeffs,
        kij_t_ref=kij_t_ref,
        id1=id1,
        id2=id2,
        equilibrium_type="lle",
        eos=eos_ref,
        data=data,
        data_full=data_full,
        ard=ard,
        scipy_result=poly_result,
        time_elapsed=time.perf_counter() - t0,
        t_filter_min_K=float(t_min / si.KELVIN) if t_min is not None else float("nan"),
        t_filter_max_K=float(t_max / si.KELVIN) if t_max is not None else float("nan"),
        _record1=record1,
        _record2=record2,
    )


def _individual_lle_points(
    T_arr: np.ndarray,
    x1_I_arr: "np.ndarray | None",
    x1_II_arr: "np.ndarray | None",
    t_scale: float,
) -> "list[tuple[float, float | None, float | None]]":
    """Return one entry per CSV row without averaging.

    Returns list of (T_K, x1_I_or_None, x1_II_or_None), preserving row order.
    """
    T_K_arr = T_arr * t_scale
    result = []
    for i in range(len(T_K_arr)):
        exp_I = None
        exp_II = None
        if x1_I_arr is not None and not np.isnan(x1_I_arr[i]):
            exp_I = float(x1_I_arr[i])
        if x1_II_arr is not None and not np.isnan(x1_II_arr[i]):
            exp_II = float(x1_II_arr[i])
        if exp_I is not None or exp_II is not None:
            result.append((float(T_K_arr[i]), exp_I, exp_II))
    return result


def _aggregate_lle_data(
    T_arr: np.ndarray,
    x1_I_arr: "np.ndarray | None",
    x1_II_arr: "np.ndarray | None",
    t_scale: float,
) -> "list[tuple[float, float | None, float | None]]":
    """Group rows by unique temperature, averaging available compositions.

    Returns list of (T_K, mean_x1_I_or_None, mean_x1_II_or_None).
    """
    T_K_arr = T_arr * t_scale
    unique_T = np.unique(np.round(T_K_arr, 4))
    result = []
    for T_K in unique_T:
        mask = np.abs(T_K_arr - T_K) < 1e-3
        exp_I = None
        exp_II = None
        if x1_I_arr is not None:
            vals = x1_I_arr[mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                exp_I = float(np.mean(valid))
        if x1_II_arr is not None:
            vals = x1_II_arr[mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                exp_II = float(np.mean(valid))
        if exp_I is not None or exp_II is not None:
            result.append((float(T_K), exp_I, exp_II))
    return result


def _exp_feeds(exp_I: "float | None", exp_II: "float | None") -> "list[float]":
    """Data-guided feed compositions to try first, before the 51-point grid."""
    candidates = []
    if exp_I is not None and exp_II is not None:
        candidates.append((exp_I + exp_II) / 2.0)
        candidates.append(0.3 * exp_I + 0.7 * exp_II)
        candidates.append(0.7 * exp_I + 0.3 * exp_II)
    elif exp_II is not None:
        candidates.append(exp_II * 0.9)
        candidates.append(exp_II * 0.5)
    elif exp_I is not None:
        candidates.append(min(exp_I + 0.1, 0.99))
    return [float(np.clip(c, 0.01, 0.99)) for c in candidates]


def _residuals_at_T(
    kij_arr: "list[float]",
    T_K: float,
    exp_I: "float | None",
    exp_II: "float | None",
    record1,
    record2,
    pressure,
    feeds: "list[float]",
    T_anchor_K: "float | None" = None,
    relative_residuals: bool = True,
) -> np.ndarray:
    """Residual vector for least_squares at a single temperature.

    Returns composition errors on each available phase — relative
    ((x_pred-x)/x) when relative_residuals=True, absolute (x_pred-x)
    otherwise. A penalty of [1.0, ...] is returned on tp_flash failure.

    When T_anchor_K is provided and T_K > T_anchor_K, a warm-start PE is built
    at T_anchor_K using the *same* EOS (same k_ij) and passed as initial_state.
    This steers the flash toward LLE at higher temperatures without EOS-
    incompatibility issues or Jacobian flattening.
    """
    kij = float(kij_arr[0])
    n_resid = (1 if exp_I is not None else 0) + (1 if exp_II is not None else 0)
    penalty = np.ones(n_resid)

    eos = _build_binary_eos(record1, record2, kij)

    # Build anchor PE at T_anchor using the same EOS — EOS-compatible warm start
    anchor_pe = None
    if T_anchor_K is not None and T_K > T_anchor_K + 0.5 and len(feeds) > 0:
        try:
            feed_a = np.array([feeds[0], 1.0 - feeds[0]]) * si.MOL
            s_a = feos.State(
                eos,
                T_anchor_K * si.KELVIN,
                pressure=pressure,
                moles=feed_a,
                density_initialization="liquid",
            )
            anchor_pe = s_a.tp_flash(max_iter=500)
        except Exception:
            pass

    for z1 in feeds:
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            feed_state = feos.State(
                eos,
                T_K * si.KELVIN,
                pressure=pressure,
                moles=feed,
                density_initialization="liquid",
            )
            pe = feed_state.tp_flash(initial_state=anchor_pe, max_iter=1000)
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            if abs(x_a - x_b) < 1e-4:
                continue
            pred_I, pred_II = min(x_a, x_b), max(x_a, x_b)
            resids = []
            if exp_I is not None:
                resids.append((pred_I - exp_I) / max(exp_I, 1e-6) if relative_residuals else pred_I - exp_I)
            if exp_II is not None:
                resids.append((pred_II - exp_II) / max(exp_II, 1e-6) if relative_residuals else pred_II - exp_II)
            return np.array(resids)
        except Exception:
            continue

    return penalty
