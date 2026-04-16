"""SLE k_ij fitting from solid-liquid equilibrium (solubility) data."""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _build_binary_eos,
    _fit_kij_polynomial,
    _kij_at_T,
    _load_pure_records,
    _make_binary_jac_fn,
)
from fit_pcsaft._csv import SCHEMA_SLE, load_csv
from fit_pcsaft._binary.result import BinaryFitResult

_R = si.RGAS / (si.JOULE / (si.MOL * si.KELVIN))
_N_KIJ_SCAN = 13


def fit_kij_sle(
    id1: str,
    id2: str,
    sle_path: "Path | str",
    params_path: "Path | str",
    tm: "si.SIObject",
    delta_hfus: "si.SIObject",
    solid_index: int = 0,
    tm2: "si.SIObject | None" = None,
    delta_hfus2: "si.SIObject | None" = None,
    kij_order: int = 0,
    kij_t_ref: float = 293.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    t_min: "si.SIObject | None" = None,
    t_max: "si.SIObject | None" = None,
    scipy_kwargs: "dict | None" = None,
    kij_per_point: bool = False,
) -> BinaryFitResult:
    """Fit binary interaction parameter k_ij from SLE solubility data.

    Uses the Schröder-van Laar equation combined with PC-SAFT activity
    coefficients to predict the liquid-phase composition at saturation.

    For eutectic systems where both components can solidify, supply ``tm2``
    and ``delta_hfus2`` for the second solid.  The code then auto-assigns
    each data point to the branch (solid 1 or solid 2) that gives the
    smaller residual.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    sle_path : Path | str
        CSV file with columns T (temperature) and x (mole fraction of id1
        in the saturated liquid).
    params_path : Path | str
        Feos-compatible JSON parameter file (output of FitResult.to_json).
    tm : si.SIObject
        Melting point of the solid described by ``solid_index``.
    delta_hfus : si.SIObject
        Molar enthalpy of fusion of the solid described by ``solid_index``.
    solid_index : int
        Which component is the first solid: 0 = id1, 1 = id2. Default: 0.
        Use solid_index=1 when id2 crystallises out and the CSV x column is
        x_id1 (the liquid fraction, → 0 as T → Tm).
    tm2 : si.SIObject, optional
        Melting point of the second solid (the other component).
        Required for eutectic systems.
    delta_hfus2 : si.SIObject, optional
        Molar enthalpy of fusion of the second solid.
        Required together with tm2.
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, …
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K]. Default: 293.15 K.
    kij_bounds : tuple
        (lower, upper) bounds for the constant term k_ij0.
    temperature_unit : si.SIObject
        Unit of T column in CSV (default: K).
    t_min : si.SIObject | None
        Lower temperature bound. Rows with T < t_min are excluded.
    t_max : si.SIObject | None
        Upper temperature bound. Rows with T > t_max are excluded.
    scipy_kwargs : dict | None
        Overrides for scipy.optimize.least_squares keyword arguments.

    Returns
    -------
    BinaryFitResult
    """
    if solid_index not in (0, 1):
        raise ValueError("solid_index must be 0 or 1")
    if (tm2 is None) != (delta_hfus2 is None):
        raise ValueError("tm2 and delta_hfus2 must both be provided or both omitted")

    record1, record2 = _load_pure_records(params_path, id1, id2)
    data = load_csv(sle_path, SCHEMA_SLE)
    data_full = {k: v.copy() for k, v in data.items()}

    # --- Temperature filter --------------------------------------------------
    if t_min is not None or t_max is not None:
        mask = np.ones(len(data["T"]), dtype=bool)
        if t_min is not None:
            mask &= data["T"] >= float(t_min / temperature_unit)
        if t_max is not None:
            mask &= data["T"] <= float(t_max / temperature_unit)
        data = {k: v[mask] for k, v in data.items()}

    T_arr = data["T"]
    x1_arr = data["x1"]
    n_rows = len(T_arr)

    Tm_K = float(tm / si.KELVIN)
    dHfus_J = float(delta_hfus / (si.JOULE / si.MOL))
    eutectic = tm2 is not None
    Tm2_K = float(tm2 / si.KELVIN) if eutectic else float("nan")
    dHfus2_J = float(delta_hfus2 / (si.JOULE / si.MOL)) if eutectic else float("nan")
    solid_index2 = 1 - solid_index  # the other solid
    t_scale = float(temperature_unit / si.KELVIN)

    def _predict_x1_for(eos, T_K, x1_start, si_idx, Tm, dHfus):
        """Solve Schröder-van Laar for a given solid component index."""
        rhs = -(dHfus / _R) * (1.0 / T_K - 1.0 / Tm)
        if si_idx == 0:
            x_iter = float(np.clip(x1_start, 1e-6, 1.0 - 1e-6))
            for _ in range(50):
                try:
                    liq = _liquid_state(eos, T_K, x_iter)
                    ln_gamma = float(liq.ln_symmetric_activity_coefficient()[0])
                except Exception:
                    return float("nan")
                x_new = float(np.clip(np.exp(rhs - ln_gamma), 1e-9, 1.0 - 1e-9))
                if abs(x_new - x_iter) < 1e-9:
                    return x_new
                x_iter = x_new
            return x_iter
        else:
            x_iter = float(np.clip(1.0 - x1_start, 1e-6, 1.0 - 1e-6))
            for _ in range(50):
                try:
                    liq = _liquid_state(eos, T_K, 1.0 - x_iter)
                    ln_gamma = float(liq.ln_symmetric_activity_coefficient()[1])
                except Exception:
                    return float("nan")
                x_new = float(np.clip(np.exp(rhs - ln_gamma), 1e-9, 1.0 - 1e-9))
                if abs(x_new - x_iter) < 1e-9:
                    return 1.0 - x_new
                x_iter = x_new
            return 1.0 - x_iter

    def _predict_x1(eos, T_K: float, x1_start: float) -> float:
        return _predict_x1_for(eos, T_K, x1_start, solid_index, Tm_K, dHfus_J)

    def _predict_x1_branch2(eos, T_K: float, x1_start: float) -> float:
        return _predict_x1_for(eos, T_K, x1_start, solid_index2, Tm2_K, dHfus2_J)

    def fun(coeffs: np.ndarray) -> np.ndarray:
        resids = np.empty(n_rows)
        kij_per_row = np.array(
            [_kij_at_T(coeffs, float(T_arr[i]), kij_t_ref) for i in range(n_rows)]
        )
        eos_map: dict[float, object] = {}
        for kij_val in np.unique(kij_per_row):
            try:
                eos_map[kij_val] = _build_binary_eos(record1, record2, float(kij_val))
            except Exception:
                eos_map[kij_val] = None

        for i in range(n_rows):
            T_i = float(T_arr[i]) * t_scale
            x1_i = float(x1_arr[i])
            eos = eos_map[kij_per_row[i]]
            if eos is None:
                resids[i] = 1.0
                continue
            try:
                x1_pred = _predict_x1(eos, T_i, x1_i)
                resid = 1.0 if np.isnan(x1_pred) else (x1_pred - x1_i)
            except Exception:
                resid = 1.0
            if eutectic:
                try:
                    x1_pred2 = _predict_x1_branch2(eos, T_i, x1_i)
                    resid2 = 1.0 if np.isnan(x1_pred2) else (x1_pred2 - x1_i)
                except Exception:
                    resid2 = 1.0
                resid = resid if abs(resid) <= abs(resid2) else resid2
            resids[i] = resid
        return resids

    t0 = time.perf_counter()

    # --- Per-point two-stage fitting -----------------------------------------
    if kij_per_point:
        T_fitted, kij_fitted, ard_fitted = [], [], []
        total_nfev = 0

        for i in range(n_rows):
            T_K_i = float(T_arr[i]) * t_scale
            x1_i  = float(x1_arr[i])

            def resid_fn(kij_arr, T_K=T_K_i, x1=x1_i):
                try:
                    eos = _build_binary_eos(record1, record2, float(kij_arr[0]))
                    x1_pred = _predict_x1(eos, T_K, x1)
                    resid = 1.0 if np.isnan(x1_pred) else (x1_pred - x1)
                    if eutectic:
                        x1_pred2 = _predict_x1_branch2(eos, T_K, x1)
                        resid2 = 1.0 if np.isnan(x1_pred2) else (x1_pred2 - x1)
                        resid = resid if abs(resid) <= abs(resid2) else resid2
                    return np.array([resid])
                except Exception:
                    return np.array([1.0])

            kij_scan = np.linspace(kij_bounds[0], kij_bounds[1], _N_KIJ_SCAN)
            best_x0, best_cost = 0.0, np.inf
            for kv in kij_scan:
                try:
                    c = 0.5 * float(resid_fn([kv])[0] ** 2)
                    if c < best_cost:
                        best_cost, best_x0 = c, kv
                except Exception:
                    pass

            try:
                res = least_squares(
                    resid_fn, x0=[best_x0],
                    bounds=([kij_bounds[0]], [kij_bounds[1]]),
                    method="trf",
                    ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=500,
                )
                total_nfev += res.nfev
                if res.cost < 0.5 * 0.99:
                    T_fitted.append(T_K_i)
                    kij_fitted.append(float(res.x[0]))
                    ard_fitted.append(100.0 * abs(float(res.fun[0])))
            except Exception:
                continue

        if not T_fitted:
            raise RuntimeError("No SLE points converged. Try relaxing kij_bounds.")

        T_fitted_arr  = np.array(T_fitted)
        kij_fitted_arr = np.array(kij_fitted)
        kij_coeffs, poly_resid = _fit_kij_polynomial(
            T_fitted_arr, kij_fitted_arr, np.array(ard_fitted), kij_order, kij_t_ref
        )
        eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))
        ard = float(np.mean(ard_fitted))

        data["T_kij"]         = T_fitted_arr
        data["kij_pointwise"] = kij_fitted_arr
        data["ard_pointwise"] = np.array(ard_fitted)

        return BinaryFitResult(
            kij_coeffs=kij_coeffs,
            kij_t_ref=kij_t_ref,
            id1=id1,
            id2=id2,
            equilibrium_type="sle",
            eos=eos_ref,
            data=data,
            ard=ard,
            scipy_result=SimpleNamespace(
                x=kij_coeffs, fun=poly_resid,
                cost=float(np.sum(poly_resid**2)) / 2.0,
                success=True, nfev=total_nfev,
                message="Per-point SLE fitting completed",
            ),
            time_elapsed=time.perf_counter() - t0,
            tm_K=Tm_K, delta_hfus_J=dHfus_J, solid_index=solid_index,
            tm2_K=Tm2_K, delta_hfus2_J=dHfus2_J,
            t_filter_min_K=float(t_min / si.KELVIN) if t_min is not None else float("nan"),
            t_filter_max_K=float(t_max / si.KELVIN) if t_max is not None else float("nan"),
            data_full=data_full,
        )

    # --- Global polynomial fit (default) -------------------------------------
    n_coeffs = kij_order + 1
    jac = _make_binary_jac_fn(fun, n_coeffs)

    x0 = np.zeros(n_coeffs)
    lb = [kij_bounds[0]] + [-0.01] * kij_order
    ub = [kij_bounds[1]] + [0.01] * kij_order

    ls_kwargs = {
        "method": "trf",
        "bounds": (lb, ub),
        "ftol": 1e-8,
        "xtol": 1e-8,
        "gtol": 1e-8,
        "max_nfev": 2000,
    }
    if scipy_kwargs:
        ls_kwargs.update(scipy_kwargs)

    result = least_squares(fun, x0, jac=jac, **ls_kwargs)
    time_elapsed = time.perf_counter() - t0

    kij_coeffs = result.x
    eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))

    # ARD — reuse residuals from final evaluation
    final_resids = fun(kij_coeffs)
    valid = np.abs(final_resids) < 0.99
    x1_data_valid = x1_arr.astype(float)[valid]
    mean_x = np.mean(x1_data_valid) if len(x1_data_valid) > 0 else float("nan")
    ard = (
        100.0 * float(np.mean(np.abs(final_resids[valid]))) / mean_x
        if mean_x > 0
        else float("nan")
    )

    return BinaryFitResult(
        kij_coeffs=kij_coeffs,
        kij_t_ref=kij_t_ref,
        id1=id1,
        id2=id2,
        equilibrium_type="sle",
        eos=eos_ref,
        data=data,
        ard=ard,
        scipy_result=result,
        time_elapsed=time_elapsed,
        tm_K=Tm_K,
        delta_hfus_J=dHfus_J,
        solid_index=solid_index,
        tm2_K=Tm2_K,
        delta_hfus2_J=dHfus2_J,
        t_filter_min_K=float(t_min / si.KELVIN) if t_min is not None else float("nan"),
        t_filter_max_K=float(t_max / si.KELVIN) if t_max is not None else float("nan"),
        data_full=data_full,
    )


def _liquid_state(eos, T_K: float, x1: float):
    import feos

    return feos.State(
        eos,
        temperature=T_K * si.KELVIN,
        pressure=1.0 * si.BAR,
        molefracs=np.array([x1, 1.0 - x1]),
        density_initialization="liquid",
    )
