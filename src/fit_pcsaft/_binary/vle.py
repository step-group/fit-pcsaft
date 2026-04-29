"""VLE k_ij fitting from bubble-point data."""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _apply_induced_association,
    _build_binary_eos,
    _fit_kij_polynomial,
    _kij_at_T,
    _load_pure_records,
    _make_binary_jac_fn,
)
from fit_pcsaft._csv import SCHEMA_VLE, load_csv
from fit_pcsaft._binary.result import BinaryFitResult

_N_KIJ_SCAN = 13


def fit_kij_vle(
    id1: str,
    id2: str,
    vle_path: "Path | str",
    params_path: "Path | str",
    kij_order: int = 0,
    kij_t_ref: float = 293.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    pressure_unit=si.KILO * si.PASCAL,
    t_min: "si.SIObject | None" = None,
    t_max: "si.SIObject | None" = None,
    scipy_kwargs: "dict | None" = None,
    kij_per_point: bool = False,
    induced_assoc: bool = False,
    relative_residuals: bool = True,
) -> BinaryFitResult:
    """Fit binary interaction parameter k_ij from VLE bubble-point data.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    vle_path : Path | str
        CSV file with columns: T, P, x1 (required); y1 (optional).
    params_path : Path | str
        Feos-compatible JSON parameter file (output of FitResult.to_json).
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, 2=quadratic, 3=cubic.
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K]. Default: 293.15 K.
    kij_bounds : tuple
        (lower, upper) bounds for the constant term k_ij0. Higher-order
        coefficients use tighter bounds of ±0.01.
    temperature_unit : si.SIObject
        Unit of T column in CSV (default: K).
    pressure_unit : si.SIObject
        Unit of P column in CSV (default: kPa).
    t_min : si.SIObject | None
        Lower temperature bound. Rows with T < t_min are excluded.
    t_max : si.SIObject | None
        Upper temperature bound. Rows with T > t_max are excluded.
    scipy_kwargs : dict | None
        Overrides for scipy.optimize.least_squares keyword arguments.
    kij_per_point : bool
        If False (default), fit a single k_ij polynomial to all data simultaneously.
        If True, use a two-stage approach: fit one k_ij per data point (considering
        both bubble-P and dew-y1 residuals when y1 is present), then fit a polynomial
        to the collected (T, k_ij) pairs. Stores diagnostic arrays T_kij,
        kij_pointwise, ard_pointwise, and ard_pointwise_poly in the result.
    induced_assoc : bool
        If True, apply the induced-association mixing rule. Requires exactly one
        self-associating component (with epsilon_k_ab > 0). The non-associating
        component is assigned epsilon_k_ab = 0 and kappa_ab copied from the
        self-associating component, with na = nb = 1 (2B scheme). Typical use:
        water (self-associating) + polar non-associating solvent (e.g. MIBK, acetone).

    Returns
    -------
    BinaryFitResult
    """
    record1, record2 = _load_pure_records(params_path, id1, id2)
    if induced_assoc:
        record1, record2 = _apply_induced_association(record1, record2)
    data = load_csv(vle_path, SCHEMA_VLE)
    data_full = {k: v.copy() for k, v in data.items()}

    # --- Drop pure-component endpoints (x1≈0 or x1≈1) ----------------------
    # Bubble/dew point calculations are singular for pure components.
    _x1_raw = data["x1"].astype(float)
    _mix_mask = (_x1_raw > 1e-4) & (_x1_raw < 1.0 - 1e-4)
    if not _mix_mask.all():
        data = {k: v[_mix_mask] for k, v in data.items()}

    # --- Temperature filter --------------------------------------------------
    if t_min is not None or t_max is not None:
        mask = np.ones(len(data["T"]), dtype=bool)
        if t_min is not None:
            mask &= data["T"] >= float(t_min / temperature_unit)
        if t_max is not None:
            mask &= data["T"] <= float(t_max / temperature_unit)
        data = {k: v[mask] for k, v in data.items()}

    T_arr = data["T"]
    P_arr = data["P"]
    x1_arr = data["x1"]
    has_y1 = "y1" in data
    y1_arr = data["y1"] if has_y1 else None

    t0 = time.perf_counter()
    t_scale = float(temperature_unit / si.KELVIN)

    # --- Per-point two-stage fitting -----------------------------------------
    if kij_per_point:
        n_rows = len(T_arr)
        T_fitted, kij_fitted, cost_fitted, fitted_point_meta = [], [], [], []
        total_nfev = 0

        for i in range(n_rows):
            T_csv_i = float(T_arr[i])
            P_i = float(P_arr[i])
            x1_i = float(x1_arr[i])
            y1_i = float(y1_arr[i]) if has_y1 else None
            T_K_i = T_csv_i * t_scale

            def resid_fn(kij_arr, T_csv=T_csv_i, P=P_i, x1=x1_i, y1=y1_i):
                return _residuals_vle_point(
                    kij_arr, T_csv, P, x1, y1,
                    record1, record2, temperature_unit, pressure_unit,
                    relative_residuals=relative_residuals,
                )

            kij_scan = np.linspace(kij_bounds[0], kij_bounds[1], _N_KIJ_SCAN)
            best_x0, best_scan_cost = 0.0, np.inf
            for kij_val in kij_scan:
                try:
                    c = 0.5 * float(np.sum(resid_fn([kij_val]) ** 2))
                    if c < best_scan_cost:
                        best_scan_cost, best_x0 = c, kij_val
                except Exception:
                    pass

            n_r = 2 if has_y1 else 1
            penalty_cost = 0.5 * n_r * 0.99
            try:
                res = least_squares(
                    resid_fn,
                    x0=[best_x0],
                    bounds=([kij_bounds[0]], [kij_bounds[1]]),
                    method="trf",
                    ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=500,
                )
                total_nfev += res.nfev
                if res.cost < penalty_cost:
                    T_fitted.append(T_K_i)
                    kij_fitted.append(float(res.x[0]))
                    cost_fitted.append(100.0 * float(np.mean(np.abs(res.fun))))
                    fitted_point_meta.append((T_csv_i, P_i, x1_i, y1_i))
            except Exception:
                continue

        if len(T_fitted) == 0:
            raise RuntimeError("No points converged. Try relaxing kij_bounds.")

        T_fitted_arr = np.array(T_fitted)
        kij_fitted_arr = np.array(kij_fitted)
        kij_coeffs, _ = _fit_kij_polynomial(
            T_fitted_arr, kij_fitted_arr, np.array(cost_fitted), kij_order, kij_t_ref
        )

        # Post-poly ARD: re-evaluate at polynomial k_ij
        ard_poly = []
        for (T_csv_i, P_i, x1_i, y1_i), T_K_i in zip(fitted_point_meta, T_fitted_arr):
            kij_poly = _kij_at_T(kij_coeffs, float(T_K_i), kij_t_ref)
            try:
                r = _residuals_vle_point(
                    [kij_poly], T_csv_i, P_i, x1_i, y1_i,
                    record1, record2, temperature_unit, pressure_unit,
                )
                ard_poly.append(100.0 * float(np.mean(np.abs(r))))
            except Exception:
                pass
        ard_poly_arr = np.array(ard_poly)

        data["T_kij"] = T_fitted_arr
        data["kij_pointwise"] = kij_fitted_arr
        data["ard_pointwise"] = np.array(cost_fitted)
        data["ard_pointwise_poly"] = ard_poly_arr

        meaningful = ard_poly_arr[ard_poly_arr > 0.01]
        ard = float(meaningful.mean()) if len(meaningful) > 0 else float(np.mean(ard_poly_arr))

        poly_resid_vals = kij_fitted_arr - np.array(
            [_kij_at_T(kij_coeffs, T, kij_t_ref) for T in T_fitted_arr]
        )
        poly_result = SimpleNamespace(
            x=kij_coeffs,
            fun=poly_resid_vals,
            cost=float(np.sum(poly_resid_vals**2)) / 2.0,
            success=True,
            nfev=total_nfev,
            message="Point-wise VLE fitting completed",
        )
        eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))
        return BinaryFitResult(
            kij_coeffs=kij_coeffs,
            kij_t_ref=kij_t_ref,
            id1=id1,
            id2=id2,
            equilibrium_type="vle",
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

    # --- Global polynomial fit (default) -------------------------------------
    n_rows = len(T_arr)
    n_resid = n_rows * (2 if has_y1 else 1)

    def fun(coeffs: np.ndarray) -> np.ndarray:
        resids = np.empty(n_resid)
        # Group rows by unique kij to avoid rebuilding EOS for every row.
        # For kij_order=0 this means 1 EOS build per fun() call.
        kij_per_row = np.array([_kij_at_T(coeffs, float(T_arr[i]), kij_t_ref) for i in range(n_rows)])
        unique_kijs = np.unique(kij_per_row)
        eos_map: dict[float, object] = {}
        for kij_val in unique_kijs:
            try:
                eos_map[kij_val] = _build_binary_eos(record1, record2, float(kij_val))
            except Exception:
                eos_map[kij_val] = None

        r = 0
        for i in range(n_rows):
            T_i = float(T_arr[i])
            P_i = float(P_arr[i])
            x1_i = float(x1_arr[i])
            eos = eos_map[kij_per_row[i]]
            n_r = 2 if has_y1 else 1
            if eos is None:
                resids[r:r + n_r] = 1.0
            else:
                try:
                    bp = _bubble_point(eos, T_i, x1_i, P_i, temperature_unit, pressure_unit)
                    P_pred = bp.liquid.pressure() / pressure_unit
                    resids[r] = (P_pred - P_i) / P_i
                    if has_y1:
                        resids[r + 1] = float(bp.vapor.molefracs[0]) - float(y1_arr[i])
                except Exception:
                    resids[r:r + n_r] = 1.0
            r += n_r
        return resids

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

    # ARD on pressure — reuse fun() residuals to avoid extra EOS builds
    final_resids = fun(kij_coeffs)
    P_resids = final_resids[::2] if has_y1 else final_resids
    valid = np.abs(P_resids) < 0.99  # sentinel 1.0 → failed
    P_pred_all = P_arr[valid] * (1.0 + P_resids[valid])
    P_data_all = P_arr[valid]

    if len(P_pred_all) > 0:
        ard = 100.0 * float(np.mean(np.abs(P_resids[valid])))
    else:
        ard = float("nan")

    return BinaryFitResult(
        kij_coeffs=kij_coeffs,
        kij_t_ref=kij_t_ref,
        id1=id1,
        id2=id2,
        equilibrium_type="vle",
        eos=eos_ref,
        data=data,
        data_full=data_full,
        ard=ard,
        scipy_result=result,
        time_elapsed=time_elapsed,
        t_filter_min_K=float(t_min / si.KELVIN) if t_min is not None else float("nan"),
        t_filter_max_K=float(t_max / si.KELVIN) if t_max is not None else float("nan"),
        _record1=record1,
        _record2=record2,
    )


def _bubble_point(eos, T_K: float, x1: float, P_guess: float, temperature_unit, pressure_unit):
    """Compute bubble point with experimental P as warm-start."""
    import feos

    return feos.PhaseEquilibrium.bubble_point(
        eos,
        T_K * temperature_unit,
        np.array([x1, 1.0 - x1]),
        tp_init=P_guess * pressure_unit,
    )


def _residuals_vle_point(
    kij_arr, T_csv: float, P_i: float, x1_i: float, y1_i: "float | None",
    record1, record2, temperature_unit, pressure_unit,
    relative_residuals: bool = True,
) -> np.ndarray:
    """Residual vector for a single VLE data point.

    Returns [P_error] when y1 is absent, [P_error, abs_y1_error] when y1 is
    present. P_error is relative ((P_pred-P)/P) when relative_residuals=True,
    absolute (P_pred-P) otherwise. Returns a penalty vector of ones on failure.
    """
    n_r = 2 if y1_i is not None else 1
    penalty = np.ones(n_r)
    try:
        eos = _build_binary_eos(record1, record2, float(kij_arr[0]))
        bp = _bubble_point(eos, T_csv, x1_i, P_i, temperature_unit, pressure_unit)
        P_pred = bp.liquid.pressure() / pressure_unit
        P_err = (P_pred - P_i) / P_i if relative_residuals else (P_pred - P_i)
        resids = [P_err]
        if y1_i is not None:
            resids.append(float(bp.vapor.molefracs[0]) - y1_i)
        return np.array(resids)
    except Exception:
        return penalty
