"""VLE k_ij fitting from bubble-point data."""
import time
from pathlib import Path

import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _build_binary_eos,
    _kij_at_T,
    _load_binary_csv,
    _load_pure_records,
    _make_binary_jac_fn,
)
from fit_pcsaft._binary.result import BinaryFitResult


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
    scipy_kwargs: "dict | None" = None,
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
    scipy_kwargs : dict | None
        Overrides for scipy.optimize.least_squares keyword arguments.

    Returns
    -------
    BinaryFitResult
    """
    record1, record2 = _load_pure_records(params_path, id1, id2)
    data = _load_binary_csv(vle_path)

    T_arr = data["T"]
    P_arr = data["P"]
    x1_arr = data["x1"]
    has_y1 = "y1" in data
    y1_arr = data["y1"] if has_y1 else None

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

    t0 = time.perf_counter()
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
        ard=ard,
        scipy_result=result,
        time_elapsed=time_elapsed,
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
