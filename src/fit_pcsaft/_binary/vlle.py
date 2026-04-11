"""VLLE k_ij fitting from three-phase (heteroazeotrope) equilibrium data.

Each data row is a measured three-phase point:
  - T, P: temperature and pressure of the heteroazeotrope
  - x1_I, x1_II (optional): the two liquid-phase compositions
  - y1 (optional): the vapor-phase composition

Residuals per point:
  - Relative T error:        (T_pred − T_exp) / T_exp          (always)
  - Absolute composition errors: x1_I_pred − x1_I_exp, etc.   (when present)
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _apply_induced_association,
    _build_binary_eos,
    _kij_at_T,
    _load_pure_records,
    _make_binary_jac_fn,
)
from fit_pcsaft._binary.result import BinaryFitResult
from fit_pcsaft._csv import SCHEMA_VLLE, load_csv

_N_KIJ_SCAN = 13


def fit_kij_vlle(
    id1: str,
    id2: str,
    vlle_path: "Path | str",
    params_path: "Path | str | list[Path | str]",
    kij_order: int = 0,
    kij_t_ref: float = 298.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    pressure_unit=si.KILO * si.PASCAL,
    t_min: "si.SIObject | None" = None,
    t_max: "si.SIObject | None" = None,
    scipy_kwargs: "dict | None" = None,
    induced_assoc: bool = False,
) -> BinaryFitResult:
    """Fit binary interaction parameter k_ij from VLLE heteroazeotrope data.

    Each CSV row is a measured three-phase equilibrium point (heteroazeotrope).
    The model predicts the heteroazeotrope via
    ``feos.PhaseEquilibrium.heteroazeotrope`` and the residuals are:

    - Relative temperature error ``(T_pred − T_exp) / T_exp`` — always.
    - Absolute composition errors for x1_I, x1_II, y1 — when the
      corresponding columns are present in the CSV.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    vlle_path : Path | str
        CSV file with columns: T, P (required); x1_I, x1_II, y1 (optional).
    params_path : Path | str | list
        Feos-compatible JSON parameter file(s).
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, 2=quadratic.
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K].
    kij_bounds : tuple
        (lower, upper) bounds for the constant k_ij term.
    temperature_unit : si.SIObject
        Unit of the T column in the CSV (default: K).
    pressure_unit : si.SIObject
        Unit of the P column in the CSV (default: kPa).
    t_min, t_max : si.SIObject | None
        Optional temperature filter applied before fitting.
    scipy_kwargs : dict | None
        Overrides for ``scipy.optimize.least_squares``.
    induced_assoc : bool
        Apply the induced-association mixing rule (see ``fit_kij_vle``).

    Returns
    -------
    BinaryFitResult
        ``equilibrium_type`` is ``"vlle"``.
    """
    record1, record2 = _load_pure_records(params_path, id1, id2)
    if induced_assoc:
        record1, record2 = _apply_induced_association(record1, record2)

    data = load_csv(vlle_path, SCHEMA_VLLE)

    # --- Temperature filter ---------------------------------------------------
    if t_min is not None or t_max is not None:
        mask = np.ones(len(data["T"]), dtype=bool)
        if t_min is not None:
            mask &= data["T"] >= float(t_min / temperature_unit)
        if t_max is not None:
            mask &= data["T"] <= float(t_max / temperature_unit)
        data = {k: v[mask] for k, v in data.items()}

    T_arr = data["T"].astype(float)
    P_arr = data["P"].astype(float)
    has_xI  = "x1_I"  in data
    has_xII = "x1_II" in data
    has_y1  = "y1"    in data

    t_scale = float(temperature_unit / si.KELVIN)
    p_scale = float(pressure_unit / si.PASCAL)

    t0 = time.perf_counter()

    # Number of residuals per point: 1 (T) + optional compositions
    n_comp = int(has_xI) + int(has_xII) + int(has_y1)
    n_per_point = 1 + n_comp

    def _residuals_point(kij_val: float, i: int) -> np.ndarray:
        """Residual vector for a single VLLE data point."""
        T_K = float(T_arr[i]) * t_scale
        P_si = float(P_arr[i]) * p_scale * si.PASCAL
        penalty = np.ones(n_per_point)
        try:
            eos = _build_binary_eos(record1, record2, kij_val)
            # Use experimental compositions as warm-start if available
            x_I_init  = float(data["x1_I"][i])  if has_xI  else 0.01
            x_II_init = float(data["x1_II"][i]) if has_xII else 0.90
            import feos as _feos
            ha = _feos.PhaseEquilibrium.heteroazeotrope(
                eos, P_si,
                x_init=(float(np.clip(x_I_init,  1e-6, 1.0 - 1e-6)),
                        float(np.clip(x_II_init, 1e-6, 1.0 - 1e-6))),
                tp_init=T_K * si.KELVIN,
            )
            T_pred  = float(ha.vapor.temperature / si.KELVIN)
            x1_I_pred  = float(ha.liquid1.molefracs[0])
            x1_II_pred = float(ha.liquid2.molefracs[0])
            y1_pred    = float(ha.vapor.molefracs[0])
            # Ensure phase I < phase II
            if x1_I_pred > x1_II_pred:
                x1_I_pred, x1_II_pred = x1_II_pred, x1_I_pred

            resids = [(T_pred - T_K) / T_K]
            if has_xI:
                resids.append(x1_I_pred  - float(data["x1_I"][i]))
            if has_xII:
                resids.append(x1_II_pred - float(data["x1_II"][i]))
            if has_y1:
                resids.append(y1_pred    - float(data["y1"][i]))
            return np.array(resids)
        except Exception:
            return penalty

    # --- Per-point k_ij fitting → polynomial ----------------------------------
    n_rows = len(T_arr)
    T_fitted, kij_fitted, ard_fitted = [], [], []
    total_nfev = 0

    for i in range(n_rows):
        T_K_i = float(T_arr[i]) * t_scale

        def resid_fn(kij_arr, _i=i):
            return _residuals_point(float(kij_arr[0]), _i)

        # Coarse scan for best initial guess
        kij_scan = np.linspace(kij_bounds[0], kij_bounds[1], _N_KIJ_SCAN)
        best_x0, best_cost = 0.0, np.inf
        for kv in kij_scan:
            try:
                c = 0.5 * float(np.sum(resid_fn([kv]) ** 2))
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
            penalty_cost = 0.5 * n_per_point * 0.99
            if res.cost < penalty_cost:
                T_fitted.append(T_K_i)
                kij_fitted.append(float(res.x[0]))
                ard_fitted.append(100.0 * float(np.mean(np.abs(res.fun[:1]))))  # T ARD
        except Exception:
            continue

    if len(T_fitted) == 0:
        raise RuntimeError("No VLLE points converged. Try relaxing kij_bounds.")

    # --- Polynomial fit to (T, k_ij) pairs -----------------------------------
    effective_order = min(kij_order, len(T_fitted) - 1)
    T_arr_fit = np.array(T_fitted)
    kij_arr_fit = np.array(kij_fitted)
    dT = T_arr_fit - kij_t_ref

    ols_rev = np.polyfit(dT, kij_arr_fit, effective_order)
    x0_poly = ols_rev[::-1]

    if effective_order == 0 or len(T_fitted) == 1:
        kij_coeffs = x0_poly
        poly_resid = kij_arr_fit - sum(c * dT**j for j, c in enumerate(kij_coeffs))
        poly_result = SimpleNamespace(
            x=kij_coeffs, fun=poly_resid,
            cost=float(np.sum(poly_resid**2)) / 2.0,
            success=True, nfev=total_nfev,
            message="VLLE per-point fitting completed",
        )
    else:
        def _poly_resid(coeffs):
            return sum(c * dT**j for j, c in enumerate(coeffs)) - kij_arr_fit

        rob = least_squares(
            _poly_resid, x0_poly,
            loss="cauchy", f_scale=0.01,
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        total_nfev += rob.nfev
        kij_coeffs = rob.x
        poly_result = SimpleNamespace(
            x=kij_coeffs, fun=rob.fun, cost=rob.cost,
            success=rob.success, nfev=total_nfev,
            message="VLLE per-point fitting completed",
        )

    ard = float(np.mean(ard_fitted))
    eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))

    data["T_kij"]       = T_arr_fit
    data["kij_pointwise"] = kij_arr_fit
    data["ard_pointwise"] = np.array(ard_fitted)

    return BinaryFitResult(
        kij_coeffs=kij_coeffs,
        kij_t_ref=kij_t_ref,
        id1=id1,
        id2=id2,
        equilibrium_type="vlle",
        eos=eos_ref,
        data=data,
        data_full={k: v.copy() for k, v in data.items()
                   if k not in ("T_kij", "kij_pointwise", "ard_pointwise")},
        ard=ard,
        scipy_result=poly_result,
        time_elapsed=time.perf_counter() - t0,
        t_filter_min_K=float(t_min / si.KELVIN) if t_min is not None else float("nan"),
        t_filter_max_K=float(t_max / si.KELVIN) if t_max is not None else float("nan"),
        _record1=record1,
        _record2=record2,
    )
