"""Combined VLE + LLE k_ij fitting.

Two-stage approach:
  Stage 1: fit one k_ij per data point independently for VLE and LLE.
  Stage 2: fit a single k_ij(T) polynomial to all (T, k_ij) pairs combined.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _build_binary_eos,
    _kij_at_T,
    _load_pure_records,
)
from fit_pcsaft._binary.lle import fit_kij_lle
from fit_pcsaft._binary.result import BinaryFitResult
from fit_pcsaft._binary.vle import fit_kij_vle


def fit_kij_vle_lle(
    id1: str,
    id2: str,
    vle_path: "Path | str",
    lle_path: "Path | str",
    params_path: "Path | str | list[Path | str]",
    kij_order: int = 0,
    kij_t_ref: float = 293.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    pressure_unit=si.KILO * si.PASCAL,
    t_min_vle: "si.SIObject | None" = None,
    t_max_vle: "si.SIObject | None" = None,
    t_min_lle: "si.SIObject | None" = None,
    t_max_lle: "si.SIObject | None" = None,
    pressure: "si.SIObject" = 1.01325 * si.BAR,
    require_both_phases: bool = True,
    induced_assoc: bool = False,
) -> BinaryFitResult:
    """Fit k_ij using both VLE and LLE data simultaneously.

    Uses a two-stage approach:
    1. For each VLE data point and each LLE temperature, solve an independent
       1D least-squares problem to find the per-point k_ij.
    2. Fit a single k_ij(T) polynomial to all combined (T, k_ij) pairs.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    vle_path : Path | str
        CSV file with columns: T, P, x1 (required); y1 (optional).
    lle_path : Path | str
        CSV file with columns: T, x1_I (required); x1_II (optional).
    params_path : Path | str | list
        Feos-compatible JSON parameter file(s).
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, 2=quadratic.
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K].
    kij_bounds : tuple
        (lower, upper) bounds for k_ij at each data point.
    temperature_unit : si.SIObject
        Unit of T column in both CSVs (default: K).
    pressure_unit : si.SIObject
        Unit of P column in VLE CSV (default: kPa).
    t_min_vle, t_max_vle : si.SIObject | None
        Temperature filter for VLE data.
    t_min_lle, t_max_lle : si.SIObject | None
        Temperature filter for LLE data.
    pressure : si.SIObject
        Pressure for LLE tp_flash calculations (default: 1 bar).
    require_both_phases : bool
        If True (default), skip LLE temperatures where only one phase
        composition is available.
    induced_assoc : bool
        If True, apply the induced-association mixing rule.

    Returns
    -------
    BinaryFitResult
        ``equilibrium_type`` is ``"vle_lle"``.
        ``data["T_kij"]``, ``data["kij_pointwise"]``, ``data["source"]``
        contain the combined per-point fit results.
        ``data["ard_vle"]`` and ``data["ard_lle"]`` hold the per-type ARDs.
    """
    t0 = time.perf_counter()

    # --- Stage 1a: VLE per-point k_ij -----------------------------------------
    vle_res = fit_kij_vle(
        id1, id2, vle_path, params_path,
        kij_order=0,
        kij_t_ref=kij_t_ref,
        kij_bounds=kij_bounds,
        temperature_unit=temperature_unit,
        pressure_unit=pressure_unit,
        t_min=t_min_vle,
        t_max=t_max_vle,
        kij_per_point=True,
        induced_assoc=induced_assoc,
    )
    T_vle_K = vle_res.data["T_kij"]
    kij_vle = vle_res.data["kij_pointwise"]
    ard_vle = float(np.mean(vle_res.data["ard_pointwise"]))

    # --- Stage 1b: LLE per-point k_ij -----------------------------------------
    lle_res = fit_kij_lle(
        id1, id2, lle_path, params_path,
        kij_order=0,
        kij_t_ref=kij_t_ref,
        kij_bounds=kij_bounds,
        temperature_unit=temperature_unit,
        t_min=t_min_lle,
        t_max=t_max_lle,
        pressure=pressure,
        require_both_phases=require_both_phases,
        kij_per_point=True,
        induced_assoc=induced_assoc,
    )
    T_lle_K = lle_res.data["T_kij"]
    kij_lle = lle_res.data["kij_pointwise"]
    ard_lle = float(np.mean(lle_res.data["ard_pointwise"]))

    # --- Stage 2: polynomial fit to combined (T, k_ij) pairs ------------------
    T_combined = np.concatenate([T_vle_K, T_lle_K])
    kij_combined = np.concatenate([kij_vle, kij_lle])
    source = np.array(["vle"] * len(T_vle_K) + ["lle"] * len(T_lle_K))

    effective_order = min(kij_order, len(T_combined) - 1)
    dT = T_combined - kij_t_ref

    ols_rev = np.polyfit(dT, kij_combined, effective_order)
    x0_poly = ols_rev[::-1]  # lowest-order first

    total_nfev = vle_res.scipy_result.nfev + lle_res.scipy_result.nfev

    if effective_order == 0 or len(T_combined) == 1:
        kij_coeffs = x0_poly
        poly_resid = kij_combined - sum(
            c * dT**j for j, c in enumerate(kij_coeffs)
        )
        poly_result = SimpleNamespace(
            x=kij_coeffs,
            fun=poly_resid,
            cost=float(np.sum(poly_resid**2)) / 2.0,
            success=True,
            nfev=total_nfev,
            message="VLE+LLE two-stage fitting completed",
        )
    else:
        def _poly_resid(coeffs):
            pred = sum(c * dT**j for j, c in enumerate(coeffs))
            return pred - kij_combined

        rob = least_squares(
            _poly_resid, x0_poly,
            loss="cauchy", f_scale=0.01,
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        total_nfev += rob.nfev
        kij_coeffs = rob.x
        poly_result = SimpleNamespace(
            x=kij_coeffs,
            fun=rob.fun,
            cost=rob.cost,
            success=rob.success,
            nfev=total_nfev,
            message="VLE+LLE two-stage fitting completed",
        )

    # Combined ARD (weighted by number of points)
    n_vle = len(T_vle_K)
    n_lle = len(T_lle_K)
    ard = (ard_vle * n_vle + ard_lle * n_lle) / (n_vle + n_lle)

    record1, record2 = _load_pure_records(params_path, id1, id2)
    eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))

    return BinaryFitResult(
        kij_coeffs=kij_coeffs,
        kij_t_ref=kij_t_ref,
        id1=id1,
        id2=id2,
        equilibrium_type="vle_lle",
        eos=eos_ref,
        data={
            "T_kij": T_combined,
            "kij_pointwise": kij_combined,
            "source": source,
            "ard_pointwise": np.concatenate([
                vle_res.data["ard_pointwise"],
                lle_res.data["ard_pointwise"],
            ]),
            "ard_vle": np.array([ard_vle]),
            "ard_lle": np.array([ard_lle]),
        },
        data_full={},
        ard=ard,
        scipy_result=poly_result,
        time_elapsed=time.perf_counter() - t0,
        _record1=record1,
        _record2=record2,
    )
