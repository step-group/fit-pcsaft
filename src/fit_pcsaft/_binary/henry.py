"""Henry's law k_ij fitting from Henry's law constant data."""
import time
from pathlib import Path

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
from fit_pcsaft._csv import SCHEMA_HENRY, load_csv


def fit_kij_henry(
    id1: str,
    id2: str,
    henry_path: "Path | str",
    params_path: "Path | str",
    kij_order: int = 0,
    kij_t_ref: float = 293.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    henry_unit=si.MEGA * si.PASCAL,
    scipy_kwargs: "dict | None" = None,
    induced_assoc: bool = False,
) -> BinaryFitResult:
    """Fit binary interaction parameter k_ij from Henry's law constant data.

    Component 1 is treated as the solute, component 2 as the solvent
    (convention of feos.State.henrys_law_constant_binary).

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
        id1 is the solute, id2 is the solvent.
    henry_path : Path | str
        CSV file with columns: T, H.
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
    henry_unit : si.SIObject or "molfrac"
        Unit of H column in CSV (default: MPa).
        Pass ``"molfrac"`` if your data is the dimensionless K = y1/x1
        at infinite dilution; the EOS prediction is then converted via
        K = H_feos / P_vap_solvent(T).
    scipy_kwargs : dict | None
        Overrides for scipy.optimize.least_squares keyword arguments.
    induced_assoc : bool
        If True, apply the induced-association mixing rule. Requires exactly one
        self-associating component (with epsilon_k_ab > 0). The non-associating
        component is assigned epsilon_k_ab = 0 and kappa_ab copied from the
        self-associating component, with na = nb = 1 (2B scheme).

    Returns
    -------
    BinaryFitResult
    """
    import feos

    record1, record2 = _load_pure_records(params_path, id1, id2)
    if induced_assoc:
        record1, record2 = _apply_induced_association(record1, record2)
    data = load_csv(henry_path, SCHEMA_HENRY)
    T_arr = data["T"]
    H_arr = data["H"]
    n_rows = len(T_arr)

    use_molfrac = henry_unit == "molfrac"
    if use_molfrac:
        # Build pure solvent EOS once for vapor pressure calculations
        eos_solvent = feos.EquationOfState.pcsaft(
            feos.Parameters.new_pure(record2)
        )
        # Cache solvent vapor pressures at each T [Pa]
        p_vap_solvent = np.empty(n_rows)
        for i in range(n_rows):
            vp = feos.PhaseEquilibrium.vapor_pressure(
                eos_solvent, float(T_arr[i]) * temperature_unit
            )
            p_vap_solvent[i] = vp[0] / si.PASCAL

    def fun(coeffs: np.ndarray) -> np.ndarray:
        resids = np.empty(n_rows)
        kij_per_row = np.array([_kij_at_T(coeffs, float(T_arr[i]), kij_t_ref) for i in range(n_rows)])
        unique_kijs = np.unique(kij_per_row)
        eos_map: dict[float, object] = {}
        for kij_val in unique_kijs:
            try:
                eos_map[kij_val] = _build_binary_eos(record1, record2, float(kij_val))
            except Exception:
                eos_map[kij_val] = None

        for i in range(n_rows):
            T_i = float(T_arr[i])
            H_i = float(H_arr[i])
            eos = eos_map[kij_per_row[i]]
            if eos is None:
                resids[i] = 1.0
            else:
                try:
                    H_feos = feos.State.henrys_law_constant_binary(
                        eos, T_i * temperature_unit
                    )
                    if use_molfrac:
                        H_pred = H_feos / si.PASCAL / p_vap_solvent[i]
                    else:
                        H_pred = H_feos / henry_unit
                    resids[i] = (H_pred - H_i) / H_i
                except Exception:
                    resids[i] = 1.0
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

    final_resids = fun(kij_coeffs)
    valid = np.abs(final_resids) < 0.99
    ard = 100.0 * float(np.mean(np.abs(final_resids[valid]))) if valid.any() else float("nan")

    return BinaryFitResult(
        kij_coeffs=kij_coeffs,
        kij_t_ref=kij_t_ref,
        id1=id1,
        id2=id2,
        equilibrium_type="henry",
        eos=eos_ref,
        data=data,
        data_full=data,
        ard=ard,
        scipy_result=result,
        time_elapsed=time_elapsed,
        _solvent_record=record2,
    )
