"""LLE k_ij fitting from liquid-liquid equilibrium data."""
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


def fit_kij_lle(
    id1: str,
    id2: str,
    lle_path: "Path | str",
    params_path: "Path | str",
    kij_order: int = 0,
    kij_t_ref: float = 293.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    phases: "tuple[str, ...] | None" = None,
    scipy_kwargs: "dict | None" = None,
) -> BinaryFitResult:
    """Fit binary interaction parameter k_ij from LLE data.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    lle_path : Path | str
        CSV file with column T (required) and at least one of x1_I, x1_II.
        Optional column P [bar]; defaults to 1 bar if absent.
    params_path : Path | str
        Feos-compatible JSON parameter file (output of FitResult.to_json).
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, 2=quadratic, 3=cubic.
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K]. Default: 293.15 K.
    kij_bounds : tuple
        (lower, upper) bounds for the constant term k_ij0.
    temperature_unit : si.SIObject
        Unit of T column in CSV (default: K).
    scipy_kwargs : dict | None
        Overrides for scipy.optimize.least_squares keyword arguments.

    Returns
    -------
    BinaryFitResult
    """
    import feos

    record1, record2 = _load_pure_records(params_path, id1, id2)
    data = _load_binary_csv(lle_path)

    T_arr = data["T"]
    has_phase_I = "x1_I" in data
    has_phase_II = "x1_II" in data
    if not has_phase_I and not has_phase_II:
        raise ValueError("LLE CSV must contain at least one of: x1_I, x1_II")

    if phases is not None:
        has_phase_I = has_phase_I and "I" in phases
        has_phase_II = has_phase_II and "II" in phases
        if not has_phase_I and not has_phase_II:
            raise ValueError(f"phases={phases!r} excluded all available phases from the CSV")

    x1_I_arr = data["x1_I"] if has_phase_I else None
    x1_II_arr = data["x1_II"] if has_phase_II else None
    P_arr = data.get("P", np.ones(len(T_arr)))  # default 1 bar

    n_rows = len(T_arr)
    n_phases = int(has_phase_I) + int(has_phase_II)
    n_resid = n_rows * n_phases * 2  # x1 and x2 for each phase

    def fun(coeffs: np.ndarray) -> np.ndarray:
        resids = np.empty(n_resid)
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
            P_i = float(P_arr[i]) if hasattr(P_arr, "__len__") else float(P_arr)
            eos = eos_map[kij_per_row[i]]
            if eos is None:
                resids[r:r + n_phases] = 1.0
            else:
                try:
                    flash = feos.PhaseEquilibrium.tp_flash(
                        eos,
                        T_i * temperature_unit,
                        P_i * si.BAR,
                        np.array([0.5, 0.5]) * si.MOL,
                    )
                    x_pred = sorted(
                        [float(flash.liquid.molefracs[0]), float(flash.vapor.molefracs[0])]
                    )
                    x_exp = []
                    if has_phase_I:
                        x_exp.append(float(x1_I_arr[i]))
                    if has_phase_II:
                        x_exp.append(float(x1_II_arr[i]))
                    for j, x_e in enumerate(sorted(x_exp)):
                        xp = x_pred[j]
                        resids[r + j * 2]     = (xp - x_e) / max(x_e, 1e-10)
                        resids[r + j * 2 + 1] = ((1 - xp) - (1 - x_e)) / max(1 - x_e, 1e-10)
                except Exception:
                    resids[r:r + n_phases * 2] = 1.0
            r += n_phases * 2
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

    # ARD on mole fractions
    abs_devs = []
    x_exp_vals = []
    for i in range(n_rows):
        T_i = float(T_arr[i])
        P_i = float(P_arr[i]) if hasattr(P_arr, "__len__") else float(P_arr)
        kij = _kij_at_T(kij_coeffs, T_i, kij_t_ref)
        try:
            eos = _build_binary_eos(record1, record2, kij)
            flash = feos.PhaseEquilibrium.tp_flash(
                eos,
                T_i * temperature_unit,
                P_i * si.BAR,
                np.array([0.5, 0.5]) * si.MOL,
            )
            x_pred = sorted(
                [float(flash.liquid.molefracs[0]), float(flash.vapor.molefracs[0])]
            )
            x_exp = []
            if has_phase_I:
                x_exp.append(float(x1_I_arr[i]))
            if has_phase_II:
                x_exp.append(float(x1_II_arr[i]))
            x_exp_sorted = sorted(x_exp)
            for j, x_e in enumerate(x_exp_sorted):
                xp = x_pred[j]
                abs_devs.append(abs(xp - x_e) / max(x_e, 1e-10))
                abs_devs.append(abs((1 - xp) - (1 - x_e)) / max(1 - x_e, 1e-10))
                x_exp_vals.append(x_e)
        except Exception:
            pass

    if abs_devs:
        ard = 100.0 * float(np.mean(abs_devs))
    else:
        ard = float("nan")

    return BinaryFitResult(
        kij_coeffs=kij_coeffs,
        kij_t_ref=kij_t_ref,
        id1=id1,
        id2=id2,
        equilibrium_type="lle",
        eos=eos_ref,
        data=data,
        ard=ard,
        scipy_result=result,
        time_elapsed=time_elapsed,
    )
