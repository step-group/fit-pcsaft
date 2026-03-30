"""LLE k_ij fitting from liquid-liquid equilibrium data."""
import time
from pathlib import Path

import feos
import numpy as np
import si_units as si
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _build_binary_eos,
    _kij_at_T,
    _load_lle_csv,
    _load_pure_records,
    _make_binary_jac_fn,
)
from fit_pcsaft._binary.result import BinaryFitResult

# 5 evenly-spaced global-composition candidates for one-sided tieline rows
_Z_CANDIDATES = np.linspace(0.1, 0.9, 5)


def _bubble_P(eos, T_si, z1: float):
    """Bubble-point pressure at mole fraction z1. Returns None on failure."""
    try:
        z = np.array([z1, 1.0 - z1])
        bp = feos.PhaseEquilibrium.bubble_point(eos, T_si, z)
        return bp.liquid.pressure()
    except Exception:
        return None


def _is_lle(flash) -> bool:
    """True when both phases of a tp_flash result are liquid-like (density > 50 kg/m³)."""
    try:
        rho_a = flash.liquid.mass_density() / (si.KILOGRAM / si.METER**3)
        rho_b = flash.vapor.mass_density() / (si.KILOGRAM / si.METER**3)
        return min(rho_a, rho_b) > 50.0
    except Exception:
        return False


def _find_z(eos0, T_si, xi_I: float, xi_II: "float | None") -> float:
    """Pre-compute the global composition z1 for one data point.

    Full tieline (both phases known): z1 = midpoint.
    One-sided (xi_II is None or NaN): try 5 evenly-spaced candidates at the
    initial EOS; fall back to 0.5 if none yield a valid LLE flash.
    """
    if xi_II is not None and not np.isnan(xi_II):
        return 0.5 * (xi_I + xi_II)

    for z1 in _Z_CANDIDATES:
        P = _bubble_P(eos0, T_si, z1)
        if P is None:
            continue
        try:
            flash = feos.PhaseEquilibrium.tp_flash(
                eos0, T_si, P, np.array([z1, 1.0 - z1]) * si.MOL
            )
            if _is_lle(flash):
                return float(z1)
        except Exception:
            continue
    return 0.5  # fallback


def fit_kij_lle(
    id1: str,
    id2: str,
    lle_path: "Path | str",
    params_path: "Path | str",
    kij_order: int = 0,
    kij_t_ref: float = 293.15,
    kij_bounds: tuple = (-0.5, 0.5),
    temperature_unit=si.KELVIN,
    temperature_offset: float = 0.0,
    phases: "tuple[str, ...] | None" = None,
    composition: str = "molefrac",
    t_min: "si.SIObject | None" = None,
    t_max: "si.SIObject | None" = None,
    scipy_kwargs: "dict | None" = None,
) -> BinaryFitResult:
    """Fit binary interaction parameter k_ij from LLE data.

    The CSV is read **by column position** (header names are ignored)::

        2 columns  →  (T, x1_I)            one-sided tieline
        3+ columns →  (T, x1_I, x1_II)     full tieline

    For each data point the global flash composition is:

    - **Full tieline**: midpoint of the two experimental compositions.
    - **One-sided**: the first of five evenly-spaced candidates
      [0.1, 0.3, 0.5, 0.7, 0.9] that yields a valid LLE flash with the
      initial EOS (k_ij = 0); falls back to 0.5.

    The flash pressure is the bubble-point pressure of the mixture at the
    global composition (recomputed each evaluation); falls back to 1 bar when
    the bubble-point calculation fails.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    lle_path : Path | str
        CSV file — column order determines meaning, not header names.
    params_path : Path | str
        Feos-compatible JSON parameter file.
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, 2=quadratic, 3=cubic.
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K]. Default: 293.15 K.
    kij_bounds : tuple
        (lower, upper) bounds for the constant term k_ij0.
    temperature_unit : si.SIObject
        Unit of the first CSV column (default: ``si.KELVIN``).
    temperature_offset : float
        Added to every T value before applying ``temperature_unit`` (default: 0.0).
        Use ``273.15`` to convert °C → K when ``temperature_unit=si.KELVIN``.
    phases : tuple[str, ...] | None
        Restrict which phases contribute to the residuals.  Pass ``("I",)`` or
        ``("II",)`` to use only one phase; ``None`` (default) uses all available.
    composition : str
        ``"molefrac"`` (default) or ``"massfrac"``.  When ``"massfrac"``, columns
        2 and 3 are treated as mass fractions and converted to mole fractions
        using the molar masses from the params JSON.
    t_min : si.SIObject | None
        Lower temperature bound. Rows with T < t_min are excluded.
    t_max : si.SIObject | None
        Upper temperature bound. Rows with T > t_max are excluded.
    scipy_kwargs : dict | None
        Overrides for ``scipy.optimize.least_squares`` keyword arguments.

    Returns
    -------
    BinaryFitResult
    """
    record1, record2 = _load_pure_records(params_path, id1, id2)

    # --- Load CSV by column position -----------------------------------------
    T_raw, x1_I_raw, x1_II_raw = _load_lle_csv(lle_path)

    # Massfrac → molefrac conversion
    if composition == "massfrac":
        M1 = float(record1.molarweight)
        M2 = float(record2.molarweight)

        def w2x(w: np.ndarray) -> np.ndarray:
            return (w / M1) / (w / M1 + (1.0 - w) / M2)

        x1_I_raw = w2x(x1_I_raw)
        if x1_II_raw is not None:
            x1_II_raw = w2x(x1_II_raw)
    elif composition != "molefrac":
        raise ValueError(f"composition must be 'molefrac' or 'massfrac', got {composition!r}")

    T_arr = T_raw + temperature_offset

    # Capture full data (after massfrac conversion, before filtering)
    T_arr_full = T_arr.copy()
    x1_I_raw_full = x1_I_raw.copy()
    x1_II_raw_full = x1_II_raw.copy() if x1_II_raw is not None else None

    # --- Temperature filter --------------------------------------------------
    if t_min is not None or t_max is not None:
        mask = np.ones(len(T_arr), dtype=bool)
        if t_min is not None:
            mask &= T_arr >= float(t_min / temperature_unit)
        if t_max is not None:
            mask &= T_arr <= float(t_max / temperature_unit)
        T_raw = T_raw[mask]
        T_arr = T_arr[mask]
        x1_I_raw = x1_I_raw[mask]
        if x1_II_raw is not None:
            x1_II_raw = x1_II_raw[mask]

    n_rows = len(T_arr)

    has_phase_I = True
    has_phase_II = x1_II_raw is not None

    if phases is not None:
        has_phase_I = has_phase_I and "I" in phases
        has_phase_II = has_phase_II and "II" in phases
        if not has_phase_I and not has_phase_II:
            raise ValueError(f"phases={phases!r} excluded all available phases from the CSV")

    x1_I_arr = x1_I_raw if has_phase_I else None
    x1_II_arr = x1_II_raw if has_phase_II else None

    data: dict = {"T": T_raw}
    if x1_I_arr is not None:
        data["x1_I"] = x1_I_arr
    if x1_II_arr is not None:
        data["x1_II"] = x1_II_arr

    data_full: dict = {"T": T_arr_full, "x1_I": x1_I_raw_full}
    if x1_II_raw_full is not None:
        data_full["x1_II"] = x1_II_raw_full

    # --- Pre-compute global compositions z1 for each row ---------------------
    eos0 = _build_binary_eos(record1, record2, 0.0)
    z_arr = np.empty(n_rows)
    for i in range(n_rows):
        T_si = T_arr[i] * temperature_unit
        xi_I = float(x1_I_arr[i]) if x1_I_arr is not None else 0.5
        xi_II = float(x1_II_arr[i]) if x1_II_arr is not None else None
        z_arr[i] = _find_z(eos0, T_si, xi_I, xi_II)

    # --- Build cost function -------------------------------------------------
    n_phases = int(has_phase_I) + int(has_phase_II)
    n_resid = n_rows * n_phases * 2

    def fun(coeffs: np.ndarray) -> np.ndarray:
        resids = np.empty(n_resid)
        kij_per_row = np.array(
            [_kij_at_T(coeffs, float(T_arr[i]), kij_t_ref) for i in range(n_rows)]
        )
        unique_kijs = np.unique(kij_per_row)
        eos_map: dict = {}
        for kij_val in unique_kijs:
            try:
                eos_map[kij_val] = _build_binary_eos(record1, record2, float(kij_val))
            except Exception:
                eos_map[kij_val] = None

        r = 0
        for i in range(n_rows):
            T_si = T_arr[i] * temperature_unit
            z1 = z_arr[i]
            z = np.array([z1, 1.0 - z1])
            eos = eos_map[kij_per_row[i]]

            if eos is None:
                resids[r : r + n_phases * 2] = 1.0
                r += n_phases * 2
                continue

            # Saturation pressure: bubble point at global composition z
            P_si = _bubble_P(eos, T_si, z1)
            P_si = P_si if P_si is not None else si.BAR

            try:
                flash = feos.PhaseEquilibrium.tp_flash(eos, T_si, P_si, z * si.MOL)
                x_pred = sorted(
                    [float(flash.liquid.molefracs[0]), float(flash.vapor.molefracs[0])]
                )
                x_exp = []
                if has_phase_I:
                    v = float(x1_I_arr[i])
                    if not np.isnan(v):
                        x_exp.append(v)
                if has_phase_II:
                    v = float(x1_II_arr[i])
                    if not np.isnan(v):
                        x_exp.append(v)
                j = 0
                for x_e in sorted(x_exp):
                    xp = x_pred[j]
                    resids[r + j * 2] = (xp - x_e) / max(x_e, 1e-10)
                    resids[r + j * 2 + 1] = ((1 - xp) - (1 - x_e)) / max(1 - x_e, 1e-10)
                    j += 1
                for k in range(j * 2, n_phases * 2):
                    resids[r + k] = 0.0
            except Exception:
                resids[r : r + n_phases * 2] = 1.0
            r += n_phases * 2
        return resids

    # --- Optimize ------------------------------------------------------------
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

    # --- ARD -----------------------------------------------------------------
    abs_devs: list[float] = []
    for i in range(n_rows):
        T_i = float(T_arr[i])
        T_si = T_i * temperature_unit
        z1 = z_arr[i]
        z = np.array([z1, 1.0 - z1])
        kij = _kij_at_T(kij_coeffs, T_i, kij_t_ref)
        try:
            eos = _build_binary_eos(record1, record2, kij)
            P_si = _bubble_P(eos, T_si, z1)
            P_si = P_si if P_si is not None else si.BAR
            flash = feos.PhaseEquilibrium.tp_flash(eos, T_si, P_si, z * si.MOL)
            x_pred = sorted(
                [float(flash.liquid.molefracs[0]), float(flash.vapor.molefracs[0])]
            )
            x_exp = []
            if has_phase_I:
                v = float(x1_I_arr[i])
                if not np.isnan(v):
                    x_exp.append(v)
            if has_phase_II:
                v = float(x1_II_arr[i])
                if not np.isnan(v):
                    x_exp.append(v)
            for j, x_e in enumerate(sorted(x_exp)):
                xp = x_pred[j]
                abs_devs.append(abs(xp - x_e) / max(x_e, 1e-10))
                abs_devs.append(abs((1 - xp) - (1 - x_e)) / max(1 - x_e, 1e-10))
        except Exception:
            pass

    ard = 100.0 * float(np.mean(abs_devs)) if abs_devs else float("nan")

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
        scipy_result=result,
        time_elapsed=time_elapsed,
        t_filter_min_K=float(t_min / si.KELVIN) if t_min is not None else float("nan"),
        t_filter_max_K=float(t_max / si.KELVIN) if t_max is not None else float("nan"),
    )
