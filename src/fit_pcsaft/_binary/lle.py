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
)
from fit_pcsaft._binary.result import BinaryFitResult

# 5 evenly-spaced global-composition candidates for one-sided tieline rows
_Z_CANDIDATES = np.linspace(0.1, 0.9, 5)

# Number of kij0 scan points for choosing the initial guess
_N_KIJ_SCAN = 9

# Pressure used for LLE diagram computations
_LLE_P = 1.0 * si.BAR


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


def _tp_flash_robust(eos, T_si, P_si, feed, prev_flash=None):
    """Attempt tp_flash with warm-start, falling back to cold start.

    Returns a PhaseEquilibrium on success, or None on failure.
    """
    attempts = [prev_flash, None] if prev_flash is not None else [None]
    for attempt_state in attempts:
        try:
            if attempt_state is not None:
                flash = feos.PhaseEquilibrium.tp_flash(
                    eos, T_si, P_si, feed, initial_state=attempt_state
                )
            else:
                flash = feos.PhaseEquilibrium.tp_flash(eos, T_si, P_si, feed)
            return flash
        except Exception:
            continue
    return None


def _lle_diagram(eos, T_arr_K, z1: float):
    """Compute an LLE phase diagram and return (x_I, x_II) interpolated at T_arr_K.

    Uses ``PhaseDiagram.lle`` (continuation method) which does not require a
    cold-start initial guess.  Returns (x_I, x_II) arrays of length len(T_arr_K)
    with NaN where interpolation fails, or (None, None) if the diagram call fails.

    x_I  = component-1 mole fraction in the lean (low x1) phase.
    x_II = component-1 mole fraction in the rich (high x1) phase.
    """
    T_min_K = float(T_arr_K.min())
    T_max_K = float(T_arr_K.max())
    # Expand range slightly so the diagram covers all data points.
    margin = max(1.0, 0.01 * (T_max_K - T_min_K))
    try:
        feed = np.array([z1, 1.0 - z1]) * si.MOL
        diag = feos.PhaseDiagram.lle(
            eos,
            _LLE_P,
            feed=feed,
            min_tp=(T_min_K - margin) * si.KELVIN,
            max_tp=(T_max_K + margin) * si.KELVIN,
            npoints=max(50, 4 * len(T_arr_K)),
        )
        T_diag = diag.liquid.temperature / si.KELVIN  # shape (npoints,)
        if len(T_diag) == 0:
            return None, None
        # liquid = denser phase (water-rich), vapor = lighter phase (toluene-rich)
        x_I_diag = diag.vapor.molefracs[:, 0]   # lean phase, comp-1
        x_II_diag = diag.liquid.molefracs[:, 0]  # rich phase, comp-1

        # Interpolate at experimental temperatures.
        x_I = np.interp(T_arr_K, T_diag, x_I_diag)
        x_II = np.interp(T_arr_K, T_diag, x_II_diag)
        return x_I, x_II
    except BaseException:
        return None, None


def _find_z(
    eos0, T_si, xi_I: float, xi_II: "float | None",
    prev_flash=None,
) -> "tuple[float, object | None]":
    """Pre-compute the global composition z1 for one data point.

    Full tieline (both phases known): z1 = midpoint; also attempts a flash to
    produce a warm-start hint for the next temperature point.
    One-sided (xi_II is None or NaN): try 5 evenly-spaced candidates; falls
    back to 0.5 if none yield a valid LLE flash.

    Returns (z1, flash_result) where flash_result can be passed as prev_flash
    to the next call for warm-starting.
    """
    if xi_II is not None and not np.isnan(xi_II):
        z1 = 0.5 * (xi_I + xi_II)
        P = _bubble_P(eos0, T_si, z1)
        if P is not None:
            z = np.array([z1, 1.0 - z1])
            flash = _tp_flash_robust(eos0, T_si, P, z * si.MOL, prev_flash)
            if flash is not None and _is_lle(flash):
                return z1, flash
        return z1, prev_flash  # keep propagating even if this flash failed

    for z1 in _Z_CANDIDATES:
        P = _bubble_P(eos0, T_si, z1)
        if P is None:
            continue
        z = np.array([z1, 1.0 - z1])
        flash = _tp_flash_robust(eos0, T_si, P, z * si.MOL, prev_flash)
        if flash is not None and _is_lle(flash):
            return float(z1), flash

    return 0.5, prev_flash  # fallback


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

    For each candidate k_ij the LLE phase diagram is computed via
    ``PhaseDiagram.lle`` (continuation method), which is robust to the
    cold-start convergence issues of plain ``tp_flash``.  The predicted
    compositions are interpolated at each experimental temperature and
    compared to the data.

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

    # --- Pre-compute global compositions z1, sorted by temperature -----------
    eos0 = _build_binary_eos(record1, record2, 0.0)
    z_arr = np.empty(n_rows)
    sort_idx = np.argsort(T_arr)  # ascending temperature order

    prev_flash = None
    for idx in sort_idx:
        T_si = T_arr[idx] * temperature_unit
        xi_I = float(x1_I_arr[idx]) if x1_I_arr is not None else 0.5
        xi_II = float(x1_II_arr[idx]) if x1_II_arr is not None else None
        z_arr[idx], prev_flash = _find_z(eos0, T_si, xi_I, xi_II, prev_flash)

    # Global feed composition for PhaseDiagram.lle: use the mean z1 across all rows.
    z1_global = float(np.mean(z_arr))

    # --- Build cost function using PhaseDiagram.lle --------------------------
    # n_phases active experimental columns
    n_phases = int(has_phase_I) + int(has_phase_II)
    n_resid = n_rows * n_phases * 2

    def fun(coeffs: np.ndarray) -> np.ndarray:
        resids = np.ones(n_resid)  # default penalty = 1.0

        # For kij_order > 0 the kij varies with T; we run a separate diagram
        # per unique kij.  For order=0 there is one diagram.
        kij_per_row = np.array(
            [_kij_at_T(coeffs, float(T_arr[i]), kij_t_ref) for i in range(n_rows)]
        )
        unique_kijs, inverse = np.unique(kij_per_row, return_inverse=True)

        # Build one LLE diagram per unique kij
        diag_cache: dict = {}
        for uid, kij_val in enumerate(unique_kijs):
            try:
                eos = _build_binary_eos(record1, record2, float(kij_val))
            except Exception:
                diag_cache[uid] = (None, None)
                continue
            x_I_pred, x_II_pred = _lle_diagram(eos, T_arr, z1_global)
            diag_cache[uid] = (x_I_pred, x_II_pred)

        for i in range(n_rows):
            r = i * n_phases * 2
            uid = int(inverse[i])
            x_I_pred, x_II_pred = diag_cache[uid]

            x_pred_list: list = []
            if has_phase_I and x_I_pred is not None:
                x_pred_list.append(float(x_I_pred[i]))
            if has_phase_II and x_II_pred is not None:
                x_pred_list.append(float(x_II_pred[i]))

            if not x_pred_list or any(np.isnan(v) for v in x_pred_list):
                # diagram failed or out of range — keep penalty 1.0
                continue

            x_exp_list: list = []
            if has_phase_I:
                v = float(x1_I_arr[i])
                if not np.isnan(v):
                    x_exp_list.append(v)
            if has_phase_II:
                v = float(x1_II_arr[i])
                if not np.isnan(v):
                    x_exp_list.append(v)

            j = 0
            for xp, x_e in zip(x_pred_list, x_exp_list):
                resids[r + j * 2] = (xp - x_e) / max(x_e, 1e-10)
                resids[r + j * 2 + 1] = ((1 - xp) - (1 - x_e)) / max(1 - x_e, 1e-10)
                j += 1
            for jj in range(j * 2, n_phases * 2):
                resids[r + jj] = 0.0

        return resids

    # --- Optimize ------------------------------------------------------------
    n_coeffs = kij_order + 1
    lb = [kij_bounds[0]] + [-0.01] * kij_order
    ub = [kij_bounds[1]] + [0.01] * kij_order

    ls_kwargs = {
        "method": "trf",
        "jac": "3-point",
        "diff_step": 0.01,
        "bounds": (lb, ub),
        "ftol": 1e-6,
        "xtol": 1e-6,
        "gtol": 1e-6,
        "max_nfev": 2000,
    }
    if scipy_kwargs:
        ls_kwargs.update(scipy_kwargs)

    # Coarse scan: evaluate fun at _N_KIJ_SCAN evenly-spaced kij0 values and
    # pick the starting point with the lowest initial cost.
    kij0_scan = np.linspace(kij_bounds[0], kij_bounds[1], _N_KIJ_SCAN)
    best_x0 = np.zeros(n_coeffs)
    best_scan_cost = np.inf
    for kij0 in kij0_scan:
        x0 = np.zeros(n_coeffs)
        x0[0] = kij0
        cost = 0.5 * np.sum(fun(x0) ** 2)
        if cost < best_scan_cost:
            best_scan_cost = cost
            best_x0 = x0.copy()

    t0 = time.perf_counter()
    result = least_squares(fun, best_x0, **ls_kwargs)
    time_elapsed = time.perf_counter() - t0

    kij_coeffs = result.x
    eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))

    # --- ARD -----------------------------------------------------------------
    abs_devs: list[float] = []
    kij_at_row = [_kij_at_T(kij_coeffs, float(T_arr[i]), kij_t_ref) for i in range(n_rows)]
    unique_kijs_ard = np.unique(kij_at_row)
    diag_ard: dict = {}
    for kij_val in unique_kijs_ard:
        try:
            eos = _build_binary_eos(record1, record2, float(kij_val))
            x_I_pred, x_II_pred = _lle_diagram(eos, T_arr, z1_global)
            diag_ard[float(kij_val)] = (x_I_pred, x_II_pred)
        except Exception:
            diag_ard[float(kij_val)] = (None, None)

    for i in range(n_rows):
        kij = _kij_at_T(kij_coeffs, float(T_arr[i]), kij_t_ref)
        x_I_pred_arr, x_II_pred_arr = diag_ard.get(float(kij), (None, None))
        if x_I_pred_arr is None:
            continue
        x_I_pred = float(x_I_pred_arr[i])
        x_II_pred = float(x_II_pred_arr[i])
        if np.isnan(x_I_pred) or np.isnan(x_II_pred):
            continue

        if has_phase_I:
            x_e = float(x1_I_arr[i])
            xp = x_I_pred
            abs_devs.append(abs(xp - x_e) / max(x_e, 1e-10))
            abs_devs.append(abs((1 - xp) - (1 - x_e)) / max(1 - x_e, 1e-10))
        if has_phase_II:
            x_e = float(x1_II_arr[i])
            xp = x_II_pred
            abs_devs.append(abs(xp - x_e) / max(x_e, 1e-10))
            abs_devs.append(abs((1 - xp) - (1 - x_e)) / max(1 - x_e, 1e-10))

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
