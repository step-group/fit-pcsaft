"""Binary phase diagram plotting for BinaryFitResult."""

import numpy as np
import si_units as si

_LINE_COLOR = "#000000"
_EXP_COLOR_1 = "#E32F2F"  # liquid / phase I
_EXP_COLOR_2 = "#1F77B4"  # vapor / phase II
_GRAY = "#AAAAAA"  # filtered-out (unused) data points
_PRED_COLOR = "#888888"  # unfitted / predictive (k_ij = 0) curves

_R = si.RGAS / (si.JOULE / (si.MOL * si.KELVIN))


def _pressure_label(pu) -> str:
    scale = pu / si.PASCAL
    if abs(scale - 1e3) < 1:
        return "kPa"
    if abs(scale - 1e6) < 1:
        return "MPa"
    if abs(scale - 1e5) < 1:
        return "bar"
    return "Pa"


def _scatter_kw(color: str, marker: str = "o") -> dict:
    return dict(
        s=40,
        marker=marker,
        facecolors="white",
        edgecolors=color,
        linewidths=1.2,
        zorder=5,
    )


def _curve_plot(ax, x_arr, y_arr, fit_min_K, fit_max_K, y_is_T: bool, **line_kw):
    """Plot a curve solid within the fitting range and dashed outside it.

    Parameters
    ----------
    x_arr, y_arr : array-like
        Curve coordinates.
    fit_min_K, fit_max_K : float
        Fitting temperature bounds in K (NaN = no bound).
    y_is_T : bool
        True when the temperature axis is Y (LLE/SLE); False when it is X (unused here).
    """
    x_arr = np.asarray(x_arr)
    y_arr = np.asarray(y_arr)
    T_arr = y_arr if y_is_T else x_arr

    has_min = not np.isnan(fit_min_K)
    has_max = not np.isnan(fit_max_K)
    if not has_min and not has_max:
        ax.plot(x_arr, y_arr, **line_kw)
        return

    lo = fit_min_K if has_min else -np.inf
    hi = fit_max_K if has_max else np.inf
    in_range = (T_arr >= lo) & (T_arr <= hi)

    # Draw dashed line for the full curve, then solid on top for the in-range part
    out_kw = {**line_kw, "linestyle": "--", "alpha": 0.45, "label": None}
    ax.plot(x_arr, y_arr, **out_kw)
    if in_range.any():
        ax.plot(x_arr[in_range], y_arr[in_range], **line_kw)


def _plot_binary(
    result,
    path=None,
    temperature_unit=si.KELVIN,
    pressure_unit=si.KILO * si.PASCAL,
    henry_unit=si.MEGA * si.PASCAL,
    plot_unfitted: bool = False,
):
    eq = result.equilibrium_type
    if eq == "vle":
        return _plot_vle(
            result, path, temperature_unit, pressure_unit, plot_unfitted=plot_unfitted
        )
    elif eq == "lle":
        return _plot_lle(result, path, temperature_unit, plot_unfitted=plot_unfitted)
    elif eq == "sle":
        return _plot_sle(result, path, temperature_unit)
    elif eq == "henry":
        return _plot_henry(result, path, temperature_unit, henry_unit)
    else:
        raise ValueError(f"Unknown equilibrium_type: {eq!r}")


# ---------------------------------------------------------------------------
# VLE
# ---------------------------------------------------------------------------


def _lle_feed_z1(result) -> float:
    """Estimate a representative feed composition z1 from experimental LLE data.

    If both tieline compositions are available, z1 is the mean midpoint.
    If only one phase is known, z1 is the mean of that phase.
    Falls back to 0.5 if neither is present.
    """
    df = result.data_full
    x_I = df["x1_I"].astype(float) if "x1_I" in df else None
    x_II = df["x1_II"].astype(float) if "x1_II" in df else None
    # Use per-column nanmean so that non-overlapping rows (different T per phase)
    # don't produce an all-NaN element-wise sum.
    mean_I = float(np.nanmean(x_I)) if x_I is not None else float("nan")
    mean_II = float(np.nanmean(x_II)) if x_II is not None else float("nan")
    if not np.isnan(mean_I) and not np.isnan(mean_II):
        z1 = 0.5 * (mean_I + mean_II)
    elif not np.isnan(mean_I):
        z1 = mean_I
    elif not np.isnan(mean_II):
        z1 = mean_II
    else:
        z1 = 0.5
    return float(np.clip(z1, 0.05, 0.95))


def _build_eos_kij0(result):
    """Return a new EOS with k_ij=0.0, or None if the parameters are unavailable."""
    import feos

    try:
        pure_records = result.eos.parameters.pure_records
        params = feos.Parameters.new_binary(pure_records, k_ij=0.0)
        return feos.EquationOfState.pcsaft(params)
    except Exception:
        return None


def _plot_vle(
    result, path, temperature_unit, pressure_unit, plot_unfitted: bool = False
):
    import feos
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    data = result.data
    t_scale = float(temperature_unit / si.KELVIN)
    T_data = data["T"].astype(float) * t_scale  # → K
    P_data = data["P"].astype(float)
    x1_data = data["x1"].astype(float)
    has_y1 = "y1" in data

    # Is the data approximately isobaric?
    p_cv = np.std(P_data) / np.mean(P_data)  # coefficient of variation
    is_isobaric = p_cv < 0.05

    fig, ax = plt.subplots(figsize=(8, 6))
    p_lbl = _pressure_label(pressure_unit)

    fit_min_K = result.t_filter_min_K
    fit_max_K = result.t_filter_max_K

    T_data_full = result.data_full["T"].astype(float) * t_scale
    lo = fit_min_K if not np.isnan(fit_min_K) else -np.inf
    hi = fit_max_K if not np.isnan(fit_max_K) else np.inf
    unused_mask = (T_data_full < lo) | (T_data_full > hi)

    if is_isobaric:
        # T-x-y diagram at mean pressure
        P_mean = float(np.mean(P_data))
        try:
            vle_pd = feos.PhaseDiagram.binary_vle(
                result.eos, P_mean * pressure_unit, npoints=200
            )
            T_curve = vle_pd.liquid.temperature / si.KELVIN
            _curve_plot(
                ax,
                vle_pd.liquid.molefracs[:, 0],
                T_curve,
                fit_min_K,
                fit_max_K,
                y_is_T=True,
                color=_EXP_COLOR_1,
                linestyle="-",
                label="PC-SAFT (bubble)",
            )
            _curve_plot(
                ax,
                vle_pd.vapor.molefracs[:, 0],
                T_curve,
                fit_min_K,
                fit_max_K,
                y_is_T=True,
                color=_EXP_COLOR_2,
                linestyle="-",
                label="PC-SAFT (dew)",
            )
        except Exception:
            pass

        if plot_unfitted:
            eos_u = _build_eos_kij0(result)
            if eos_u is not None:
                try:
                    vle_u = feos.PhaseDiagram.binary_vle(
                        eos_u, P_mean * pressure_unit, npoints=200
                    )
                    T_u = vle_u.liquid.temperature / si.KELVIN
                    ax.plot(
                        vle_u.liquid.molefracs[:, 0],
                        T_u,
                        color=_PRED_COLOR,
                        linestyle="--",
                        label="Predictive (k_ij = 0)",
                    )
                    ax.plot(
                        vle_u.vapor.molefracs[:, 0],
                        T_u,
                        color=_PRED_COLOR,
                        linestyle="--",
                    )
                except Exception:
                    pass

        if unused_mask.any():
            x1_full = result.data_full["x1"].astype(float)
            ax.scatter(
                x1_full[unused_mask], T_data_full[unused_mask], **_scatter_kw(_GRAY)
            )
            if "y1" in result.data_full:
                ax.scatter(
                    result.data_full["y1"].astype(float)[unused_mask],
                    T_data_full[unused_mask],
                    **_scatter_kw(_GRAY, "^"),
                )

        ax.scatter(x1_data, T_data, label=r"Exp. $x_1$", **_scatter_kw(_EXP_COLOR_1))
        if has_y1:
            ax.scatter(
                data["y1"].astype(float),
                T_data,
                label=r"Exp. $y_1$",
                **_scatter_kw(_EXP_COLOR_2, "^"),
            )
        ax.set_xlabel(rf"$x_1,\,y_1$ ({result.id1})")
        ax.set_ylabel("$T$ / K")
        ax.set_xlim(0, 1)
        ax.set_title(f"VLE: {result.id1} + {result.id2}  ($p$ = {P_mean:.0f} {p_lbl})")

    else:
        # Multi-isothermal P-x-y diagram: one curve per unique temperature (full data)
        T_data_full = result.data_full["T"].astype(float) * t_scale
        unique_Ts = sorted(np.unique(np.round(T_data_full, 0)))
        cmap = plt.cm.plasma
        colors = [cmap(i / max(1, len(unique_Ts) - 1)) for i in range(len(unique_Ts))]

        x1_full = result.data_full["x1"].astype(float)
        P_full = result.data_full["P"].astype(float)
        y1_full = (
            result.data_full["y1"].astype(float) if "y1" in result.data_full else None
        )

        for k, T_iso in enumerate(unique_Ts):
            color = colors[k]
            iso_in_range = lo <= T_iso <= hi
            alpha = 1.0 if iso_in_range else 0.45
            ls = "-" if iso_in_range else "--"
            scatter_color = color if iso_in_range else _GRAY
            try:
                vle_iso = feos.PhaseDiagram.binary_vle(
                    result.eos, T_iso * si.KELVIN, npoints=200
                )
                P_curve = vle_iso.liquid.pressure / pressure_unit
                ax.plot(
                    vle_iso.liquid.molefracs[:, 0],
                    P_curve,
                    color=color,
                    linestyle=ls,
                    alpha=alpha,
                )
                ax.plot(
                    vle_iso.vapor.molefracs[:, 0],
                    P_curve,
                    color=color,
                    linestyle=ls,
                    alpha=alpha,
                )
            except Exception:
                pass

            iso_mask_full = np.abs(T_data_full - T_iso) < 0.6
            ax.scatter(
                x1_full[iso_mask_full],
                P_full[iso_mask_full],
                label=f"{T_iso:.0f} K" if iso_in_range else None,
                **_scatter_kw(scatter_color),
            )
            if y1_full is not None:
                ax.scatter(
                    y1_full[iso_mask_full],
                    P_full[iso_mask_full],
                    **_scatter_kw(scatter_color, "^"),
                )

        ax.set_xlabel(rf"$x_1,\,y_1$ ({result.id1})")
        ax.set_ylabel(f"$p$ / {p_lbl}")
        ax.set_xlim(0, 1)
        ax.set_title(f"VLE: {result.id1} + {result.id2}")
        ax.legend(fontsize="small", title="$T$")

    if is_isobaric:
        ax.legend(fontsize="small")
    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# LLE
# ---------------------------------------------------------------------------


def _lle_curve_kij_T(result, z1: float, T_min: float, T_max: float, npoints: int = 301):
    """Compute LLE phase-boundary curve with temperature-dependent k_ij(T).

    For each T, builds a fresh EOS from k_ij(T) and flashes with liquid
    initialization + warm-start chained from the previous step's PE.

    Returns (T_arr, x_I_list, x_II_list) — lists of successfully converged points.
    """
    import feos

    from fit_pcsaft._binary._utils import _build_binary_eos, _kij_at_T

    if result._record1 is None or result._record2 is None:
        return np.array([]), [], []

    from fit_pcsaft._binary.lle import _LLE_FEEDS

    dT = (T_max - T_min) / npoints  # step size from data range
    pressure = 1.0 * si.BAR
    # Primary feed + sigmoid grid as fallback (mirrors fitting routine)
    base_feeds = [z1] + _LLE_FEEDS

    T_out, x_I_out, x_II_out = [], [], []
    n_consec_fail = 0
    # Fixed anchor at T_min: always deep in the LLE region, so the warm-start
    # PE is reliable throughout the march (including near the UCST).
    T_anchor_K = T_min

    T_K = T_min
    while T_K <= T_max + 500.0:  # extend up to 500 K past data range
        kij_T = _kij_at_T(result.kij_coeffs, T_K, result.kij_t_ref)
        eos_T = _build_binary_eos(result._record1, result._record2, kij_T)

        # Prepend a targeted feed at the midpoint of the converging phases so
        # the flash stays on the LLE branch near the UCST.
        if len(x_I_out) >= 1:
            mid = 0.5 * (x_I_out[-1] + x_II_out[-1])
            feeds = [float(np.clip(mid, 0.01, 0.99))] + base_feeds
        else:
            feeds = base_feeds

        # Acceptance threshold: relax as the phases converge toward the UCST.
        if len(x_I_out) >= 2:
            last_split = x_II_out[-1] - x_I_out[-1]
            min_split = max(0.01, last_split * 0.05)
        else:
            min_split = 0.025

        # Warm-start anchor: same eos_T at T_min — EOS-compatible, and always
        # within the LLE region, giving a reliable initial guess for the flash.
        anchor_pe = None
        if T_K > T_anchor_K + 0.5:
            try:
                moles_a = np.array([feeds[0], 1.0 - feeds[0]]) * si.MOL
                s_a = feos.State(
                    eos_T,
                    T_anchor_K * si.KELVIN,
                    pressure=pressure,
                    moles=moles_a,
                    density_initialization="liquid",
                )
                anchor_pe = s_a.tp_flash(max_iter=500)
            except Exception:
                pass

        pe = None
        for z in feeds:
            try:
                moles = np.array([z, 1.0 - z]) * si.MOL
                s = feos.State(
                    eos_T,
                    T_K * si.KELVIN,
                    pressure=pressure,
                    moles=moles,
                    density_initialization="liquid",
                )
                candidate = s.tp_flash(initial_state=anchor_pe)
                x_a = float(candidate.liquid.molefracs[0])
                x_b = float(candidate.vapor.molefracs[0])
                if max(x_a, x_b) - min(x_a, x_b) > min_split:
                    pe = candidate
                    break
            except Exception:
                pass

        if pe is not None:
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            T_out.append(T_K)
            x_I_out.append(min(x_a, x_b))  # Phase I  = id1-lean
            x_II_out.append(max(x_a, x_b))  # Phase II = id1-rich
            n_consec_fail = 0
        else:
            n_consec_fail += 1
            if n_consec_fail >= 10:
                break  # envelope has closed

        T_K += dT

    # Remove outlier points where composition jumps discontinuously from the
    # last accepted point (VLE solutions sneaking through the gap filter).
    if len(T_out) > 2:
        xI_arr = np.array(x_I_out)
        xII_arr = np.array(x_II_out)
        good = [True]
        last = 0
        for i in range(1, len(xI_arr)):
            if (
                abs(xI_arr[i] - xI_arr[last]) < 0.15
                and abs(xII_arr[i] - xII_arr[last]) < 0.15
            ):
                good.append(True)
                last = i
            else:
                good.append(False)
        good = np.array(good)
        T_out = np.array(T_out)[good].tolist()
        x_I_out = xI_arr[good].tolist()
        x_II_out = xII_arr[good].tolist()

    return np.array(T_out), x_I_out, x_II_out


def _plot_lle(result, path, temperature_unit, plot_unfitted: bool = False):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    data = result.data
    t_scale = float(temperature_unit / si.KELVIN)
    T_data = data["T"].astype(float) * t_scale
    has_I = "x1_I" in data
    has_II = "x1_II" in data

    T_full = result.data_full["T"].astype(float) * t_scale
    T_pad = max((T_full.max() - T_full.min()) * 0.05, 1.0)

    fit_min_K = result.t_filter_min_K
    fit_max_K = result.t_filter_max_K
    curve_T_min = float(T_full.min()) - T_pad
    curve_T_max = float(T_full.max()) + T_pad

    fig, ax = plt.subplots(figsize=(8, 6))

    def _log_odds(x):
        x = np.asarray(x, dtype=float)
        return np.log10(np.clip(x, 1e-15, 1.0) / np.clip(1.0 - x, 1e-15, 1.0))

    # Unused (filtered-out) points mask
    lo = fit_min_K if not np.isnan(fit_min_K) else -np.inf
    hi = fit_max_K if not np.isnan(fit_max_K) else np.inf
    unused_mask = (T_full < lo) | (T_full > hi)

    z1 = _lle_feed_z1(result)

    # Compute LLE curve with temperature-dependent k_ij(T) + warm-start continuation.
    # Each T gets its own EOS built with kij(T); initial_state from the previous T
    # steers the flash toward LLE rather than VLE.
    T_curve_arr, x_I_curve_list, x_II_curve_list = _lle_curve_kij_T(
        result, z1, curve_T_min, curve_T_max, npoints=301
    )
    if (
        len(T_curve_arr) > 0
        and np.max(np.abs(np.array(x_I_curve_list) - np.array(x_II_curve_list))) > 1e-04
    ):
        _curve_plot(
            ax,
            _log_odds(x_I_curve_list),
            T_curve_arr,
            fit_min_K,
            fit_max_K,
            y_is_T=True,
            color=_EXP_COLOR_1,
            linestyle="-",
            label="PC-SAFT (phase I)",
        )
        _curve_plot(
            ax,
            _log_odds(x_II_curve_list),
            T_curve_arr,
            fit_min_K,
            fit_max_K,
            y_is_T=True,
            color=_EXP_COLOR_2,
            linestyle="-",
            label="PC-SAFT (phase II)",
        )

    if plot_unfitted and result._record1 is not None and result._record2 is not None:
        from types import SimpleNamespace

        mock = SimpleNamespace(
            _record1=result._record1,
            _record2=result._record2,
            kij_coeffs=np.array([0.0]),
            kij_t_ref=result.kij_t_ref,
        )
        T_u, x_I_u, x_II_u = _lle_curve_kij_T(
            mock, z1, curve_T_min, curve_T_max, npoints=301
        )
        if len(T_u) > 0 and np.max(np.abs(np.array(x_I_u) - np.array(x_II_u))) > 1e-04:
            ax.plot(
                _log_odds(x_I_u),
                T_u,
                color=_PRED_COLOR,
                linestyle="--",
                label="Predictive (k_ij = 0)",
            )
            ax.plot(_log_odds(x_II_u), T_u, color=_PRED_COLOR, linestyle="--")

    # Unused experimental points (gray, drawn first so used points sit on top)
    if unused_mask.any():
        T_unused = T_full[unused_mask]
        if "x1_I" in result.data_full:
            ax.scatter(
                _log_odds(result.data_full["x1_I"].astype(float)[unused_mask]),
                T_unused,
                **_scatter_kw(_GRAY),
            )
        if "x1_II" in result.data_full:
            ax.scatter(
                _log_odds(result.data_full["x1_II"].astype(float)[unused_mask]),
                T_unused,
                **_scatter_kw(_GRAY, "^"),
            )

    if has_I:
        ax.scatter(
            _log_odds(data["x1_I"].astype(float)),
            T_data,
            label="Exp. phase I",
            **_scatter_kw(_EXP_COLOR_1),
        )
    if has_II:
        ax.scatter(
            _log_odds(data["x1_II"].astype(float)),
            T_data,
            label="Exp. phase II",
            **_scatter_kw(_EXP_COLOR_2, "^"),
        )

    ax.set_xlabel(rf"$\log_{{10}}(x_1/x_2)$  ({result.id1} / {result.id2})")
    ax.set_ylabel("$T$ / K")
    ax.set_title(f"LLE: {result.id1} + {result.id2}")
    ax.legend(fontsize="small")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# k_ij vs T diagnostic (LLE point-wise)
# ---------------------------------------------------------------------------


def _plot_kij_vs_T(
    T_pw, kij_pw, kij_coeffs, kij_t_ref, id1, id2,
    equilibrium_type="lle", ard_pw=None, source=None, path=None,
):
    """Scatter pointwise k_ij values and the fitted polynomial k_ij(T).

    When *source* is provided (array of "vle"/"lle" strings), points are
    colored by type. Otherwise points are colored by per-point ARD% when
    ard_pw is provided, or in a single color.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(7, 5))

    if source is not None:
        # Color VLE and LLE points differently
        mask_vle = source == "vle"
        mask_lle = source == "lle"
        ax.scatter(
            T_pw[mask_vle], kij_pw[mask_vle],
            color=_EXP_COLOR_2, zorder=3, label="VLE $k_{ij}$", s=60, marker="o",
        )
        ax.scatter(
            T_pw[mask_lle], kij_pw[mask_lle],
            color=_EXP_COLOR_1, zorder=3, label="LLE $k_{ij}$", s=60, marker="s",
        )
    elif ard_pw is not None:
        sc = ax.scatter(
            T_pw,
            kij_pw,
            c=ard_pw,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=max(float(np.max(ard_pw)), 20.0),
            zorder=3,
            label="Point-wise $k_{ij}$",
            s=60,
        )
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("ARD %", fontsize="small")
    else:
        ax.scatter(
            T_pw, kij_pw, color=_EXP_COLOR_1, zorder=3, label="Point-wise $k_{ij}$"
        )

    T_lo = float(T_pw.min()) - 5.0
    T_hi = float(T_pw.max()) + 5.0
    T_curve = np.linspace(T_lo, T_hi, 300)
    kij_curve = sum(c * (T_curve - kij_t_ref) ** i for i, c in enumerate(kij_coeffs))
    order = len(kij_coeffs) - 1
    ax.plot(
        T_curve,
        kij_curve,
        color=_LINE_COLOR,
        linestyle="-",
        label=f"Poly fit (order {order})",
    )

    ax.axhline(0, color=_GRAY, linewidth=0.7, linestyle=":")
    ax.set_xlabel("$T$ / K")
    ax.set_ylabel("$k_{ij}$")
    ax.set_title(f"$k_{{ij}}(T)$: {id1} + {id2} ({equilibrium_type.upper()})")
    ax.legend(fontsize="small")
    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# SLE
# ---------------------------------------------------------------------------


def _find_eutectic(eos, Tm1, dHfus1, si_idx1, Tm2, dHfus2, si_idx2):
    """Find eutectic (T, x1) where both SvL branches cross.

    Searches from 80% of the lower melting point up to just below it.
    """
    from scipy.optimize import brentq

    T_hi = min(Tm1, Tm2) - 0.1
    T_lo = min(Tm1, Tm2) * 0.80

    def diff(T):
        x1a = _sle_fixed_point(eos, T, Tm1, dHfus1, si_idx1, 0.5)
        x1b = _sle_fixed_point(eos, T, Tm2, dHfus2, si_idx2, 0.5)
        return x1a - x1b

    try:
        T_test = np.linspace(T_lo, T_hi, 50)
        diffs = [diff(t) for t in T_test]
        bracket = None
        for i in range(len(diffs) - 1):
            if (
                np.isfinite(diffs[i])
                and np.isfinite(diffs[i + 1])
                and diffs[i] * diffs[i + 1] < 0
            ):
                bracket = (T_test[i], T_test[i + 1])
                break
        if bracket is None:
            return float("nan"), float("nan")
        T_eut = brentq(diff, *bracket, xtol=1e-4)
        x1_eut = _sle_fixed_point(eos, T_eut, Tm1, dHfus1, si_idx1, 0.5)
        return T_eut, x1_eut
    except Exception:
        return float("nan"), float("nan")


def _plot_sle(result, path, temperature_unit):
    import feos
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    data = result.data
    t_scale = float(temperature_unit / si.KELVIN)
    T_data = data["T"].astype(float) * t_scale
    x1_data = data["x1"].astype(float)

    Tm_K = result.tm_K
    dHfus_J = result.delta_hfus_J
    solid_index = result.solid_index
    eutectic = not np.isnan(result.tm2_K)
    Tm2_K = result.tm2_K
    dHfus2_J = result.delta_hfus2_J
    solid_index2 = 1 - solid_index

    T_min = float(T_data.min())
    T_full = result.data_full["T"].astype(float) * t_scale

    fit_min_K = result.t_filter_min_K
    fit_max_K = result.t_filter_max_K
    curve_T_min = float(T_full.min())

    fig, ax = plt.subplots(figsize=(8, 6))

    if eutectic:
        # Find eutectic point to properly clip each branch
        T_eut, x1_eut = _find_eutectic(
            result.eos,
            Tm_K,
            dHfus_J,
            solid_index,
            Tm2_K,
            dHfus2_J,
            solid_index2,
        )
        T_start = T_eut if not np.isnan(T_eut) else curve_T_min * 0.995

        def _plot_branch(Tm, dHfus, si_idx, x0_start, label, linestyle="-"):
            x1_curve, T_curve = [], []
            T_range = np.linspace(T_start, Tm, 120)
            x0 = x0_start
            for T_i in T_range:
                try:
                    x1 = _sle_fixed_point(result.eos, T_i, Tm, dHfus, si_idx, x0)
                    if not np.isnan(x1) and 0.0 <= x1 <= 1.0:
                        x1_curve.append(x1)
                        T_curve.append(T_i)
                        x0 = x1
                except Exception:
                    pass
            if x1_curve:
                _curve_plot(
                    ax,
                    np.array(x1_curve),
                    np.array(T_curve),
                    fit_min_K,
                    fit_max_K,
                    y_is_T=True,
                    color=_LINE_COLOR,
                    linestyle="-",
                    label=label,
                )

        solid_name = result.id2 if solid_index == 1 else result.id1
        solid_name2 = result.id1 if solid_index == 1 else result.id2
        x0_eut = x1_eut if not np.isnan(x1_eut) else float(x1_data[np.argmin(T_data)])
        _plot_branch(Tm_K, dHfus_J, solid_index, x0_eut, f"PC-SAFT ({solid_name})")
        _plot_branch(Tm2_K, dHfus2_J, solid_index2, x0_eut, f"PC-SAFT ({solid_name2})")

        if not np.isnan(T_eut):
            ax.scatter(
                [x1_eut],
                [T_eut],
                marker="D",
                s=60,
                color=_LINE_COLOR,
                zorder=6,
                label=f"Eutectic ({x1_eut:.3f}, {T_eut:.1f} K)",
            )

        title = f"SLE: {result.id1} + {result.id2}  (eutectic)"
    else:
        solid_name = result.id2 if solid_index == 1 else result.id1
        x1_curve, T_curve = [], []
        T_range = np.linspace(curve_T_min * 0.995, Tm_K, 120)
        x0 = float(x1_data[np.argmin(T_data)])
        for T_i in T_range:
            try:
                x1 = _sle_fixed_point(result.eos, T_i, Tm_K, dHfus_J, solid_index, x0)
                if not np.isnan(x1) and 0.0 <= x1 <= 1.0:
                    x1_curve.append(x1)
                    T_curve.append(T_i)
                    x0 = x1
            except Exception:
                pass
        if x1_curve:
            _curve_plot(
                ax,
                np.array(x1_curve),
                np.array(T_curve),
                fit_min_K,
                fit_max_K,
                y_is_T=True,
                color=_LINE_COLOR,
                linestyle="-",
                label="PC-SAFT",
            )
        title = f"SLE: {result.id1} + {result.id2}  (solid: {solid_name})"

    # Unused experimental points (gray, behind used ones)
    lo_f = fit_min_K if not np.isnan(fit_min_K) else -np.inf
    hi_f = fit_max_K if not np.isnan(fit_max_K) else np.inf
    unused_mask = (T_full < lo_f) | (T_full > hi_f)
    if unused_mask.any():
        x1_full = result.data_full["x1"].astype(float)
        ax.scatter(x1_full[unused_mask], T_full[unused_mask], **_scatter_kw(_GRAY))

    ax.scatter(x1_data, T_data, label="Exp.", **_scatter_kw(_EXP_COLOR_1))

    ax.set_xlabel(rf"$x_1$ ({result.id1})")
    ax.set_ylabel("$T$ / K")
    lo, hi = ax.get_xlim()
    ax.set_xlim(max(0.0, lo), min(1.0, hi))
    ax.set_title(title)
    ax.legend(fontsize="small")
    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Henry
# ---------------------------------------------------------------------------


def _henry_label(hu) -> str:
    if hu == "molfrac":
        return "mol/mol"
    scale = hu / si.PASCAL
    if abs(scale - 1e6) < 1e2:
        return "MPa"
    if abs(scale - 1e5) < 1e1:
        return "bar"
    if abs(scale - 101325) < 10:
        return "atm"
    if abs(scale - 1e3) < 1:
        return "kPa"
    return f"{scale:.4g} Pa"


def _plot_henry(result, path, temperature_unit, henry_unit):
    import feos
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    data = result.data
    t_scale = float(temperature_unit / si.KELVIN)
    T_data = data["T"].astype(float) * t_scale  # → K
    H_data = data["H"].astype(float)

    use_molfrac = henry_unit == "molfrac"

    # Build pure solvent EOS for molfrac conversion
    eos_solvent = None
    if use_molfrac and result._solvent_record is not None:
        eos_solvent = feos.EquationOfState.pcsaft(
            feos.Parameters.new_pure(result._solvent_record), max_iter_cross_assoc=100
        )

    def _h_pred(T_K: float) -> "float | None":
        try:
            H_pa = (
                feos.State.henrys_law_constant_binary(result.eos, T_K * si.KELVIN)
                / si.PASCAL
            )
            if use_molfrac:
                if eos_solvent is None:
                    return None
                pvap = (
                    feos.PhaseEquilibrium.vapor_pressure(eos_solvent, T_K * si.KELVIN)[
                        0
                    ]
                    / si.PASCAL
                )
                return H_pa / pvap
            else:
                return H_pa / (henry_unit / si.PASCAL)
        except Exception:
            return None

    T_min, T_max = float(T_data.min()), float(T_data.max())
    T_pad = max((T_max - T_min) * 0.05, 2.0)
    T_curve = np.linspace(T_min - T_pad, T_max + T_pad, 120)
    H_curve = [_h_pred(T) for T in T_curve]
    mask_curve = [h is not None for h in H_curve]
    T_curve_valid = T_curve[mask_curve]
    H_curve_valid = np.array([h for h in H_curve if h is not None])

    fig, ax = plt.subplots(figsize=(8, 6))
    h_lbl = _henry_label(henry_unit)

    if len(T_curve_valid) > 0:
        ax.plot(T_curve_valid, H_curve_valid, color=_LINE_COLOR, label="PC-SAFT")

    ax.scatter(T_data, H_data, label="Exp.", **_scatter_kw(_EXP_COLOR_1))

    ax.set_xlabel("$T$ / K")
    ax.set_ylabel(f"$H$ / {h_lbl}")
    ax.set_title(f"Henry: {result.id1} (solute) + {result.id2} (solvent)")
    ax.legend(fontsize="small")
    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def _sle_fixed_point(
    eos, T_K: float, Tm_K: float, dHfus_J: float, solid_index: int, x0: float
) -> float:
    """Solve Schröder-van Laar, return mole fraction of id1 in the liquid."""
    import feos

    rhs = -(dHfus_J / _R) * (1.0 / T_K - 1.0 / Tm_K)

    if solid_index == 0:
        x_iter = float(np.clip(x0, 1e-6, 1.0 - 1e-6))
        for _ in range(50):
            liq = feos.State(
                eos,
                temperature=T_K * si.KELVIN,
                pressure=1.0 * si.BAR,
                molefracs=np.array([x_iter, 1.0 - x_iter]),
                density_initialization="liquid",
            )
            ln_gamma = float(liq.ln_symmetric_activity_coefficient()[0])
            x_new = float(np.clip(np.exp(rhs - ln_gamma), 1e-9, 1.0 - 1e-9))
            if abs(x_new - x_iter) < 1e-9:
                return x_new
            x_iter = x_new
        return x_iter  # x_id1
    else:
        # id2 is solid: iterate on x_id2, return x_id1 = 1 - x_id2
        x_iter = float(np.clip(1.0 - x0, 1e-6, 1.0 - 1e-6))
        for _ in range(50):
            liq = feos.State(
                eos,
                temperature=T_K * si.KELVIN,
                pressure=1.0 * si.BAR,
                molefracs=np.array([1.0 - x_iter, x_iter]),
                density_initialization="liquid",
            )
            ln_gamma = float(liq.ln_symmetric_activity_coefficient()[1])
            x_new = float(np.clip(np.exp(rhs - ln_gamma), 1e-9, 1.0 - 1e-9))
            if abs(x_new - x_iter) < 1e-9:
                return 1.0 - x_new
            x_iter = x_new
        return 1.0 - x_iter  # return x_id1
