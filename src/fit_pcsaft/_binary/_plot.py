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


def _normalize_result_for_type(result, type_name: str):
    """Strip the ``type_name + "_"`` prefix from data keys for single-type plotters.

    ``BinaryKijFitter`` stores data under prefixed keys (``"vle_T"``, ``"lle_T"``,
    …) to avoid collisions when multiple sources are combined.  Single-type
    plotters (``_plot_vle``, ``_plot_lle``) expect the canonical unprefixed
    names.  This helper transparently remaps them and provides a fallback
    ``data_full`` when it is empty (as is always the case for
    ``BinaryKijFitter`` results).
    """
    import dataclasses

    prefix = type_name + "_"
    plen = len(prefix)
    data = result.data

    # If the prefixed key is absent the result came from a direct fit_kij_*
    # call — nothing to do.
    if f"{prefix}T" not in data:
        return result

    # Re-key: strip our prefix, drop keys that belong to *other* type prefixes.
    _all_prefixes = {"vle_", "lle_", "vlle_", "sle_"}
    _meta = {"T_kij", "kij_pointwise", "ard_pointwise", "source"}
    new_data: dict = {}
    for k, v in data.items():
        if any(k.startswith(p) for p in _all_prefixes if p != prefix):
            continue  # belongs to another source type — drop it
        new_data[k[plen:] if k.startswith(prefix) else k] = v

    # data_full is empty for BinaryKijFitter; fall back to the experimental
    # columns (exclude fitting metadata so the unused-mask logic sees no gaps).
    data_full = result.data_full
    if not data_full:
        data_full = {k: v for k, v in new_data.items()
                     if k not in _meta and not k.startswith("ard_")}

    return dataclasses.replace(result, data=new_data, data_full=data_full)


def _plot_binary(
    result,
    path=None,
    temperature_unit=si.KELVIN,
    pressure_unit=si.KILO * si.PASCAL,
    henry_unit=si.MEGA * si.PASCAL,
    plot_unfitted: bool = False,
):
    eq = result.equilibrium_type
    # Normalise to a frozenset of type tokens.
    # Supports both legacy "vle_lle" (from fit_kij_vle_lle) and
    # new "vle+lle", "vle+lle+vlle", etc. (from BinaryKijFitter).
    _tokens = frozenset(eq.replace("_", "+").split("+"))

    if _tokens == {"vle"}:
        return _plot_vle(_normalize_result_for_type(result, "vle"),
                         path, temperature_unit, pressure_unit,
                         plot_unfitted=plot_unfitted)
    elif _tokens == {"lle"}:
        return _plot_lle(_normalize_result_for_type(result, "lle"),
                         path, temperature_unit, plot_unfitted=plot_unfitted)
    elif _tokens == {"sle"}:
        return _plot_sle(_normalize_result_for_type(result, "sle"),
                         path, temperature_unit)
    elif _tokens == {"vlle"}:
        return _plot_vlle(_normalize_result_for_type(result, "vlle"),
                          path, temperature_unit, pressure_unit)
    elif _tokens == {"henry"}:
        return _plot_henry(result, path, temperature_unit, henry_unit)
    elif "vle" in _tokens:
        # vle+lle, vle+vlle, vle+lle+vlle, vle_lle (legacy) — all go to _plot_vle_lle
        return _plot_vle_lle(result, path, temperature_unit, pressure_unit,
                             plot_unfitted=plot_unfitted)
    elif "vlle" in _tokens:
        # vlle-only or vlle+lle (no VLE): conjoined T-x / P-x VLLE plot
        return _plot_vlle(result, path, temperature_unit, pressure_unit)
    elif "lle" in _tokens:
        return _plot_lle(_normalize_result_for_type(result, "lle"),
                         path, temperature_unit, plot_unfitted=plot_unfitted)
    else:
        raise ValueError(
            f"No plot implemented for equilibrium_type {eq!r}.\n"
            f"Supported: vle, lle, sle, henry, vle+lle, vle+vlle, vle+lle+vlle "
            f"(and legacy vle_lle)."
        )


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

    if result._record1 is None or result._record2 is None:
        return None
    try:
        params = feos.Parameters.new_binary(
            [result._record1, result._record2], k_ij=0.0
        )
        return feos.EquationOfState.pcsaft(params, max_iter_cross_assoc=100)
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

    fig, ax = plt.subplots(figsize=(9, 6))
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
        ax.legend(fontsize="small", title="$T$", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    if is_isobaric:
        ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

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

    fig, ax = plt.subplots(figsize=(9, 6))

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
    ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

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

    fig, ax = plt.subplots(figsize=(9, 6))

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
    ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

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

    fig, ax = plt.subplots(figsize=(9, 6))

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
    ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

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

    fig, ax = plt.subplots(figsize=(9, 6))
    h_lbl = _henry_label(henry_unit)

    if len(T_curve_valid) > 0:
        ax.plot(T_curve_valid, H_curve_valid, color=_LINE_COLOR, label="PC-SAFT")

    ax.scatter(T_data, H_data, label="Exp.", **_scatter_kw(_EXP_COLOR_1))

    ax.set_xlabel("$T$ / K")
    ax.set_ylabel(f"$H$ / {h_lbl}")
    ax.set_title(f"Henry: {result.id1} (solute) + {result.id2} (solvent)")
    ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

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


# ---------------------------------------------------------------------------
# VLE + LLE combined
# ---------------------------------------------------------------------------


def _find_heteroazeotrope(result, pressure_si, x_I_init: float, x_II_init: float,
                          T_init_K: float):
    """Find the heteroazeotrope with PhaseEquilibrium.heteroazeotrope.

    Builds the EOS at k_ij(T_init_K) and calls feos' heteroazeotrope solver.

    Returns (T_het_K, x1_liq_I, x1_liq_II, y1_vap) or None on failure.
    The two liquid compositions are the water-rich (x1_liq_I, small x1) and
    toluene-rich (x1_liq_II, large x1) phases at the three-phase point.
    """
    import feos

    from fit_pcsaft._binary._utils import _build_binary_eos, _kij_at_T

    if result._record1 is None or result._record2 is None:
        return None

    kij = _kij_at_T(result.kij_coeffs, T_init_K, result.kij_t_ref)
    eos = _build_binary_eos(result._record1, result._record2, kij)
    try:
        ha = feos.PhaseEquilibrium.heteroazeotrope(
            eos, pressure_si,
            x_init=(float(x_I_init), float(x_II_init)),
            tp_init=T_init_K * si.KELVIN,
        )
        # ThreePhaseEquilibrium has liquid1, liquid2, vapor attributes
        T_het = float(ha.vapor.temperature / si.KELVIN)
        x1_liq_I = float(ha.liquid1.molefracs[0])   # one liquid phase (low x1)
        x1_liq_II = float(ha.liquid2.molefracs[0])  # other liquid phase (high x1)
        y1_vap = float(ha.vapor.molefracs[0])        # vapor composition
        # Ensure x1_liq_I < x1_liq_II (phase I = water-rich, phase II = MIBK-rich)
        if x1_liq_I > x1_liq_II:
            x1_liq_I, x1_liq_II = x1_liq_II, x1_liq_I
        return T_het, x1_liq_I, x1_liq_II, y1_vap
    except Exception:
        return None


def _vle_branch_isobaric(
    result, pressure_si, x1_start: float, x1_end: float,
    T_start_K: float, npoints: int = 80,
) -> "tuple[list, list, list]":
    """Trace one arm of the isobaric VLE bubble curve via PhaseEquilibrium.bubble_point.

    Passes *pressure_si* as the first argument to ``bubble_point`` so that feos
    solves for the **bubble temperature** at fixed pressure (isobaric mode).
    Steps the liquid composition x1 from *x1_start* → *x1_end* (toward a pure
    component) and uses the previous T as the temperature initial guess.

    Returns (x1_bubble, T_K_bubble, y1_dew) — the dew-curve y1 values come
    naturally as output of each bubble_point call.
    """
    import feos

    from fit_pcsaft._binary._utils import _build_binary_eos, _kij_at_T

    x1_arr = np.linspace(x1_start, x1_end, npoints + 1)[1:]  # skip start point
    x1_out, T_out, y1_out = [], [], []
    T_K = T_start_K

    for x1 in x1_arr:
        x1 = float(np.clip(x1, 1e-6, 1.0 - 1e-6))
        kij = _kij_at_T(result.kij_coeffs, T_K, result.kij_t_ref)
        eos = _build_binary_eos(result._record1, result._record2, kij)
        try:
            bp = feos.PhaseEquilibrium.bubble_point(
                eos, pressure_si,
                np.array([x1, 1.0 - x1]),
                tp_init=T_K * si.KELVIN,
            )
            T_K = float(bp.liquid.temperature / si.KELVIN)
            y1 = float(bp.vapor.molefracs[0])
            x1_out.append(x1)
            T_out.append(T_K)
            y1_out.append(y1)
        except Exception:
            break

    return x1_out, T_out, y1_out


def _plot_vle_lle(
    result, path, temperature_unit, pressure_unit, plot_unfitted: bool = False
):
    """T-x diagram overlaying VLE bubble/dew curves and LLE binodal.

    VLE data (liquid = circles, vapour = triangles) and LLE data (phase I =
    squares, phase II = diamonds) are shown on a linear composition axis.

    The heteroazeotrope is located with ``PhaseEquilibrium.heteroazeotrope``.
    From it:
      - Bubble curves go outward (toward pure components) using
        ``feos.State`` with ``density_initialization="liquid"``.
      - Dew curves go outward using ``density_initialization="vapor"``.
      - The LLE binodal goes downward in T using ``_lle_curve_kij_T``
        (also liquid-initialized), honouring the T-dependent k_ij polynomial.
    The three-phase horizontal line is drawn at T_heteroazeotrope.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(9, 6))

    t_scale = float(temperature_unit / si.KELVIN)
    data = result.data

    # ---- VLE experimental data -------------------------------------------------
    vle_T = data["vle_T"].astype(float) * t_scale  # in K
    vle_P = data["vle_P"].astype(float)
    vle_x1 = data["vle_x1"].astype(float)
    has_y1 = "vle_y1" in data

    P_mean = float(np.mean(vle_P))
    is_isobaric = np.std(vle_P) / P_mean < 0.05
    p_lbl = _pressure_label(pressure_unit)
    pressure_si = P_mean * pressure_unit

    # ---- LLE experimental data (optional) -------------------------------------
    has_lle = "lle_T" in data
    lle_T = data["lle_T"].astype(float) * t_scale if has_lle else np.array([])
    has_I  = "lle_x1_I"  in data
    has_II = "lle_x1_II" in data

    vals_I  = data["lle_x1_I"].astype(float)  if has_I  else np.array([])
    vals_II = data["lle_x1_II"].astype(float) if has_II else np.array([])

    # ---- VLLE experimental data (optional) ------------------------------------
    has_vlle = "vlle_T" in data
    if has_vlle:
        vlle_T_arr  = data["vlle_T"].astype(float) * t_scale
        vlle_x1_I   = data["vlle_x1_I"].astype(float)  if "vlle_x1_I"  in data else None
        vlle_x1_II  = data["vlle_x1_II"].astype(float) if "vlle_x1_II" in data else None
        vlle_y1     = data["vlle_y1"].astype(float)     if "vlle_y1"    in data else None

    # ---- Initial guess for heteroazeotrope compositions -----------------------
    # Prefer LLE data; fall back to VLLE phase compositions if LLE is absent.
    mean_I = float(np.nanmean(vals_I)) if len(vals_I) else float("nan")
    mean_II = float(np.nanmean(vals_II)) if len(vals_II) else float("nan")
    if np.isnan(mean_I) and has_vlle and vlle_x1_I is not None:
        mean_I = float(np.nanmean(vlle_x1_I))
    if np.isnan(mean_II) and has_vlle and vlle_x1_II is not None:
        mean_II = float(np.nanmean(vlle_x1_II))
    x_I_init  = mean_I  if not np.isnan(mean_I)  else 0.01
    x_II_init = mean_II if not np.isnan(mean_II) else 0.90

    # ---- Find heteroazeotrope --------------------------------------------------
    T_vle_mean = float(np.mean(vle_T))
    ha = _find_heteroazeotrope(result, pressure_si, x_I_init, x_II_init, T_vle_mean)

    # ---- VLE computed curves (only for isobaric data) -------------------------
    if is_isobaric and ha is not None:
        T_het, x1_I_het, x1_II_het, y1_het = ha

        # Bubble + dew curves: each arm returns (x1_bubble, T, y1_dew) via
        # isobaric bubble_point (pressure first → finds bubble temperature).
        # Water-rich arm: x1_I_het → 0 (pure water)
        xb1, Tb1, yd1 = _vle_branch_isobaric(
            result, pressure_si, x1_I_het, 1e-5, T_het, npoints=80,
        )
        # MIBK-rich arm: x1_II_het → 1 (pure MIBK)
        xb2, Tb2, yd2 = _vle_branch_isobaric(
            result, pressure_si, x1_II_het, 1.0 - 1e-5, T_het, npoints=80,
        )

        def _above_het(xs, Ts, ys=None):
            """Drop any points where T < T_het (numerical noise near three-phase point)."""
            mask = [t >= T_het for t in Ts]
            xs_f = [x for x, m in zip(xs, mask) if m]
            Ts_f = [t for t, m in zip(Ts, mask) if m]
            if ys is not None:
                ys_f = [y for y, m in zip(ys, mask) if m]
                return xs_f, Ts_f, ys_f
            return xs_f, Ts_f

        xb1, Tb1, yd1 = _above_het(xb1, Tb1, yd1)
        xb2, Tb2, yd2 = _above_het(xb2, Tb2, yd2)

        # Bubble curves (liquid compositions on x-axis)
        if xb1:
            ax.plot(xb1, Tb1, color=_EXP_COLOR_1, linestyle="-", label="PC-SAFT (bubble)")
        if xb2:
            ax.plot(xb2, Tb2, color=_EXP_COLOR_1, linestyle="-")

        # Dew curves (vapor compositions on x-axis, same T as bubble tie-line)
        if yd1:
            ax.plot(yd1, Tb1[:len(yd1)], color=_EXP_COLOR_2, linestyle="-", label="PC-SAFT (dew)")
        if yd2:
            ax.plot(yd2, Tb2[:len(yd2)], color=_EXP_COLOR_2, linestyle="-")

        # Three-phase horizontal line at T_het
        ax.hlines(T_het, x1_I_het, x1_II_het,
                  colors=_GRAY, linewidth=1.2, linestyle="-", label=f"VLLE  ({T_het:.1f} K)")

    # ---- LLE computed binodal (only when LLE or VLLE data is present) ---------
    if has_lle or has_vlle:
        z1 = float(np.clip(0.5 * (x_I_init + x_II_init), 0.05, 0.95))
        _T_ref_lo = lle_T.min() if has_lle else (vlle_T_arr.min() if has_vlle else vle_T.min())
        _T_ref_hi = lle_T.max() if has_lle else (vlle_T_arr.max() if has_vlle else vle_T.mean())
        T_pad = max((_T_ref_hi - _T_ref_lo) * 0.05, 1.0)
        curve_T_min = float(_T_ref_lo) - T_pad
        curve_T_max = (float(ha[0]) if ha is not None else float(_T_ref_hi)) + T_pad

        T_c, x_I_c, x_II_c = _lle_curve_kij_T(result, z1, curve_T_min, curve_T_max, npoints=301)
        if len(T_c) > 0:
            T_c = np.array(T_c)
            x_I_c = np.array(x_I_c)
            x_II_c = np.array(x_II_c)
            if ha is not None:
                mask = T_c <= ha[0]
                T_c, x_I_c, x_II_c = T_c[mask], x_I_c[mask], x_II_c[mask]
            if len(T_c) > 0 and np.max(np.abs(x_I_c - x_II_c)) > 1e-4:
                ax.plot(x_I_c, T_c, color=_EXP_COLOR_1, linestyle="--", label="PC-SAFT (LLE phase I)")
                ax.plot(x_II_c, T_c, color=_EXP_COLOR_2, linestyle="--", label="PC-SAFT (LLE phase II)")

        if plot_unfitted and result._record1 is not None and result._record2 is not None:
            from types import SimpleNamespace as _NS
            mock = _NS(
                _record1=result._record1, _record2=result._record2,
                kij_coeffs=np.array([0.0]), kij_t_ref=result.kij_t_ref,
            )
            T_u, x_I_u, x_II_u = _lle_curve_kij_T(mock, z1, curve_T_min, curve_T_max, npoints=301)
            if len(T_u) > 0 and np.max(np.abs(np.array(x_I_u) - np.array(x_II_u))) > 1e-4:
                ax.plot(x_I_u, T_u, color=_PRED_COLOR, linestyle=":", label="Predictive (k_ij = 0)")
                ax.plot(x_II_u, T_u, color=_PRED_COLOR, linestyle=":")

    # ---- Scatter: VLE ----------------------------------------------------------
    ax.scatter(vle_x1, vle_T, label=r"Exp. VLE $x_1$", **_scatter_kw(_EXP_COLOR_1))
    if has_y1:
        ax.scatter(data["vle_y1"].astype(float), vle_T,
                   label=r"Exp. VLE $y_1$", **_scatter_kw(_EXP_COLOR_2, "^"))

    # ---- Scatter: LLE ----------------------------------------------------------
    _lle_kw_I  = dict(s=40, marker="s", facecolors="white", edgecolors="#8B4513", linewidths=1.2, zorder=5)
    _lle_kw_II = dict(s=40, marker="D", facecolors="white", edgecolors="#2E8B57", linewidths=1.2, zorder=5)
    if has_I:
        x_I_arr = data["lle_x1_I"].astype(float)
        valid = ~np.isnan(x_I_arr)
        ax.scatter(x_I_arr[valid], lle_T[valid], label="Exp. LLE phase I", **_lle_kw_I)
    if has_II:
        x_II_arr = data["lle_x1_II"].astype(float)
        valid = ~np.isnan(x_II_arr)
        ax.scatter(x_II_arr[valid], lle_T[valid], label="Exp. LLE phase II", **_lle_kw_II)

    # ---- Scatter: VLLE (three-phase experimental points) ----------------------
    if has_vlle:
        _vlle_kw = dict(s=60, marker="*", color="#9400D3", zorder=6)
        if vlle_x1_I is not None:
            ax.scatter(vlle_x1_I, vlle_T_arr, label="Exp. VLLE phase I", **_vlle_kw)
        if vlle_x1_II is not None:
            ax.scatter(vlle_x1_II, vlle_T_arr, **_vlle_kw)
        if vlle_y1 is not None:
            ax.scatter(vlle_y1, vlle_T_arr, label="Exp. VLLE vapor", **{**_vlle_kw, "marker": "p"})

    ax.set_xlabel(rf"$x_1$ ({result.id1})")
    ax.set_ylabel("$T$ / K")
    ax.set_xlim(-0.02, 1.02)
    title = f"VLE + LLE: {result.id1} + {result.id2}"
    if is_isobaric:
        title += rf"  ($p$ = {P_mean:.0f} {p_lbl})"
    ax.set_title(title)
    ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# VLLE  (multiple-pressure heteroazeotrope data)
# ---------------------------------------------------------------------------


def _vlle_locus(
    result, P_arr_si,
    x_I_init: float = 0.05, x_II_init: float = 0.90,
    T_init: float = 350.0,
):
    """Compute the VLLE heteroazeotrope locus over a range of pressures.

    For each pressure in *P_arr_si* (Pa, plain floats), calls
    ``feos.PhaseEquilibrium.heteroazeotrope`` with k_ij evaluated at the
    predicted T from the previous step as the warm-start.  Returns four
    arrays (T_K, x1_I, x1_II, y1) with ``nan`` for any pressure that failed.

    Parameters
    ----------
    x_I_init, x_II_init : float
        Initial mole-fraction guesses for the two liquid phases, used only
        for the very first pressure step (subsequent steps warm-start from
        the previous solution).  Passing values close to the actual phase
        compositions greatly improves convergence.
    T_init : float
        Initial temperature guess (K) for the very first pressure step.
        Should be close to the heteroazeotrope temperature at P_arr_si[0].

    Notes
    -----
    Sweep *P_arr_si* from the well-known end (high P, near data range) toward
    the extrapolation end (low P) so the warm-start propagates naturally into
    the unconstrained region.  Passing a descending pressure array is the
    recommended usage when low-P extrapolation is desired.
    """
    import feos

    from fit_pcsaft._binary._utils import _build_binary_eos, _kij_at_T

    if result._record1 is None or result._record2 is None:
        nan = np.full(len(P_arr_si), float("nan"))
        return nan, nan.copy(), nan.copy(), nan.copy()

    T_K = float("nan")
    x_I = float("nan")
    x_II = float("nan")

    T_out, xI_out, xII_out, y1_out = [], [], [], []
    # Last two successful (T, ln P) pairs for extrapolation when warm-start fails.
    _prev_ok: "list[tuple[float, float]]" = []

    for P_si in P_arr_si:
        T_guess = T_K if not np.isnan(T_K) else T_init
        x_I_g = x_I if not np.isnan(x_I) else x_I_init
        x_II_g = x_II if not np.isnan(x_II) else x_II_init

        # Build candidate T guesses: warm-start first, then extrapolation + T_init.
        T_candidates: "list[float]" = [T_guess]
        if len(_prev_ok) >= 2:
            T1, lnP1 = _prev_ok[-1]
            T2, lnP2 = _prev_ok[-2]
            dlnP = lnP1 - lnP2
            if abs(dlnP) > 1e-8:
                T_extrap = T1 + (T1 - T2) / dlnP * (np.log(float(P_si)) - lnP1)
                T_extrap = float(np.clip(T_extrap, 200.0, 700.0))
                if abs(T_extrap - T_guess) > 3.0:
                    T_candidates.append(T_extrap)
        if abs(T_init - T_guess) > 3.0:
            T_candidates.append(T_init)

        success = False
        for T_try in T_candidates:
            try:
                kij = _kij_at_T(result.kij_coeffs, T_try, result.kij_t_ref)
                eos = _build_binary_eos(result._record1, result._record2, kij)
                ha = feos.PhaseEquilibrium.heteroazeotrope(
                    eos, float(P_si) * si.PASCAL,
                    x_init=(float(np.clip(x_I_g, 1e-4, 1.0 - 1e-4)),
                            float(np.clip(x_II_g, 1e-4, 1.0 - 1e-4))),
                    tp_init=T_try * si.KELVIN,
                )
                T_K = float(ha.vapor.temperature / si.KELVIN)
                x_I = float(ha.liquid1.molefracs[0])
                x_II = float(ha.liquid2.molefracs[0])
                y1 = float(ha.vapor.molefracs[0])
                _prev_ok.append((T_K, np.log(float(P_si))))
                if len(_prev_ok) > 2:
                    _prev_ok.pop(0)
                success = True
                break
            except Exception:
                pass

        if success:
            # No swap: keep feos' liquid1/liquid2 order; _plot_vlle will align
            # the two branches to the experimental phase convention by proximity.
            T_out.append(T_K); xI_out.append(x_I)
            xII_out.append(x_II); y1_out.append(y1)
        else:
            T_K = float("nan")
            T_out.append(float("nan")); xI_out.append(float("nan"))
            xII_out.append(float("nan")); y1_out.append(float("nan"))

    return (np.array(T_out), np.array(xI_out),
            np.array(xII_out), np.array(y1_out))


def _plot_vlle(result, path, temperature_unit, pressure_unit):
    """Conjoined T-x / P-x diagram for multi-pressure VLLE data.

    Two side-by-side panels sharing the composition (x1) axis:

    * **Left** — T vs x1: how the three-phase compositions change with T.
      If LLE data is present in the result, the LLE binodal is also drawn.
    * **Right** — P vs x1: same three-phase compositions plotted against P.

    The model locus is obtained by sweeping pressure over the experimental
    range (±10 % padding) and calling
    ``feos.PhaseEquilibrium.heteroazeotrope`` at each step.  The two liquid
    branches from feos are matched to the experimental phase convention
    (I vs II) by proximity of their mean compositions.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    data = result.data
    t_scale = float(temperature_unit / si.KELVIN)
    p_scale = float(pressure_unit / si.PASCAL)   # data units → Pa
    p_lbl = _pressure_label(pressure_unit)

    # --- Experimental VLLE data ----------------------------------------------
    vlle_key = "vlle_T" if "vlle_T" in data else "T"
    vlle_prefix = "vlle_" if "vlle_T" in data else ""
    T_exp = data[vlle_prefix + "T"].astype(float) * t_scale
    P_exp = data[vlle_prefix + "P"].astype(float)
    has_xI  = (vlle_prefix + "x1_I")  in data
    has_xII = (vlle_prefix + "x1_II") in data
    has_y1  = (vlle_prefix + "y1")    in data
    xI_exp  = data[vlle_prefix + "x1_I"].astype(float)  if has_xI  else None
    xII_exp = data[vlle_prefix + "x1_II"].astype(float) if has_xII else None
    y1_exp  = data[vlle_prefix + "y1"].astype(float)    if has_y1  else None

    # --- Model locus ---------------------------------------------------------
    P_min_si = float(P_exp.min()) * p_scale
    P_max_si = float(P_exp.max()) * p_scale
    P_pad_si = max((P_max_si - P_min_si) * 0.10, 500.0)  # ≥ 500 Pa padding
    # Sweep HIGH → LOW so the warm-start is seeded inside the data range and
    # propagates naturally into the low-P extrapolation.  Starting from low P
    # with no warm-start fails because the T guess (experimental mean) can be
    # 50+ K above the actual heteroazeotrope temperature at low P.
    P_locus_si = np.linspace(P_max_si + P_pad_si, P_min_si - P_pad_si, 80)
    # Seed with experimental means; T_init from the highest data temperature
    # (close to the heteroazeotrope T at P_max).
    x_I_init  = float(np.nanmean(xI_exp))  if has_xI  else 0.05
    x_II_init = float(np.nanmean(xII_exp)) if has_xII else 0.90
    T_init_K  = float(T_exp.max())   # T_exp is in K (t_scale applied above)
    T_loc, xloc1, xloc2, y1_loc = _vlle_locus(
        result, P_locus_si,
        x_I_init=x_I_init, x_II_init=x_II_init, T_init=T_init_K,
    )
    P_loc_plot = P_locus_si / p_scale   # back to data pressure unit
    valid = ~np.isnan(T_loc)

    # Match model locus branches to experimental phase convention (I vs II)
    # by checking which branch's mean composition is closer to mean xI_exp.
    xI_loc, xII_loc = xloc1, xloc2   # default: feos liquid1 = phase I
    if has_xI and has_xII and valid.any():
        mean_xI_exp  = float(np.nanmean(xI_exp))
        mean_xloc1   = float(np.nanmean(xloc1[valid]))
        mean_xloc2   = float(np.nanmean(xloc2[valid]))
        if abs(mean_xloc2 - mean_xI_exp) < abs(mean_xloc1 - mean_xI_exp):
            xI_loc, xII_loc = xloc2, xloc1  # swap so phase I matches exp

    # --- Colors / markers ----------------------------------------------------
    _color_vap = "#2ca02c"
    _kw_I   = dict(s=45, marker="o", facecolors="white",
                   edgecolors=_EXP_COLOR_1, linewidths=1.2, zorder=5)
    _kw_II  = dict(s=45, marker="s", facecolors="white",
                   edgecolors=_EXP_COLOR_2, linewidths=1.2, zorder=5)
    _kw_vap = dict(s=45, marker="^", facecolors="white",
                   edgecolors=_color_vap, linewidths=1.2, zorder=5)

    # --- Figure: two panels --------------------------------------------------
    fig, (ax_T, ax_P) = plt.subplots(1, 2, figsize=(15, 5))

    def _draw_locus_and_scatter(ax, Y_exp, Y_loc_plot, y_lbl):
        if valid.any():
            Y_v = Y_loc_plot[valid]
            Y_lo = float(np.nanmin(Y_exp))
            Y_hi = float(np.nanmax(Y_exp))
            in_range = (Y_v >= Y_lo) & (Y_v <= Y_hi)
            for x_v, color, label in [
                (xI_loc[valid],  _EXP_COLOR_1, "PC-SAFT (phase I)"),
                (xII_loc[valid], _EXP_COLOR_2, "PC-SAFT (phase II)"),
                (y1_loc[valid],  _color_vap,   "PC-SAFT (vapor)"),
            ]:
                ax.plot(x_v, Y_v, color=color, linestyle="--", alpha=0.40, label=None)
                if in_range.any():
                    ax.plot(x_v[in_range], Y_v[in_range],
                            color=color, linestyle="-", label=label)
        if has_xI:
            ax.scatter(xI_exp,  Y_exp, label=r"Exp. phase I ($x_1$)",  **_kw_I)
        if has_xII:
            ax.scatter(xII_exp, Y_exp, label=r"Exp. phase II ($x_1$)", **_kw_II)
        if has_y1:
            ax.scatter(y1_exp,  Y_exp, label=r"Exp. vapor ($y_1$)",    **_kw_vap)
        ax.set_xlabel(rf"$x_1$, $y_1$ ({result.id1})")
        ax.set_ylabel(y_lbl)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
        sns.despine(ax=ax, offset=10)

    _draw_locus_and_scatter(ax_T, T_exp, T_loc,      "$T$ / K")
    _draw_locus_and_scatter(ax_P, P_exp, P_loc_plot, f"$p$ / {p_lbl}")

    # --- Model markers at experimental pressures (T-x panel) -----------------
    # The VLLE locus in T-x space can appear as a nearly-horizontal smear when
    # the model predicts nearly constant T across pressures.  Plot the model
    # compositions interpolated at each experimental P, placed at the observed T,
    # to show the model-data comparison regardless of model T accuracy.
    if valid.any() and (has_xI or has_xII or has_y1):
        _P_v  = P_locus_si[valid]
        _srt  = np.argsort(_P_v)
        _P_s  = _P_v[_srt]
        _xI_s  = xI_loc[valid][_srt]
        _xII_s = xII_loc[valid][_srt]
        _y1_s  = y1_loc[valid][_srt]
        _P_exp_pa = P_exp * p_scale
        _ok = (_P_exp_pa >= _P_s[0]) & (_P_exp_pa <= _P_s[-1])
        if _ok.any():
            _xI_pred  = np.interp(_P_exp_pa[_ok], _P_s, _xI_s)
            _xII_pred = np.interp(_P_exp_pa[_ok], _P_s, _xII_s)
            _y1_pred  = np.interp(_P_exp_pa[_ok], _P_s, _y1_s)
            _T_ok     = T_exp[_ok]
            _kw_mp = dict(zorder=7, alpha=0.75)
            _first_lbl = "PC-SAFT (at exp. $p$)"
            if has_xI:
                ax_T.scatter(_xI_pred, _T_ok, s=35, marker="o",
                             color=_EXP_COLOR_1, label=_first_lbl, **_kw_mp)
                _first_lbl = None
            if has_xII:
                ax_T.scatter(_xII_pred, _T_ok, s=35, marker="s",
                             color=_EXP_COLOR_2, label=_first_lbl, **_kw_mp)
                _first_lbl = None
            if has_y1:
                ax_T.scatter(_y1_pred, _T_ok, s=35, marker="^",
                             color=_color_vap, label=_first_lbl, **_kw_mp)

    # --- VLE continuation above the heteroazeotrope (T-x panel, median pressure) ---
    if result._record1 is not None and result._record2 is not None:
        P_ref_pa  = float(np.nanmedian(P_exp)) * p_scale   # Pa
        P_ref_obj = P_ref_pa * si.PASCAL
        T_ref_init = float(np.nanmedian(T_exp))             # K
        ha_ref = _find_heteroazeotrope(
            result, P_ref_obj, x_I_init, x_II_init, T_ref_init
        )
        if ha_ref is not None:
            T_het, x1_I_het, x1_II_het, _y1_het = ha_ref
            xb1, Tb1, yd1 = _vle_branch_isobaric(
                result, P_ref_obj, x1_I_het, 1e-5, T_het, npoints=80
            )
            xb2, Tb2, yd2 = _vle_branch_isobaric(
                result, P_ref_obj, x1_II_het, 1.0 - 1e-5, T_het, npoints=80
            )

            def _above_het_vlle(xs, Ts, ys=None):
                mask = [t >= T_het for t in Ts]
                xf = [x for x, ok in zip(xs, mask) if ok]
                Tf = [t for t, ok in zip(Ts, mask) if ok]
                return (xf, Tf, [y for y, ok in zip(ys, mask) if ok]) if ys else (xf, Tf)

            xb1, Tb1, yd1 = _above_het_vlle(xb1, Tb1, yd1)
            xb2, Tb2, yd2 = _above_het_vlle(xb2, Tb2, yd2)

            _p_ref_lbl = f"{P_ref_pa / p_scale:.1f} {p_lbl}"
            if xb1:
                ax_T.plot(xb1, Tb1, color=_EXP_COLOR_1, linestyle="-",
                          label=f"VLE bubble ({_p_ref_lbl})")
            if xb2:
                ax_T.plot(xb2, Tb2, color=_EXP_COLOR_1, linestyle="-")
            if yd1:
                ax_T.plot(yd1, Tb1[:len(yd1)], color=_EXP_COLOR_2, linestyle="-",
                          label=f"VLE dew ({_p_ref_lbl})")
            if yd2:
                ax_T.plot(yd2, Tb2[:len(yd2)], color=_EXP_COLOR_2, linestyle="-")
            ax_T.hlines(T_het, x1_I_het, x1_II_het, colors=_GRAY, linewidth=1.2,
                        linestyle="-", label=f"3-phase ({T_het:.1f} K, {_p_ref_lbl})")

    # --- LLE binodal on the T-x panel (when LLE data is present) ---------------
    has_lle = "lle_T" in data
    if has_lle and result._record1 is not None and result._record2 is not None:
        lle_T = data["lle_T"].astype(float) * t_scale
        T_pad = max((lle_T.max() - lle_T.min()) * 0.05, 1.0)
        z1 = float(np.clip(
            0.5 * ((np.nanmean(data["lle_x1_I"]) if "lle_x1_I" in data else 0.05)
                   + (np.nanmean(data["lle_x1_II"]) if "lle_x1_II" in data else 0.90)),
            0.05, 0.95,
        ))
        _lle_T_max = float(lle_T.max())
        T_c, x_I_c, x_II_c = _lle_curve_kij_T(
            result, z1,
            float(lle_T.min()) - T_pad,
            _lle_T_max + T_pad,
            npoints=301,
        )
        _T_max_show = _lle_T_max + T_pad
        if len(T_c) > 0:
            _T_c = np.asarray(T_c)
            _keep = _T_c <= _T_max_show
            T_c    = _T_c[_keep]
            x_I_c  = np.asarray(x_I_c)[_keep]
            x_II_c = np.asarray(x_II_c)[_keep]

        # Match LLE branch colors to the VLLE locus phase convention.
        # _lle_curve_kij_T returns x_I_c = min-x1 branch, x_II_c = max-x1 branch.
        # xI_loc follows the experimental phase-I label, which may be the large-x1 branch.
        # Swap so that "phase I" color always refers to the same physical phase.
        if valid.any() and len(x_I_c) > 0:
            _mean_vlle_I = float(np.nanmean(xI_loc[valid]))
            _swap_lle = (
                abs(float(np.nanmean(x_II_c)) - _mean_vlle_I)
                < abs(float(np.nanmean(x_I_c)) - _mean_vlle_I)
            )
        else:
            _swap_lle = False
        _lle_I_curve = x_II_c if _swap_lle else x_I_c
        _lle_II_curve = x_I_c if _swap_lle else x_II_c
        _lle_I_key    = "lle_x1_II" if _swap_lle else "lle_x1_I"
        _lle_II_key   = "lle_x1_I"  if _swap_lle else "lle_x1_II"

        if len(T_c) > 0 and np.max(np.abs(x_I_c - x_II_c)) > 1e-4:
            ax_T.plot(_lle_I_curve, T_c, color=_EXP_COLOR_1, linestyle="--",
                      label="PC-SAFT LLE (phase I)", alpha=0.7)
            ax_T.plot(_lle_II_curve, T_c, color=_EXP_COLOR_2, linestyle="--",
                      label="PC-SAFT LLE (phase II)", alpha=0.7)
        if _lle_I_key in data:
            vals = data[_lle_I_key].astype(float)
            valid_lle = ~np.isnan(vals)
            ax_T.scatter(vals[valid_lle], lle_T[valid_lle],
                         s=30, marker="s", facecolors="white",
                         edgecolors=_EXP_COLOR_1, linewidths=1.0, zorder=4,
                         label="Exp. LLE phase I")
        if _lle_II_key in data:
            vals = data[_lle_II_key].astype(float)
            valid_lle = ~np.isnan(vals)
            ax_T.scatter(vals[valid_lle], lle_T[valid_lle],
                         s=30, marker="D", facecolors="white",
                         edgecolors=_EXP_COLOR_2, linewidths=1.0, zorder=4,
                         label="Exp. LLE phase II")

    ax_T.legend(fontsize="x-small", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    fig.suptitle(f"VLLE locus: {result.id1} + {result.id2}", y=1.01)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig, ax_T, ax_P
