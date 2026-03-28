"""Binary phase diagram plotting for BinaryFitResult."""

import numpy as np
import si_units as si

_LINE_COLOR = "#000000"
_EXP_COLOR_1 = "#E32F2F"   # liquid / phase I
_EXP_COLOR_2 = "#1F77B4"   # vapor / phase II

_R = 8.314462618  # J/(mol·K)


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


def _plot_binary(result, path=None, temperature_unit=si.KELVIN, pressure_unit=si.KILO * si.PASCAL, henry_unit=si.MEGA * si.PASCAL):
    eq = result.equilibrium_type
    if eq == "vle":
        return _plot_vle(result, path, temperature_unit, pressure_unit)
    elif eq == "lle":
        return _plot_lle(result, path, temperature_unit)
    elif eq == "sle":
        return _plot_sle(result, path, temperature_unit)
    elif eq == "henry":
        return _plot_henry(result, path, temperature_unit, henry_unit)
    else:
        raise ValueError(f"Unknown equilibrium_type: {eq!r}")


# ---------------------------------------------------------------------------
# VLE
# ---------------------------------------------------------------------------

def _plot_vle(result, path, temperature_unit, pressure_unit):
    import feos
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    data = result.data
    t_scale = float(temperature_unit / si.KELVIN)
    T_data = data["T"].astype(float) * t_scale          # → K
    P_data = data["P"].astype(float)
    x1_data = data["x1"].astype(float)
    has_y1 = "y1" in data

    # Is the data approximately isobaric?
    p_cv = np.std(P_data) / np.mean(P_data)  # coefficient of variation
    is_isobaric = p_cv < 0.05

    fig, ax = plt.subplots(figsize=(8, 6))
    p_lbl = _pressure_label(pressure_unit)

    if is_isobaric:
        # T-x-y diagram at mean pressure
        P_mean = float(np.mean(P_data))
        try:
            vle_pd = feos.PhaseDiagram.binary_vle(
                result.eos, P_mean * pressure_unit, npoints=200
            )
            T_curve = vle_pd.liquid.temperature / si.KELVIN
            ax.plot(vle_pd.liquid.molefracs[:, 0], T_curve,
                    color=_LINE_COLOR, linestyle="-", label="PC-SAFT (bubble)")
            ax.plot(vle_pd.vapor.molefracs[:, 0], T_curve,
                    color=_LINE_COLOR, linestyle="--", label="PC-SAFT (dew)")
        except Exception:
            pass

        ax.scatter(x1_data, T_data, label=r"Exp. $x_1$", **_scatter_kw(_EXP_COLOR_1))
        if has_y1:
            ax.scatter(data["y1"].astype(float), T_data,
                       label=r"Exp. $y_1$", **_scatter_kw(_EXP_COLOR_2, "^"))
        ax.set_xlabel(rf"$x_1,\,y_1$ ({result.id1})")
        ax.set_ylabel("$T$ / K")
        ax.set_xlim(0, 1)
        ax.set_title(f"VLE: {result.id1} + {result.id2}  ($p$ = {P_mean:.0f} {p_lbl})")

    else:
        # Multi-isothermal P-x-y diagram: one curve per unique temperature
        unique_Ts = sorted(np.unique(np.round(T_data, 0)))
        cmap = plt.cm.plasma
        colors = [cmap(i / max(1, len(unique_Ts) - 1)) for i in range(len(unique_Ts))]

        for k, T_iso in enumerate(unique_Ts):
            color = colors[k]
            mask = np.abs(T_data - T_iso) < 0.6
            try:
                vle_iso = feos.PhaseDiagram.binary_vle(
                    result.eos, T_iso * si.KELVIN, npoints=200
                )
                P_curve = vle_iso.liquid.pressure / pressure_unit
                ax.plot(vle_iso.liquid.molefracs[:, 0], P_curve, color=color, linestyle="-")
                ax.plot(vle_iso.vapor.molefracs[:, 0], P_curve, color=color, linestyle="--")
            except Exception:
                pass

            ax.scatter(x1_data[mask], P_data[mask],
                       label=f"{T_iso:.0f} K", **_scatter_kw(color))
            if has_y1:
                ax.scatter(data["y1"].astype(float)[mask], P_data[mask],
                           **_scatter_kw(color, "^"))

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

def _plot_lle(result, path, temperature_unit):
    import feos
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    data = result.data
    t_scale = float(temperature_unit / si.KELVIN)
    T_data = data["T"].astype(float) * t_scale
    has_I = "x1_I" in data
    has_II = "x1_II" in data

    T_min = float(T_data.min())
    T_max = float(T_data.max())
    T_pad = max((T_max - T_min) * 0.05, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    def _log_odds(x):
        x = np.asarray(x, dtype=float)
        return np.log10(np.clip(x, 1e-15, 1.0) / np.clip(1.0 - x, 1e-15, 1.0))

    try:
        lle_pd = feos.PhaseDiagram.lle(
            result.eos,
            1.0 * si.BAR,
            feed=np.array([0.5, 0.5]) * si.MOL,
            min_tp=(T_min - T_pad) * si.KELVIN,
            max_tp=(T_max + T_pad) * si.KELVIN,
            npoints=200,
        )
        T_curve = lle_pd.liquid.temperature / si.KELVIN
        x_liq = lle_pd.liquid.molefracs[:, 0]
        x_vap = lle_pd.vapor.molefracs[:, 0]
        if len(T_curve) > 0 and np.max(np.abs(x_liq - x_vap)) > 1e-3:
            ax.plot(_log_odds(x_liq), T_curve,
                    color=_LINE_COLOR, linestyle="-", label="PC-SAFT (phase I)")
            ax.plot(_log_odds(x_vap), T_curve,
                    color=_LINE_COLOR, linestyle="--", label="PC-SAFT (phase II)")
    except BaseException:
        pass

    if has_I:
        ax.scatter(_log_odds(data["x1_I"].astype(float)), T_data,
                   label="Exp. phase I", **_scatter_kw(_EXP_COLOR_1))
    if has_II:
        ax.scatter(_log_odds(data["x1_II"].astype(float)), T_data,
                   label="Exp. phase II", **_scatter_kw(_EXP_COLOR_2, "^"))

    ax.set_xlabel(rf"$\log_{{10}}(x_1/x_2)$  ({result.id1} / {result.id2})")
    ax.set_ylabel("$T$ / K")
    ax.set_title(f"LLE: {result.id1} + {result.id2}")
    ax.legend(fontsize="small")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
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
            if np.isfinite(diffs[i]) and np.isfinite(diffs[i + 1]) and diffs[i] * diffs[i + 1] < 0:
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

    fig, ax = plt.subplots(figsize=(8, 6))

    if eutectic:
        # Find eutectic point to properly clip each branch
        T_eut, x1_eut = _find_eutectic(
            result.eos,
            Tm_K, dHfus_J, solid_index,
            Tm2_K, dHfus2_J, solid_index2,
        )
        T_start = T_eut if not np.isnan(T_eut) else T_min * 0.995

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
                ax.plot(x1_curve, T_curve, color=_LINE_COLOR, linestyle=linestyle, label=label)

        solid_name = result.id2 if solid_index == 1 else result.id1
        solid_name2 = result.id1 if solid_index == 1 else result.id2
        x0_eut = x1_eut if not np.isnan(x1_eut) else float(x1_data[np.argmin(T_data)])
        _plot_branch(Tm_K, dHfus_J, solid_index, x0_eut, f"PC-SAFT ({solid_name})", "-")
        _plot_branch(Tm2_K, dHfus2_J, solid_index2, x0_eut, f"PC-SAFT ({solid_name2})", "--")

        if not np.isnan(T_eut):
            ax.scatter(
                [x1_eut], [T_eut],
                marker="D", s=60, color=_LINE_COLOR, zorder=6,
                label=f"Eutectic ({x1_eut:.3f}, {T_eut:.1f} K)",
            )

        title = f"SLE: {result.id1} + {result.id2}  (eutectic)"
    else:
        solid_name = result.id2 if solid_index == 1 else result.id1
        x1_curve, T_curve = [], []
        T_range = np.linspace(T_min * 0.995, Tm_K, 120)
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
            ax.plot(x1_curve, T_curve, color=_LINE_COLOR, label="PC-SAFT")
        title = f"SLE: {result.id1} + {result.id2}  (solid: {solid_name})"

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
            feos.Parameters.new_pure(result._solvent_record)
        )

    def _h_pred(T_K: float) -> "float | None":
        try:
            H_pa = feos.State.henrys_law_constant_binary(
                result.eos, T_K * si.KELVIN
            ) / si.PASCAL
            if use_molfrac:
                if eos_solvent is None:
                    return None
                pvap = feos.PhaseEquilibrium.vapor_pressure(
                    eos_solvent, T_K * si.KELVIN
                )[0] / si.PASCAL
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
