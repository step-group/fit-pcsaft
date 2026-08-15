"""Minimal plotting for pure component PC-SAFT fit results."""

import feos
import numpy as np
import si_units as si

# Named presets
_COLOR_PRESETS: dict[str, str] = {
    "red": "#E32F2F",
    "blue": "#1F77B4",
    "green": "#2CA02C",
    "orange": "#FF7F0E",
    "purple": "#9467BD",
    "cyan": "#17BECF",
    "black": "#000000",
}
_DEFAULT_EXP_COLOR = _COLOR_PRESETS["red"]


def _plot_pure(
    result,
    path=None,
    color: str = "red",
    line_color: str = "black",
    linestyle: str = "-",
    scatter_kw: dict | None = None,
    line_kw: dict | None = None,
):
    """Two-panel phase diagram: Clausius-Clapeyron + T-ρ, with experimental data.

    Parameters
    ----------
    result : FitResult

    path : str or Path, optional
        If given, save the figure to this path.

    color : str
        Colour for experimental data points. Either a preset name
        ("red", "blue", "green", "orange", "purple", "cyan", "black")
        or any matplotlib colour string (hex, named, etc.).
        Default: "red".

    line_color : str
        Colour for the PC-SAFT curves. Accepts the same preset names or
        any matplotlib colour string. Default: "black".

    linestyle : str
        Line style for the PC-SAFT curves, e.g. "-", "--", "-.", ":".
        Default: "-".

    scatter_kw : dict, optional
        Extra kwargs merged into both scatter calls, overriding defaults.
        Example: ``scatter_kw={"s": 80, "marker": "^"}``.

    line_kw : dict, optional
        Extra kwargs merged into both line plot calls, overriding defaults.
        Example: ``line_kw={"linewidth": 2}``.

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    exp_color = _COLOR_PRESETS.get(color, color)
    eos_color = _COLOR_PRESETS.get(line_color, line_color)

    _scatter_defaults = dict(
        s=40,
        marker="o",
        facecolors="white",
        edgecolors=exp_color,
        linewidths=1.2,
        zorder=5,
    )
    _line_defaults = dict(color=eos_color, linestyle=linestyle)

    sc_kw = {**_scatter_defaults, **(scatter_kw or {})}
    ln_kw = {**_line_defaults, **(line_kw or {})}

    tu = result.units.temperature
    pu = result.units.pressure
    du = result.units.density

    all_T = [result.data.T_psat, result.data.T_rho, result.data.T_hvap]
    T_start = float(min(T.min() for T in all_T if len(T) > 0)) * tu
    phase_diagram = feos.PhaseDiagram.pure(result.eos, T_start, 501)

    T_pd = phase_diagram.vapor.temperature / si.KELVIN
    p_pd = phase_diagram.vapor.pressure / pu
    rho_vap = phase_diagram.vapor.mass_density / du
    rho_liq = phase_diagram.liquid.mass_density / du

    name = result.input_name

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- left: Clausius-Clapeyron ---
    ax = axes[0]
    ax.set_title(f"Saturation pressure — {name}")
    ax.plot(1000.0 / T_pd, np.log(p_pd), label="PC-SAFT", **ln_kw)
    ax.scatter(
        1000.0 / result.data.T_psat,
        np.log(result.data.p_psat),
        label="Experiment",
        **sc_kw,
    )
    ax.set_xlabel(r"$1000/T$ / K$^{-1}$")
    ax.set_ylabel(r"$\ln(p_\mathrm{sat})$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    # --- right: T-ρ ---
    ax = axes[1]
    ax.set_title(rf"$T$–$\rho$ diagram — {name}")
    ax.plot(rho_vap, T_pd, label="PC-SAFT", **ln_kw)
    ax.plot(rho_liq, T_pd, **ln_kw)
    ax.scatter(result.data.rho, result.data.T_rho, label="Experiment", **sc_kw)
    ax.set_xlabel(r"$\rho$ / kg m$^{-3}$")
    ax.set_ylabel(r"$T$ / K")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, axes


def _plot_residuals_pure(result, path=None):
    """Signed RD% vs temperature for psat, rho, and hvap."""
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    from fit_pcsaft.result import _compute_per_point_rd

    sns.set_context("talk")
    sns.set_style("ticks")

    df = _compute_per_point_rd(
        result.eos, result.data, result.units,
        functional=getattr(result, "functional", None),
    )
    present = set(df["property"].to_list())

    _props = [
        ("psat", "Vapor pressure",    "#1F77B4", "o"),
        ("rho",  "Liquid density",    "#E32F2F", "s"),
        ("hvap", "Enthalpy of vap.",  "#2CA02C", "^"),
        ("sft",  "Surface tension",   "#FF7F0E", "D"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for prop, label, color, marker in _props:
        if prop not in present:
            continue
        sub = df.filter(pl.col("property") == prop)
        ax.scatter(
            sub["T"].to_numpy(),
            sub["rd_pct"].to_numpy(),
            label=label,
            color=color,
            marker=marker,
            s=50,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("$T$ / K")
    ax.set_ylabel(r"RD% = (model $-$ exp) / exp $\times$ 100")
    ax.set_title(f"Residuals — {result.input_name}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax


def _plot_pareto(result, path=None, refs: tuple[float, float] | None = None):
    """Objective 2 vs objective 1 with the selected point and its tangent (paper Fig. 1).

    Axis names, and which unit each carries, come from ``result.objectives``
    read through ``_OBJECTIVES`` -- so a ``("psat", "rho")`` front is labeled
    AARD_psat / AARD_rho, not the ``("vle", "sft")`` defaults. Plain-text axis
    labels on purpose (e.g. ``"AAD_sft / mN m^-1"``, not LaTeX): these are
    diagnostic PNGs, never thesis figures.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from fit_pcsaft._pure.pareto import _DEFAULT_REFS, _OBJECTIVES, _argmin_scalarized

    objectives = tuple(result.objectives)
    if refs is None:
        refs = _DEFAULT_REFS[objectives]

    sns.set_context("talk")
    sns.set_style("ticks")

    F = result.F
    i = _argmin_scalarized(F, refs)

    # _OBJECTIVES stores the prose unit ("mN/m"); axis labels use plot
    # notation instead -- the only remap, not a new _OBJECTIVES field.
    _axis_unit = {"mN/m": "mN m^-1"}
    (n0, u0, _), (n1, u1, _) = (_OBJECTIVES[k] for k in objectives)
    xlabel = f"{n0} / {_axis_unit.get(u0, u0)}"
    ylabel = f"{n1} / {_axis_unit.get(u1, u1)}"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(F[:, 0], F[:, 1], "-o", color="#1F77B4", ms=5, lw=1.5,
            label=result.input_name or "pareto front")
    ax.scatter([F[i, 0]], [F[i, 1]], s=160, marker="X", color="#E32F2F",
               zorder=6, label="selected")

    # Axis limits come from the front alone. The tangent below is then clipped
    # to them: drawn across the full width it runs far past the data (and well
    # below zero), which rescales the axes and flattens the curve into a line.
    def _limits(v):
        lo, hi = float(v.min()), float(v.max())
        pad = 0.08 * (hi - lo) if hi > lo else max(0.08 * abs(hi), 0.1)
        return lo - pad, hi + pad

    xlim, ylim = _limits(F[:, 0]), _limits(F[:, 1])

    # tangent of slope -refs[1]/refs[0] through the selected point
    slope = -refs[1] / refs[0]
    x_at = [F[i, 0] + (y - F[i, 1]) / slope for y in ylim]
    x_t = np.clip(np.sort(x_at), *xlim)
    ax.plot(x_t, F[i, 1] + slope * (x_t - F[i, 0]),
            ls="--", color="gray", lw=1.0, zorder=1,
            label="eq-32 tangent")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Pareto front — {result.input_name}")
    ax.legend(loc="best", framealpha=0.9)

    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax
