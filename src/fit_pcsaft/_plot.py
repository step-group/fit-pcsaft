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

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

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
    ax.legend()

    # --- right: T-ρ ---
    ax = axes[1]
    ax.set_title(rf"$T$–$\rho$ diagram — {name}")
    ax.plot(rho_vap, T_pd, label="PC-SAFT", **ln_kw)
    ax.plot(rho_liq, T_pd, **ln_kw)
    ax.scatter(result.data.rho, result.data.T_rho, label="Experiment", **sc_kw)
    ax.set_xlabel(r"$\rho$ / kg m$^{-3}$")
    ax.set_ylabel(r"$T$ / K")
    ax.legend()

    sns.despine(offset=10)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, axes
