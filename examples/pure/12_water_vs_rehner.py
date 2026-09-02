"""Example: our pareto-selected water parameters against Rehner & Gross' published set.

Both parameter sets are only *evaluated* here -- nothing is fitted, so this runs
in seconds. Our set is read from the front written by 11_water_pareto.py, so the
two examples stay in sync; run that one first.

Three panels, because two of them cannot tell the sets apart. Vapour pressure
and saturated density are what a conventional PC-SAFT fit already matches, and
both sets sit on the data there. Surface tension is the objective the paper
added, and it is the panel where a parameter set can be wrong in a way the bulk
properties never reveal -- which is the whole argument of the paper.
"""

from pathlib import Path

import feos
import numpy as np
import polars as pl
import si_units as si

from fit_pcsaft import eval_pure
from fit_pcsaft._pure.pareto import _argmin_scalarized
from fit_pcsaft._pure.surface_tension import predict_surface_tension

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"
FRONT = out_dir / "water_pareto_front.csv"

REF_VLE, REF_SFT = 2.0, 0.7  # paper's water weights, eq 32

# Rehner & Gross, J. Chem. Eng. Data 2020, 65, 5698-5707, Table 1: water, 2B.
REHNER_2B = {
    "m": 1.0000,
    "sigma": 2.9375,
    "epsilon_k": 272.03,
    "kappa_ab": 0.044480,
    "epsilon_k_ab": 3125.3,
}


def _selected_from_front(path: Path) -> dict:
    """The eq-32 tangent point of the saved front, as an eval_pure params dict."""
    df = pl.read_csv(path)
    F = df.select(["aad_vle_pct", "aad_sft"]).to_numpy()
    row = df.row(_argmin_scalarized(F, (REF_VLE, REF_SFT)), named=True)
    return {k: row[k] for k in ("m", "sigma", "epsilon_k", "kappa_ab", "epsilon_k_ab")}


def _evaluate(params: dict):
    return eval_pure(
        id="water",
        psat_path=data_dir / "psat" / "water.csv",
        density_path=data_dir / "density" / "water.csv",
        sft_path=data_dir / "surface_tension" / "water.csv",
        params=params,
        na=1,
        nb=1,
    )


def _plot(results, path):
    """Top row: the properties. Bottom row: the residuals.

    The top row alone cannot separate the two parameter sets -- both lie on the
    data at plotting resolution, which is exactly the degeneracy the paper is
    about. The residual row is where a factor of two in AARD becomes visible.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from fit_pcsaft.result import _predict_per_property

    sns.set_context("talk")
    sns.set_style("ticks")

    ref = results[0][1]  # both share the same experimental data
    d, u = ref.data, ref.units
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5),
                            gridspec_kw={"height_ratios": [1.35, 1.0]})
    top, bot = axes[0], axes[1]

    exp_kw = dict(s=42, marker="o", facecolors="white", edgecolors="#E32F2F",
                  linewidths=1.2, zorder=5, label="IAPWS")
    top[0].scatter(1000.0 / d.T_psat, np.log(d.p_psat), **exp_kw)
    top[1].scatter(d.rho, d.T_rho, **exp_kw)
    top[2].scatter(d.T_sft, d.sft, **exp_kw)

    # gamma needs its own grid: the model curve is smooth, the data is 25 points
    T_fine = np.linspace(float(d.T_sft.min()), float(d.T_sft.max()), 60)

    for (label, res, color, ls) in results:
        pd_ = feos.PhaseDiagram.pure(res.eos, float(d.T_psat.min()) * u.temperature, 401)
        line = dict(color=color, ls=ls, lw=2.0, zorder=4)
        top[0].plot(1000.0 / (pd_.vapor.temperature / si.KELVIN),
                    np.log(pd_.vapor.pressure / u.pressure), label=label, **line)
        top[1].plot(pd_.vapor.mass_density / u.density,
                    pd_.vapor.temperature / si.KELVIN, label=label, **line)
        top[1].plot(pd_.liquid.mass_density / u.density,
                    pd_.liquid.temperature / si.KELVIN, **line)
        top[2].plot(T_fine, predict_surface_tension(res.functional, T_fine, u),
                    label=label, **line)

        preds = _predict_per_property(res.eos, res.compound.mw, d, u, functional=res.functional)
        pt = dict(color=color, marker="o" if ls == "-" else "s", ms=6, lw=1.4,
                  ls=ls, label=label, zorder=4)
        for k, (prop, T) in enumerate((("psat", d.T_psat), ("rho", d.T_rho))):
            model, exp = preds[prop]
            bot[k].plot(T, 100.0 * (model - exp) / exp, **pt)
        # gamma residuals stay absolute: gamma -> 0 at Tc, so a relative
        # deviation diverges there and would hide everything below 550 K
        model, exp = preds["sft"]
        bot[2].plot(d.T_sft, model - exp, **pt)

    top[0].set(xlabel=r"$1000/T$ / K$^{-1}$", ylabel=r"$\ln(p_\mathrm{sat}$ / kPa$)$",
               title="Saturation pressure")
    top[1].set(xlabel=r"$\rho$ / kg m$^{-3}$", ylabel=r"$T$ / K", title=r"$T$–$\rho$")
    top[2].set(xlabel=r"$T$ / K", ylabel=r"$\gamma$ / mN m$^{-1}$",
               title="Surface tension")
    # T-rho spans the full dome; the data only covers the liquid branch
    top[1].set_xlim(-30.0, 1.05 * float(d.rho.max()))

    bot[0].set(xlabel=r"$T$ / K", ylabel=r"RD / %", title=r"$p_\mathrm{sat}$ residual")
    bot[1].set(xlabel=r"$T$ / K", ylabel=r"RD / %", title=r"$\rho$ residual")
    bot[2].set(xlabel=r"$T$ / K", ylabel=r"$\Delta\gamma$ / mN m$^{-1}$",
               title=r"$\gamma$ residual (absolute)")
    for ax in bot:
        ax.axhline(0.0, color="gray", lw=0.9, ls=":", zorder=1)
    for ax in axes.ravel():
        ax.legend(loc="best", framealpha=0.9, fontsize=12)

    fig.suptitle("Water, PC-SAFT 2B — pareto selection vs Rehner & Gross (2020)", y=1.0)
    sns.despine(offset=8)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig, axes


def main() -> None:
    if not FRONT.exists():
        raise SystemExit(f"{FRONT} not found — run examples/pure/11_water_pareto.py first")

    ours = _selected_from_front(FRONT)
    results = [
        ("this work", _evaluate(ours), "#1F77B4", "-"),
        ("Rehner & Gross", _evaluate(REHNER_2B), "#2CA02C", "--"),
    ]

    print(f"{'':>16}  {'m':>7} {'sigma':>7} {'eps/k':>7} {'kappa_ab':>9} {'eps_ab/k':>9}"
          f"  {'AARD psat':>9} {'AARD rho':>9} {'AAD_sft':>8} {'eq 32':>6}")
    for label, res, _, _ in results:
        p = res.params
        aad_vle = 0.5 * (res.ard_psat + res.ard_rho)
        print(f"{label:>16}  {p['m']:7.4f} {p['sigma']:7.4f} {p['epsilon_k']:7.2f} "
              f"{p['kappa_ab']:9.6f} {p['epsilon_k_ab']:9.1f}  "
              f"{res.ard_psat:8.2f}% {res.ard_rho:8.2f}% {res.aad_sft:8.2f} "
              f"{aad_vle / REF_VLE + res.aad_sft / REF_SFT:6.3f}")

    path = out_dir / "water_vs_rehner.png"
    _plot(results, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
