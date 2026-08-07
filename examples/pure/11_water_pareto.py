"""Example: pareto-optimize PC-SAFT parameters for water using surface tension.

Reproduces the strategy of Rehner & Gross, J. Chem. Eng. Data 2020, 65,
5698-5707: bulk-phase error (AAD_vle) and interfacial error (AAD_sft) are
treated as two separate objectives, and the published parameter set is the
point on the resulting front that minimises

    AAD_vle / ref_vle + AAD_sft / ref_sft      (their eq 32)

with ref_vle = 2% and ref_sft = 0.7 mN/m for water.

The data files are quasi-data generated from reference correlations, as the
paper does, so the fit is not biased toward the temperature range where raw
experimental points happen to be dense:
  - psat, rho : Wagner & Pruss (1993) saturation line
  - sft       : IAPWS R1-76

Runtime: roughly 5-10 minutes. Objective evaluations cost about 120 ms at
sensible parameters and up to 1 s where the VLE solve fails, so wall time
falls as the population converges. Reduce pop_size/n_gen to make it quicker.
"""

from pathlib import Path

from fit_pcsaft import fit_pure_pareto

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"

REF_VLE = 2.0   # %
REF_SFT = 0.7   # mN/m


def main() -> None:
    result = fit_pure_pareto(
        id="water",
        psat_path=data_dir / "psat" / "water.csv",
        density_path=data_dir / "density" / "water.csv",
        sft_path=data_dir / "surface_tension" / "water.csv",
        na=1,
        nb=1,           # 2B association scheme
        pop_size=40,
        n_gen=30,
        verbose=False,
    )
    print(result)

    result.to_csv(out_dir / "water_pareto_front.csv")
    result.plot(path=out_dir / "water_pareto.png", ref_vle=REF_VLE, ref_sft=REF_SFT)

    best = result.select(ref_vle=REF_VLE, ref_sft=REF_SFT)
    print()
    print(best)
    print(
        "\nRehner & Gross Table 1, water PC-SAFT 2B, for comparison:\n"
        "  m=1.0000  sigma=2.9375 A  epsilon_k=272.03 K  "
        "kappa_ab=0.044480  epsilon_k_ab=3125.3 K"
    )

    best.to_json(out_dir / "examples_pure.json")
    best.plot(path=out_dir / "water_pareto_best.png")


if __name__ == "__main__":
    main()
