"""Example: a three-objective PC-SAFT front for water -- psat, rho and liquid cp.

objectives=("psat", "rho", "cp") puts the two bulk AARDs of Forte et al. 2018 on
their own axes and adds a third: the AARD of the *total* liquid isobaric heat
capacity. PC-SAFT is a residual model, so the total is feos's residual cp at
(T, P) plus an ideal-gas part -- here feos's DIPPR-107 (Aly-Lee) model with the
five coefficients passed as cp_ig, in DIPPR units J/(kmol K). Nothing in the
search touches the DFT: the triple runs on the same batched, in-process feos
path as the bulk pair, so a 60 x 60 budget is seconds, not minutes.

The cp data file has no pressure column: the quasi-data is the saturated liquid
(IAPWS-95), and a missing or blank P means "at saturation pressure" to the
loader. A real data set measured at 1 atm would carry a P column instead.

Three-objective details worth knowing before reading the output:
  - pop_size is rounded up to the Das-Dennis fan: 60 -> 66 weight vectors.
  - refs defaults to (2, 2, 2) %, under which select() returns the point with
    the lowest *mean* of the three AARDs -- the pooled best, not a trade-off
    pick. Pass refs that reflect what each property can be trusted to.
  - .plot() draws three pairwise projections coloured by the third axis; the
    eq-32 tangent has no line to draw on a surface.
  - Every MOEA/D knob in fit_pcsaft was measured with two objectives. The
    triple has not been benchmarked; treat one run's front as a first look.

Regenerate the data with examples/data/generate_water_reference.py.
Runtime: about 4 s on this machine, of which the search itself is 2 s
(66 x 60 evaluations plus refine=4), against roughly 5 minutes for the
DFT-solving ("vle", "sft") default in 11_water_pareto.py.
"""

from pathlib import Path

from fit_pcsaft import fit_pure_pareto

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"

# DIPPR 107 for water, J/(kmol K): A, B, C, D, E. Reproduces the IAPWS-95
# ideal-gas cp to 0.03% at 300 K.
WATER_DIPPR107 = [33363.0, 26790.0, 2610.5, 8896.0, 1169.0]
REFS = (2.0, 2.0, 2.0)   # % on each axis


def main() -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = fit_pure_pareto(
        id="water",
        psat_path=data_dir / "psat" / "water.csv",
        density_path=data_dir / "density" / "water.csv",
        cp_path=data_dir / "heat_capacity" / "water.csv",
        cp_ig=WATER_DIPPR107,
        na=1,
        nb=1,           # 2B association scheme
        objectives=("psat", "rho", "cp"),
        refs=REFS,
        pop_size=60,    # -> 66, the next Das-Dennis fan
        n_gen=60,
        verbose=False,
    )
    print(result)

    result.to_csv(out_dir / "water_cp_pareto_front.csv")
    result.plot(path=out_dir / "water_cp_pareto.png")

    best = result.select(refs=REFS)
    print()
    print(best)
    best.to_json(out_dir / "examples_pure.json")


if __name__ == "__main__":
    main()
