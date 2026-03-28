"""
Example: Fit PC-SAFT parameters to eugenol data.

Eugenol is an associating compound (phenolic OH, 2B scheme: na=2, nb=1).
Uses cauchy loss to handle outliers in the experimental data.
Psat extrapolation is enabled to handle near-critical data points.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"
psat_path = data_dir / "psat" / "eugenol.csv"
density_path = data_dir / "density" / "eugenol.csv"


def main() -> None:
    result = fit_pure(
        id="eugenol",
        psat_path=psat_path,
        density_path=density_path,
        na=2,
        nb=1,
        loss="cauchy",
        f_scale=0.001,
        extrapolate_psat=True,
    )
    print(result)
    result.to_json(out_dir / "examples_pure.json")
    result.plot(path=out_dir / "eugenol.png")


if __name__ == "__main__":
    main()
