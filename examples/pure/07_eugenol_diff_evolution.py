"""
Example: Fit PC-SAFT parameters to eugenol data using differential evolution.

TODO: code body is currently a placeholder (camphor). Update to use eugenol
data and fit_pure_de.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent.parent / "data"
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
    result.to_json("examples/out/examples_pure.json")
    result.plot(path="examples/out/eugenol.png")


if __name__ == "__main__":
    main()
