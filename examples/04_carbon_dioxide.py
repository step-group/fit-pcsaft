"""
Example: Fit PC-SAFT parameters to carbon dioxide data.

Carbon dioxide has a non-negligible quadrupolar moment. This example fits
m, sigma, and epsilon_k, with q (quadrupole moment) fixed at 4.4 AD.

NOTE: this one takes 5-10 minutes.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent / "data"
psat_path = data_dir / "psat" / "co2.csv"
density_path = data_dir / "density" / "co2.csv"


def main() -> None:
    result = fit_pure(
        id="carbon dioxide",
        psat_path=psat_path,
        density_path=density_path,
        q=4.4,
    )
    print(result)
    result.to_json("examples/out/examples_pure.json")
    result.plot(path="examples/out/co2.png")


if __name__ == "__main__":
    main()
