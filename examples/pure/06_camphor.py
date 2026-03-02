"""
Example: Fit PC-SAFT parameters to camphor data.

Camphor is a non-associating polar compound (mu fixed at 2.9 D). This example
fits m, sigma, and epsilon_k using vapor pressure, liquid density, and
enthalpy of vaporization data. Psat extrapolation is enabled to handle
near-critical temperatures in the dataset.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent.parent / "data"
psat_path = data_dir / "psat" / "camphor.csv"
density_path = data_dir / "density" / "camphor.csv"
hvap_path = data_dir / "hvap" / "camphor.csv"


def main() -> None:
    result = fit_pure(
        id="camphor",
        psat_path=psat_path,
        density_path=density_path,
        hvap_path=hvap_path,
        mu=2.9,
        extrapolate_psat=True,
    )
    print(result)
    result.to_json("examples/out/examples_pure.json")
    result.plot(path="examples/out/camphor.png")


if __name__ == "__main__":
    main()
