"""
Example: Fit PC-SAFT parameters to ethanol data.

Ethanol is an associating compound (hydrogen bonding). This example fits
m, sigma, epsilon_k, kappa_ab, and epsilon_k_ab using the 2B association
scheme (na=1, nb=1), with mu fixed at 0.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent.parent / "data"
psat_path = data_dir / "psat" / "ethanol.csv"
density_path = data_dir / "density" / "ethanol.csv"


def main() -> None:
    result = fit_pure(
        id="ethanol",
        psat_path=psat_path,
        density_path=density_path,
        loss="arctan",
        f_scale=0.001,
        na=1,
        nb=1,
    )
    print(result)
    result.to_json("examples/out/examples_pure.json")
    result.plot(path="examples/out/ethanol.png")


if __name__ == "__main__":
    main()
