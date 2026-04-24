"""
Example: Fit PC-SAFT parameters to 1-decanol data.

1-decanol is an associating compound (hydrogen bonding). This example fits
m, sigma, epsilon_k, kappa_ab, and epsilon_k_ab using the 2B association
scheme (na=1, nb=1), with mu fixed at 0.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"
psat_path = data_dir / "psat" / "1-decanol.csv"
density_path = data_dir / "density" / "1-decanol.csv"


def main() -> None:
    result = fit_pure(
        id="1-decanol",
        psat_path=psat_path,
        density_path=density_path,
        na=1,
        nb=1,
        extrapolate_psat=True,
    )
    print(result)
    result.to_json(out_dir / "examples_pure.json")
    result.plot(path=out_dir / "1-decanol.png")


if __name__ == "__main__":
    main()
