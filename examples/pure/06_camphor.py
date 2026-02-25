"""
Example: Fit PC-SAFT parameters to 1-decanol data.

1-decanol is an associating compound (hydrogen bonding). This example fits
m, sigma, epsilon_k, kappa_ab, and epsilon_k_ab using the 2B association
scheme (na=1, nb=1), with mu fixed at 0.
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
