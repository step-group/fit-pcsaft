"""
Example: Fit PC-SAFT parameters to acetone data.

Acetone is a polar compound (mu ~ 2.88 D). This example fits
m, sigma, epsilon_k, and mu simultaneously using the non-associating
PC-SAFT variant with dipole term.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent / "data"
psat_path = data_dir / "psat" / "acetone.csv"
density_path = data_dir / "density" / "acetone.csv"
hvap_path = data_dir / "hvap" / "acetone.csv"


def main() -> None:
    result = fit_pure(
        id="acetone",
        psat_path=psat_path,
        density_path=density_path,
        hvap_path=hvap_path,
        na=1,
        mu=None,  # fit dipole moment
    )
    print(result)
    result.to_json("examples/out/examples_pure.json")
    result.plot(path="examples/out/acetone.png")


if __name__ == "__main__":
    main()
