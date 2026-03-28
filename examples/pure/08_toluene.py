"""
Example: Fit PC-SAFT parameters to toluene data.

NOTE: numerical Jacobian makes this noticeably slower than other examples.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"
psat_path = data_dir / "psat" / "toluene.csv"
density_path = data_dir / "density" / "toluene.csv"


def main() -> None:
    result = fit_pure(
        id="toluene",
        psat_path=psat_path,
        density_path=density_path,
        loss="cauchy",
        f_scale=0.001,
        q=5.5,
    )
    print(result)
    result.to_json(out_dir / "examples_pure.json")
    result.plot(path=out_dir / "toluene.png")


if __name__ == "__main__":
    main()
