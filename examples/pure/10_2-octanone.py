"""
Example: Fit PC-SAFT parameters to 2-octanone data.

2-octanone is a polar compound (ketone, mu ~ 2.7 D). Data from multiple
literature sources with scatter; cauchy loss downweights outliers.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

data_dir = Path(__file__).parent.parent / "data" / "2-octanone"
out_dir = Path(__file__).parent.parent / "out"


def main() -> None:
    result = fit_pure(
        id="2-octanone",
        psat_path=data_dir / "psat_cleaned.csv",
        density_path=data_dir / "density_cleaned.csv",
        mu=None,  # fit dipole moment (ketone, ~2.7 D)
        loss="cauchy",
        f_scale=0.01,
    )
    print(result)
    result.to_json(out_dir / "examples_pure.json")
    result.plot(path=out_dir / "2-octanone.png")


if __name__ == "__main__":
    main()
