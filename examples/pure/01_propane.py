"""
Minimal example: Fit PC-SAFT parameters to propane data.

This example demonstrates the simplest use case: fitting vapor pressure
and liquid density data for a non-associating compound.
"""

from pathlib import Path

from fit_pcsaft import fit_pure

# Data paths (relative to this file)
data_dir = Path(__file__).parent.parent / "data"
psat_path = data_dir / "psat" / "propane.csv"
density_path = data_dir / "density" / "propane.csv"
hvap_path = data_dir / "hvap" / "propane.csv"


def main() -> None:
    # Fit PC-SAFT parameters
    result = fit_pure(
        id="propane",
        psat_path=psat_path,
        density_path=density_path,
        hvap_path=hvap_path,
        hvap_weight=1.0,
        loss="arctan",
        f_scale=0.001,
    )
    print(result)
    result.to_json("examples/out/examples_pure.json")
    result.plot(path="examples/out/propane.png", line_color="black")


if __name__ == "__main__":
    main()
