"""Example: fit k_ij from Henry's law constant data.

System: 2-methyl-3-buten-2-ol (1) + water (2)

Convention: component 1 is the solute (infinite-dilution limit),
            component 2 is the solvent.

CSV schema expected:
  T, H
  where T is temperature [K] and H is the Henry's law constant [atm].
  Adjust temperature_unit and henry_unit to match your data.
"""

from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_henry

EXAMPLES_DIR = Path(__file__).parent.parent

henry_path = EXAMPLES_DIR / "data" / "henry" / "2m3b2oh.csv"
params_path = EXAMPLES_DIR / "data" / "parameters" / "binary_params.json"

# Component identifiers — must match names in params JSON
SOLUTE = "2-methyl-3-buten-2-ol"
SOLVENT = "water"


def main() -> None:
    result = fit_kij_henry(
        id1=SOLUTE,
        id2=SOLVENT,
        henry_path=henry_path,
        params_path=params_path,
        kij_order=0,
        kij_bounds=(-0.5, 0.5),
        temperature_unit=si.KELVIN,
        henry_unit=1.01325 * si.BAR,
    )

    print(result)

    out_json = EXAMPLES_DIR / "out" / "binary_kij.json"
    result.to_json(out_json)
    result.plot(
        path=EXAMPLES_DIR / "out" / "2m3b2oh_water_henry.png",
        henry_unit=1.01325 * si.BAR,
    )
    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    main()
