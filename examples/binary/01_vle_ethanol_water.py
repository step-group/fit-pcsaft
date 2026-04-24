"""Example: fit k_ij for ethanol + water from VLE bubble-point data.

Data source: 10.1016/j.fluid.2012.12.014
Pure PC-SAFT parameters:
  - Ethanol: fitted via fit_pure (see examples/pure/03_ethanol.py)
  - Water: Gross & Sadowski (2002) literature parameters

CSV schema expected:
  temperature_K, pressure_kPa, x (liquid molefrac comp-1), y (vapor molefrac comp-1)
"""

from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_vle

EXAMPLES_DIR = Path(__file__).parent.parent

vle_path = EXAMPLES_DIR / "data" / "vle" / "ethanol_water_vle.csv"
params_path = EXAMPLES_DIR / "data" / "parameters" / "binary_params.json"

def main() -> None:
    result = fit_kij_vle(
        id1="ethanol",
        id2="water",
        vle_path=vle_path,
        params_path=params_path,
        kij_order=0,
        kij_t_ref=298.15,
        kij_bounds=(-0.2, 0.2),
    )

    print(result)

    out_json = EXAMPLES_DIR / "out" / "binary_kij.json"
    result.to_json(out_json)
    print(f"\nSaved to {out_json}")

    result.plot(
        path=EXAMPLES_DIR / "out" / "ethanol_water_vle.png",
        pressure_unit=si.KILO * si.PASCAL,
    )


if __name__ == "__main__":
    main()
