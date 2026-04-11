"""Example: fit k_ij for MIBK + water using both VLE and LLE data.

MIBK (methyl isobutyl ketone) + water exhibits a heteroazeotrope at ~1 atm,
making it an ideal test case for simultaneous VLE + LLE fitting.

Data:
  VLE: isobaric at ~101.325 kPa (Cho et al.)
  LLE: 273–373 K (Stephenson; Gross et al.)

Pure PC-SAFT parameters:
  MIBK:  examples/data/parameters/examples_pure.json  (name: "mibk")
  Water: examples/data/parameters/binary_params.json  (name: "water")
         (induced association applied — MIBK is nb=1, water is self-associating)
"""

from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_vle_lle

EXAMPLES_DIR = Path(__file__).parent.parent

vle_path = EXAMPLES_DIR / "data" / "vle_lle" / "mibk_water_vle.csv"
lle_path = EXAMPLES_DIR / "data" / "vle_lle" / "mibk_water_lle.csv"
params_path = [
    EXAMPLES_DIR / "data" / "parameters" / "examples_pure.json",
    EXAMPLES_DIR / "data" / "parameters" / "binary_params.json",
]


def main() -> None:
    result = fit_kij_vle_lle(
        id1="mibk",
        id2="water",
        vle_path=vle_path,
        lle_path=lle_path,
        params_path=params_path,
        kij_order=0,
        kij_t_ref=298.15,
        kij_bounds=(-0.3, 0.3),
        temperature_unit=si.KELVIN,
        pressure_unit=si.KILO * si.PASCAL,
        induced_assoc=True,
        require_both_phases=False,
    )

    print(result)

    out_json = EXAMPLES_DIR / "out" / "binary_kij.json"
    result.to_json(out_json)
    print(f"\nSaved to {out_json}")

    result.plot_kij(path=EXAMPLES_DIR / "out" / "mibk_water_kij.png")

    result.plot(path=EXAMPLES_DIR / "out" / "mibk_water_vle_lle.png")


if __name__ == "__main__":
    main()
