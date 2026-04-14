"""Example: fit k_ij for 2-methyl-2-butanol + water using VLLE + LLE data.

2-methyl-2-butanol (2m2boh) + water exhibits a heteroazeotrope locus measured
at multiple pressures (Khudaida et al.), making it a good test case for the
conjoined T-x / P-x VLLE plot.

Data:
  VLLE: 323–373 K, 18.6–160.9 kPa (Khudaida et al.)
  LLE:  278–402 K (multiple sources)
"""

from pathlib import Path

import si_units as si

from fit_pcsaft import BinaryKijFitter

EXAMPLES_DIR = Path(__file__).parent.parent
OUT_DIR = EXAMPLES_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

vlle_path = EXAMPLES_DIR / "data" / "vle_lle" / "2m2boh_water_vlle.csv"
lle_path = EXAMPLES_DIR / "data" / "vle_lle" / "2m2boh_water_lle.csv"
params_path = [
    EXAMPLES_DIR / "data" / "parameters" / "examples_pure.json",
    EXAMPLES_DIR / "data" / "parameters" / "binary_params.json",
]


def main() -> None:
    result = (
        BinaryKijFitter(
            "2-methyl-2-butanol",
            "water",
            params_path,
            kij_order=1,
            kij_t_ref=298.15,
        )
        .add_vlle(vlle_path, pressure_unit=si.KILO * si.PASCAL)
        .add_lle(lle_path, require_both_phases=False)
        .fit()
    )

    print(result)

    result.plot(
        path=OUT_DIR / "2m2boh_water_phase_diagram.png",
        pressure_unit=si.KILO * si.PASCAL,
    )
    print(f"\nPhase diagram saved to {OUT_DIR / '2m2boh_water_phase_diagram.png'}")

    result.plot_kij(path=OUT_DIR / "2m2boh_water_kij.png")
    print(f"k_ij plot saved to {OUT_DIR / '2m2boh_water_kij.png'}")


if __name__ == "__main__":
    main()
