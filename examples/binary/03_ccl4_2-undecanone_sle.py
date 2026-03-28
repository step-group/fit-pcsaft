"""Example: fit k_ij from SLE (solid-liquid equilibrium / solubility) data.

Uses the Schröder-van Laar equation:
    ln(x_solid) + ln_gamma_solid = -(dHfus/R) * (1/T - 1/Tm)

This is a eutectic system: both CCl4 and 2-undecanone can crystallise.
The code auto-assigns each data point to the branch (solid 1 or solid 2)
that gives the smaller residual.

  id1 = tetrachloromethane  Tm = 250.77 K   dHfus =  3.273 kJ/mol  (solid_index=0)
  id2 = 2-undecanone        Tm = 285.84 K   dHfus = 34.544 kJ/mol

CSV schema (examples/data/sle/):
    temperature_K  — temperature [K]
    x              — mole fraction of id1 (CCl4) in the saturated liquid

ARD is reported as mean(|x1_pred - x1_data|) / mean(x1_data) * 100.

Pure parameters JSON must contain entries for both components.
See examples/pure/ for how to generate it with fit_pure().
"""

from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_sle

EXAMPLES_DIR = Path(__file__).parent.parent

sle_path = EXAMPLES_DIR / "data" / "sle" / "ccl4_2-undecanone.csv"
params_path = EXAMPLES_DIR / "data" / "parameters" / "binary_params.json"


def main() -> None:
    result = fit_kij_sle(
        id1="tetrachloromethane",  # must match name in params JSON
        id2="2-undecanone",  # must match name in params JSON
        sle_path=sle_path,
        params_path=params_path,
        # Branch 1: CCl4 (id1) is solid
        tm=250.77 * si.KELVIN,
        delta_hfus=3.273 * si.KILO * si.JOULE / si.MOL,
        solid_index=0,
        # Branch 2: 2-undecanone (id2) is solid — eutectic
        tm2=285.84 * si.KELVIN,
        delta_hfus2=34.544 * si.KILO * si.JOULE / si.MOL,
        kij_order=0,
        kij_t_ref=298.0,  # near the eutectic temperature
        kij_bounds=(-0.2, 0.2),
    )

    print(result)

    out_json = EXAMPLES_DIR / "out" / "binary_kij.json"
    result.to_json(out_json)
    print(f"\nSaved to {out_json}")

    result.plot(path=EXAMPLES_DIR / "out" / "ccl4_2-undecanone_sle.png")


if __name__ == "__main__":
    main()
