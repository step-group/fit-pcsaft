"""Example: fit k_ij for MIBK + water across all available water PC-SAFT models.

Loops over every entry in water_models.json, fits a constant k_ij from the
combined VLE + LLE dataset, and saves per-model plots and a summary table.

Output goes to examples/out/mibk_water_models/.
"""

import re
from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_vle_lle

EXAMPLES_DIR = Path(__file__).parent.parent
OUT_DIR = EXAMPLES_DIR / "out" / "mibk_water_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

vle_path = EXAMPLES_DIR / "data" / "vle_lle" / "mibk_water_vle.csv"
lle_path = EXAMPLES_DIR / "data" / "vle_lle" / "mibk_water_lle.csv"
params_path = [
    EXAMPLES_DIR / "data" / "parameters" / "examples_pure.json",
    EXAMPLES_DIR / "data" / "parameters" / "water_models.json",
]

# All water model names from water_models.json
import json

with open(EXAMPLES_DIR / "data" / "parameters" / "water_models.json") as f:
    _entries = json.load(f)
WATER_MODELS = [e["identifier"]["name"] for e in _entries]


def main() -> None:
    results = {}

    for water_model in WATER_MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Water model: {water_model}")
        print(f"{'=' * 60}")
        try:
            result = fit_kij_vle_lle(
                id1="mibk",
                id2=water_model,
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
            result.plot(path=OUT_DIR / f"{water_model}_vle_lle.png")
            result.plot_kij(path=OUT_DIR / f"{water_model}_kij.png")
            results[water_model] = result
        except Exception as e:
            print(f"  FAILED: {e}")
            results[water_model] = None

    # --- Summary table ---
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Model':<35} {'k_ij0':>8}  {'ARD VLE':>9}  {'ARD LLE':>9}  {'ARD comb':>10}"
    )
    print("-" * 76)
    for name, res in results.items():
        if res is None:
            print(f"{name:<35}  FAILED")
        else:
            kij0 = res.kij_coeffs[0]
            ard_vle = (
                float(res.data["ard_vle"][0]) if "ard_vle" in res.data else float("nan")
            )
            ard_lle = (
                float(res.data["ard_lle"][0]) if "ard_lle" in res.data else float("nan")
            )
            print(
                f"{name:<35} {kij0:>8.4f}  {ard_vle:>8.2f}%  {ard_lle:>8.2f}%  {res.ard:>9.2f}%"
            )


if __name__ == "__main__":
    main()
