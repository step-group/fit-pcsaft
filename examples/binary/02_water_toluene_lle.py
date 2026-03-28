"""Example: fit k_ij from LLE data, screening all water models.

Loops over every water model in water_models.json, fits k_ij against the
water + toluene LLE data, and saves the best result (lowest ARD) to JSON.

CSV schema (examples/data/lle/):
    temperature_K  — temperature [K]
    x1_I           — mole fraction of component 1 in phase I  (water-lean)
    x1_II          — mole fraction of component 1 in phase II (water-rich)
"""

import json
import tempfile
from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_lle

EXAMPLES_DIR = Path(__file__).parent.parent

lle_path = EXAMPLES_DIR / "data" / "lle" / "water_toluene.csv"
water_models_path = EXAMPLES_DIR / "data" / "parameters" / "water_models.json"
binary_params_path = EXAMPLES_DIR / "data" / "parameters" / "binary_params.json"


def main() -> None:
    water_models = json.loads(water_models_path.read_text())
    binary_params = json.loads(binary_params_path.read_text())

    # Extract toluene record from binary_params
    toluene_record = next(
        r for r in binary_params if r["identifier"]["name"] == "toluene"
    )

    best_result = None
    best_ard = float("inf")
    best_model_name = None

    print(f"{'Model':<35} {'k_ij0':>8}  {'ARD%':>7}")
    print("-" * 55)

    for water in water_models:
        model_name = water["identifier"]["name"]
        combined = [water, toluene_record]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(combined, tmp)
            tmp_path = tmp.name

        try:
            result = fit_kij_lle(
                id1=model_name,
                id2="toluene",
                lle_path=lle_path,
                params_path=tmp_path,
                kij_order=2,
                kij_t_ref=298.15,
                kij_bounds=(-0.2, 0.2),
                temperature_unit=si.KELVIN,
            )
            ard = result.ard
            print(f"{model_name:<35} {result.kij_coeffs[0]:>8.4f}  {ard:>7.2f}%")
            plots_dir = EXAMPLES_DIR / "out" / "water_models"
            plots_dir.mkdir(parents=True, exist_ok=True)
            result.plot(path=plots_dir / f"{model_name}.png")
            if ard < best_ard:
                best_ard = ard
                best_result = result
                best_model_name = model_name
        except Exception as e:
            print(f"{model_name:<35} {'FAILED':>17}  ({e})")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    print("-" * 55)
    print(f"\nBest model: {best_model_name}  (ARD = {best_ard:.2f}%)")
    print(best_result)

    out_json = EXAMPLES_DIR / "out" / "binary_kij.json"
    best_result.to_json(out_json)
    print(f"\nSaved to {out_json}")

    best_result.plot(path=EXAMPLES_DIR / "out" / "water_toluene_lle.png")


if __name__ == "__main__":
    main()
