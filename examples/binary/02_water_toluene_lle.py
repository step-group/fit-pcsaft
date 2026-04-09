"""Example: fit k_ij from LLE data, screening all water models.

Loops over every water model in water_models.json, fits k_ij against the
water + toluene LLE data, and saves the best result (lowest ARD) to JSON.

CSV schema (examples/data/lle/):
    temperature_K  — temperature [K]
    x1_I           — mole fraction of component 1 in phase I  (water-lean)
    x1_II          — mole fraction of component 1 in phase II (water-rich)
"""

import json
import shutil
import tempfile
from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_lle

EXAMPLES_DIR = Path(__file__).parent.parent

lle_path = EXAMPLES_DIR / "data" / "lle" / "toluene_water.csv"
water_models_path = EXAMPLES_DIR / "data" / "parameters" / "water_models.json"
binary_params_path = EXAMPLES_DIR / "data" / "parameters" / "binary_params.json"


def main() -> None:
    plots_dir = EXAMPLES_DIR / "out" / "water_toluene_models"
    shutil.rmtree(plots_dir, ignore_errors=True)
    plots_dir.mkdir(parents=True)

    water_models = json.loads(water_models_path.read_text())
    binary_params = json.loads(binary_params_path.read_text())

    # Extract toluene record from binary_params
    toluene_record = next(
        r for r in binary_params if r["identifier"]["name"] == "toluene"
    )

    best_result = None
    best_ard = float("inf")
    best_model_name = None

    import math

    print(f"{'Model':<35} {'N':>4}  {'avg ARD%':>9}  {'min ARD%':>9}  {'max ARD%':>9}")
    print("-" * 75)

    for water in water_models:
        model_name = water["identifier"]["name"]
        combined = [water, toluene_record]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(combined, tmp)
            tmp_path = tmp.name

        try:
            result = fit_kij_lle(
                id1="toluene",
                id2=model_name,
                lle_path=lle_path,
                params_path=tmp_path,
                kij_order=2,
                kij_t_ref=298.15,
                kij_bounds=(-0.3, 0.3),
                temperature_unit=si.KELVIN,
                t_min=300 * si.KELVIN,
            )
            n_pts = len(result.data["T_kij"])
            ard_pw = result.data["ard_pointwise"]
            meaningful = ard_pw[ard_pw > 0.01]
            avg_ard = float(meaningful.mean()) if len(meaningful) > 0 else float("nan")
            min_ard = float(meaningful.min()) if len(meaningful) > 0 else float("nan")
            max_ard = float(meaningful.max()) if len(meaningful) > 0 else float("nan")
            fmt = lambda v: f"{v:>8.2f}%" if not math.isnan(v) else "     N/A "
            print(
                f"{model_name:<35} {n_pts:>4}  {fmt(avg_ard)}  {fmt(min_ard)}  {fmt(max_ard)}"
            )
            result.plot_kij(path=plots_dir / f"{model_name}_kij.png")
            result.plot(path=plots_dir / f"{model_name}_lle.png")
            if not math.isnan(avg_ard) and avg_ard < best_ard:
                best_ard = avg_ard
                best_result = result
                best_model_name = model_name
        except Exception as e:
            print(f"{model_name:<35} {'FAILED':>37}  ({e})")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    print("-" * 75)
    print(f"\nBest model: {best_model_name}  (avg ARD = {best_ard:.2f}%)")
    print(best_result)

    out_json = EXAMPLES_DIR / "out" / "binary_kij.json"
    best_result.to_json(out_json)
    print(f"\nSaved to {out_json}")

    best_result.plot_kij(path=EXAMPLES_DIR / "out" / "water_toluene_kij.png")
    best_result.plot(path=EXAMPLES_DIR / "out" / "water_toluene_lle.png")


if __name__ == "__main__":
    main()
