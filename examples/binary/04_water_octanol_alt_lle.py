"""Example: fit k_ij from LLE data for 1-octanol + water, screening all water models.

Alternative 1-octanol parameters (2B association scheme) from the table with
m=4.0012, sigma=3.8282, epsilon_k=268.73, kappa_ab=0.003110, epsilon_k_ab=2804.3.
"""

import json
import math
import shutil
import tempfile
from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_lle

EXAMPLES_DIR = Path(__file__).parent.parent

lle_path = EXAMPLES_DIR / "data" / "lle" / "1-octanol_water.csv"
water_models_path = EXAMPLES_DIR / "data" / "parameters" / "water_models.json"

# PC-SAFT parameters for 1-octanol, 2B association scheme (alternative set)
OCTANOL_RECORD = {
    "identifier": {
        "cas": "111-87-5",
        "name": "1-octanol",
        "iupac_name": "octan-1-ol",
        "smiles": "CCCCCCCCO",
        "inchi": "InChI=1S/C8H18O/c1-2-3-4-5-6-7-8-9/h9H,2-8H2,1H3",
        "formula": "C8H18O",
    },
    "molarweight": 130.230,
    "m": 4.0012,
    "sigma": 3.8282,
    "epsilon_k": 268.73,
    "association_sites": [
        {
            "na": 1.0,
            "nb": 1.0,
            "kappa_ab": 0.003110,
            "epsilon_k_ab": 2804.3,
        }
    ],
}


def main() -> None:
    plots_dir = EXAMPLES_DIR / "out" / "water_octanol_alt_models"
    shutil.rmtree(plots_dir, ignore_errors=True)
    plots_dir.mkdir(parents=True)

    water_models = json.loads(water_models_path.read_text())

    best_result = None
    best_ard = float("inf")
    best_model_name = None

    print(f"{'Model':<35} {'N':>4}  {'avg ARD%':>9}  {'min ARD%':>9}  {'max ARD%':>9}")
    print("-" * 75)

    for water in water_models:
        model_name = water["identifier"]["name"]
        combined = [OCTANOL_RECORD, water]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(combined, tmp)
            tmp_path = tmp.name

        try:
            result = fit_kij_lle(
                id1="1-octanol",
                id2=model_name,
                lle_path=lle_path,
                params_path=tmp_path,
                kij_order=2,
                kij_t_ref=298.15,
                kij_bounds=(-0.2, 0.2),
                temperature_unit=si.KELVIN,
                require_both_phases=False,
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
            result.plot(path=plots_dir / f"{model_name}_lle.png", plot_unfitted=True)
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

    best_result.plot_kij(path=EXAMPLES_DIR / "out" / "water_octanol_alt_kij.png")
    best_result.plot(
        path=EXAMPLES_DIR / "out" / "water_octanol_alt_lle.png", plot_unfitted=True
    )


if __name__ == "__main__":
    main()
