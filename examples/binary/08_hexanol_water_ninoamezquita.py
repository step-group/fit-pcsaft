"""Example: 1-hexanol + water LLE with the Nino-Amezquita 2012 water model.

Fits k_ij(T) (quadratic) to experimental tieline data and plots the
phase envelope on a log-odds composition axis.

Data source: examples/data/lle/1-hexanol_water.csv
Water model: water4C_ninoamezquita2012 (4C scheme)
1-hexanol:   Gross & Sadowski (2001), 2B association scheme
"""

import json
import tempfile
from pathlib import Path

import si_units as si

from fit_pcsaft import fit_kij_lle

EXAMPLES_DIR = Path(__file__).parent.parent

lle_path = EXAMPLES_DIR / "data" / "lle" / "1-hexanol_water.csv"
water_models_path = EXAMPLES_DIR / "data" / "parameters" / "water_models.json"

# PC-SAFT parameters for 1-hexanol (Gross & Sadowski, Ind. Eng. Chem. Res. 2001)
# 2B association scheme: 1 donor site, 1 acceptor site
HEXANOL_RECORD = {
    "identifier": {
        "cas": "111-27-3",
        "name": "1-hexanol",
        "iupac_name": "hexan-1-ol",
        "smiles": "CCCCCCO",
        "inchi": "InChI=1S/C6H14O/c1-2-3-4-5-6-7/h7H,2-6H2,1H3",
        "formula": "C6H14O",
    },
    "molarweight": 102.17,
    "m": 2.6895,
    "sigma": 4.0593,
    "epsilon_k": 294.85,
    "association_sites": [
        {
            "na": 1.0,
            "nb": 1.0,
            "kappa_ab": 0.002417,
            "epsilon_k_ab": 2942.29,
        }
    ],
}


def main() -> None:
    water_models = json.loads(water_models_path.read_text())
    water = next(
        w for w in water_models
        if w["identifier"]["name"] == "water4C_ninoamezquita2012"
    )

    combined = [HEXANOL_RECORD, water]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(combined, tmp)
        tmp_path = tmp.name

    try:
        result = fit_kij_lle(
            id1="1-hexanol",
            id2="water4C_ninoamezquita2012",
            lle_path=lle_path,
            params_path=tmp_path,
            kij_order=2,
            kij_t_ref=298.15,
            kij_bounds=(-0.2, 0.2),
            temperature_unit=si.KELVIN,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print(result)

    out_dir = EXAMPLES_DIR / "out"
    out_dir.mkdir(exist_ok=True)

    result.plot_kij(path=out_dir / "hexanol_water_ninoamezquita_kij.png")
    result.plot(
        path=out_dir / "hexanol_water_ninoamezquita_lle.png",
        plot_unfitted=True,
    )
    print(f"\nPlots saved to {out_dir}")


if __name__ == "__main__":
    main()
