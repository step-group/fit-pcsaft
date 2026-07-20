"""Fast offline smoke tests for binary k_ij fitting.

Pure records come from local JSON (no PubChem). One VLE fit and one LLE fit;
the LLE fit uses a SINGLE water model (the first in water_models.json), never all.
Regression-style, loose bounds.
"""
import json
import math
from pathlib import Path

from fit_pcsaft import BinaryFitResult, fit_kij_lle, fit_kij_vle

DATA = Path(__file__).parent.parent / "examples" / "data"


def test_kij_vle_ethanol_water():
    result = fit_kij_vle(
        id1="ethanol",
        id2="water",
        vle_path=DATA / "vle" / "ethanol_water_vle.csv",
        params_path=DATA / "parameters" / "binary_params.json",
        kij_order=0,
        kij_bounds=(-0.2, 0.2),
    )
    assert isinstance(result, BinaryFitResult)
    kij = result.kij_at(298.15)
    assert math.isfinite(kij)
    assert -0.2 <= kij <= 0.2
    assert math.isfinite(result.ard) and result.ard < 20.0


def test_kij_lle_octanol_water_single_model():
    water_models_path = DATA / "parameters" / "water_models.json"
    water_name = json.loads(water_models_path.read_text())[0]["identifier"]["name"]

    result = fit_kij_lle(
        id1="1-octanol",
        id2=water_name,
        lle_path=DATA / "lle" / "1-octanol_water.csv",
        params_path=[DATA / "parameters" / "alkanols_lle.json", water_models_path],
        kij_order=0,
        require_both_phases=False,
    )
    assert isinstance(result, BinaryFitResult)
    assert math.isfinite(result.kij_at(298.15))
    assert math.isfinite(result.ard)
