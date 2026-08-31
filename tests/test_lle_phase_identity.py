"""The LLE paths must not accept a vapour as one of their two liquid phases.

`density_initialization="liquid"` is already passed at every one of these flash
sites, and it does NOT prevent this: it seeds the density root, then `tp_flash`
runs its own stability analysis and converges wherever the equilibrium actually
is. `PhaseEquilibrium` then exposes `.liquid` and `.vapor` whatever the two
phases turned out to be, and the LLE code takes min/max of their compositions,
throwing the labels away. Above the solvent's boiling point tp_flash returns a
genuine vapour-liquid equilibrium and the gas composition lands in the tie line.

1-octanol + water at 423.15 K and 1.013 bar reproduces it from the fixtures in
examples/data -- one phase comes back at Z ~ 1. Offline, no PubChem.
"""
import json
import math
from pathlib import Path

import feos
import numpy as np
import si_units as si

from fit_pcsaft import BinaryFitResult, fit_kij_lle
from fit_pcsaft._binary._utils import (
    LIQUID_Z_MAX,
    _build_binary_eos,
    _is_liquid,
    _lle_split,
    _load_pure_records,
)

DATA = Path(__file__).parent.parent / "examples" / "data"
PARAMS = [DATA / "parameters" / "alkanols_lle.json",
          DATA / "parameters" / "water_models.json"]

# Where this pair, at k_ij = 0, returns one liquid and one gas.
T_VLE_K, P_ATM_BAR, Z_VLE = 423.15, 1.01325, 0.884


def _water_name():
    path = DATA / "parameters" / "water_models.json"
    return json.loads(path.read_text())[0]["identifier"]["name"]


def _eos():
    r1, r2 = _load_pure_records(PARAMS, "1-octanol", _water_name())
    return _build_binary_eos(r1, r2, 0.0)


def _state(T_K, pressure_bar, z1, init="liquid"):
    return feos.State(
        _eos(),
        T_K * si.KELVIN,
        pressure=pressure_bar * si.BAR,
        composition=np.array([z1, 1.0 - z1]) * si.MOL,
        density_initialization=init,
    )


def test_liquid_initialization_does_not_guarantee_a_liquid_phase():
    """The premise the old code rested on. Seeding the root liquid is a guess,
    not a constraint -- the flash still converges to a vapour-liquid split."""
    pe = _state(T_VLE_K, P_ATM_BAR, Z_VLE).tp_flash(max_iter=1000)
    assert _is_liquid(pe.liquid)
    assert not _is_liquid(pe.vapor), "expected a gas phase from a liquid-seeded flash"


def test_z_separates_a_gas_from_a_liquid():
    """Z rather than an absolute density floor: an ideal gas at 50 bar is
    1272 mol/m3, denser than any floor that would still pass a real liquid."""
    assert 0.0 < LIQUID_Z_MAX < 1.0
    assert _is_liquid(_state(298.15, P_ATM_BAR, 0.5, init="liquid"))
    assert not _is_liquid(_state(473.15, 0.1, 0.5, init="vapor"))


def test_the_guard_rejects_the_split_the_old_code_accepted():
    """The defect in one assertion: the same PhaseEquilibrium yields a tie line
    with the guard off and nothing with it on."""
    pe = _state(T_VLE_K, P_ATM_BAR, Z_VLE).tp_flash(max_iter=1000)
    assert _lle_split(pe, require_liquid_phases=False) is not None
    assert _lle_split(pe, require_liquid_phases=True) is None


def test_the_guard_reaches_the_fit_and_improves_it():
    """Not merely plumbed through: on this fixture, refusing the vapour splits
    sends least_squares somewhere materially different."""
    def run(**kw):
        return fit_kij_lle(
            id1="1-octanol",
            id2=_water_name(),
            lle_path=DATA / "lle" / "1-octanol_water.csv",
            params_path=PARAMS,
            kij_order=0,
            require_both_phases=False,
            **kw,
        )

    loose, guarded = run(), run(require_liquid_phases=True)
    assert math.isfinite(guarded.kij_at(298.15))
    assert guarded.ard < 0.5 * loose.ard


def test_the_result_records_the_conditions_the_fit_ran_at():
    """fit_kij_lle already took `pressure=` and dropped it on the floor, so
    residuals() and the plotted binodal each re-predicted at their own
    hardcoded pressure -- 1.01325 and 1.0 bar, neither necessarily the fit's.
    The defaults must stay those historical values."""
    assert BinaryFitResult.lle_pressure_bar == 1.01325
    assert BinaryFitResult.lle_require_liquid_phases is False

    result = fit_kij_lle(
        id1="1-octanol",
        id2=_water_name(),
        lle_path=DATA / "lle" / "1-octanol_water.csv",
        params_path=PARAMS,
        kij_order=0,
        require_both_phases=False,
        pressure=50.0 * si.BAR,
        require_liquid_phases=True,
    )
    assert result.lle_pressure_bar == 50.0
    assert result.lle_require_liquid_phases is True
