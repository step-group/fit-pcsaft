"""fit_kij_lle warm-starts each temperature from the previous k_ij, and the
scan it falls back to walks down from the upper bound and stops once it has
left the gap.

Before this the 13-point scan ran at every temperature, and each scan point
outside the miscibility gap walked all ~54 feeds through a failing tp_flash:
measured at 98 s of 115 s wall over 26 water + alkanol / toluene fits.
Offline, no PubChem.
"""
from pathlib import Path
from types import SimpleNamespace

import feos
import si_units as si

import fit_pcsaft._binary.lle as lle

DATA = Path(__file__).parent.parent / "examples" / "data"


def test_scan_runs_only_for_the_first_temperature_and_stops_early(monkeypatch):
    """Residual calls made before each least_squares: the scan for the first
    temperature, which stops before all 13 grid points because toluene's gap
    reaches the upper bound, then exactly 1 (the warm start) for every other."""
    n_outside, before_lsq = [0], []
    real_resid, real_lsq = lle._residuals_at_T, lle.least_squares

    def resid(*a, **k):
        n_outside[0] += 1
        return real_resid(*a, **k)

    def lsq(*a, **k):
        before_lsq.append(n_outside[0])
        n_outside[0] = 0
        monkeypatch.setattr(lle, "_residuals_at_T", real_resid)  # don't count inside
        try:
            return real_lsq(*a, **k)
        finally:
            monkeypatch.setattr(lle, "_residuals_at_T", resid)

    monkeypatch.setattr(lle, "_residuals_at_T", resid)
    monkeypatch.setattr(lle, "least_squares", lsq)

    result = lle.fit_kij_lle("toluene", "water", DATA / "lle" / "toluene_water.csv",
                             DATA / "parameters" / "binary_params.json", kij_order=1)

    n_T = len(result.data["T_kij"])
    assert n_T > 1
    assert 1 < before_lsq[0] < lle._N_KIJ_SCAN, before_lsq
    assert before_lsq[1:] == [1] * (n_T - 1)


def test_anchor_flash_is_not_solved_once_a_converged_equilibrium_exists(monkeypatch):
    """Every feed flash is warm-started from the last converged equilibrium.
    The anchor flash at the lowest temperature is the fallback while none
    exists, so on a data set whose first temperature converges it never runs.

    feos takes only mole fractions and the phase fraction from initial_state
    and rebuilds both phases on the feed's own EOS (feos-core tp_flash.rs,
    update_states), so an equilibrium from another k_ij is a valid guess."""
    current_T = [None]
    anchor_flashes = [0]
    T_anchor = 273.2  # lowest temperature in toluene_water.csv

    real_resid = lle._residuals_at_T

    def resid(kij_arr, T_K, *a, **k):
        current_T[0] = T_K
        return real_resid(kij_arr, T_K, *a, **k)

    class State:
        def __init__(self, eos, T, **k):
            self._s = feos.State(eos, T, **k)
            self._T = float(T / si.KELVIN)

        def tp_flash(self, **k):
            if abs(self._T - T_anchor) < 1e-6 and current_T[0] > T_anchor + 0.5:
                anchor_flashes[0] += 1
            return self._s.tp_flash(**k)

    monkeypatch.setattr(lle, "_residuals_at_T", resid)
    monkeypatch.setattr(lle, "feos", SimpleNamespace(State=State))

    result = lle.fit_kij_lle("toluene", "water", DATA / "lle" / "toluene_water.csv",
                             DATA / "parameters" / "binary_params.json", kij_order=1)

    assert len(result.data["T_kij"]) > 1
    assert anchor_flashes[0] == 0


def test_single_phase_rows_keep_the_anchor():
    """Rows with one composition have several exact k_ij roots, and the flash
    start decides which one the scan's best sample lands on: with the carried
    hint 22 of 575 such rows moved, and 1-octanol + water_esper2023 dropped
    428.2 K outright when the hint was a vapour-liquid split from 419.3 K
    (residual ~500 at every k_ij). They keep the per-call anchor, so this fit
    reproduces the pre-warm-start temperatures."""
    result = lle.fit_kij_lle(
        "1-octanol", "water_esper2023", DATA / "lle" / "1-octanol_water.csv",
        [DATA / "parameters" / "alkanols_lle.json", DATA / "parameters" / "water_models.json"],
        kij_order=2, kij_bounds=(-0.2, 0.2), require_both_phases=False,
    )
    assert 428.2 in set(round(float(T), 1) for T in result.data["T_kij"])
