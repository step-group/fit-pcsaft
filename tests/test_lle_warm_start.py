"""fit_kij_lle warm-starts each temperature from the previous k_ij.

The 13-point coarse k_ij scan is the fallback, not the default. Before this,
it ran at every temperature, and each scan point outside the miscibility gap
walked all ~54 feeds through a failing tp_flash: measured at 98 s of 115 s
wall over 26 water + alkanol / toluene fits. Offline, no PubChem.
"""
from pathlib import Path

import fit_pcsaft._binary.lle as lle

DATA = Path(__file__).parent.parent / "examples" / "data"


def test_scan_runs_only_for_the_first_temperature(monkeypatch):
    """Residual calls made before each least_squares: 13 (the scan) for the
    first temperature, then exactly 1 (the warm start) for every other."""
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
    assert before_lsq == [lle._N_KIJ_SCAN] + [1] * (n_T - 1)
