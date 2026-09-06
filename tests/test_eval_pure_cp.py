"""eval_pure scores a stored parameter set against liquid cp when asked, and stays
cp-free when not -- the opt-in the TESIS pipeline's s11 needs to reproduce the cp rows
its s04 writes straight off the Pareto FitResult."""

import numpy as np
import polars as pl
import pytest

from fit_pcsaft._fit_utils import ideal_gas_cp, predict_bulk
from fit_pcsaft._pure import fit as fit_mod
from fit_pcsaft._pure.fit import eval_pure
from fit_pcsaft._pure.pareto import _build_eos
from fit_pcsaft._types import PureData, Units
from tests.test_pareto import HEXANE_DIPPR107
from tests.test_surface_tension import HEXANE, HEXANE_P, HEXANE_SPEC

PARAMS = {"m": 3.0576, "sigma": 3.7983, "epsilon_k": 236.77}  # == HEXANE_P


@pytest.fixture
def hexane_files(tmp_path, monkeypatch):
    """Offline: identity from the fixture, cp file generated from HEXANE_P itself."""
    monkeypatch.setattr(fit_mod, "_fetch_compound", lambda _id: (HEXANE.identifier, HEXANE.mw))
    (tmp_path / "psat.csv").write_text("T,psat\n300.0,21.9\n320.0,43.9\n340.0,79.5\n")
    (tmp_path / "rho.csv").write_text("T,rho\n300.0,654.9\n320.0,635.6\n340.0,615.4\n")
    T = np.array([300.0, 320.0, 340.0])
    probe = PureData(T_psat=T, p_psat=np.ones(3), T_rho=T, rho=np.ones(3), T_cp=T,
                     P_cp=np.full(3, np.nan), cp=np.ones(3),
                     cp_ig=ideal_gas_cp(HEXANE, HEXANE_DIPPR107, T))
    cp = predict_bulk(_build_eos(HEXANE_P, HEXANE, HEXANE_SPEC), HEXANE.mw, probe, Units())["cp"]
    (tmp_path / "cp.csv").write_text(
        "T,cp\n" + "\n".join(f"{t},{c:.6f}" for t, c in zip(T, cp)) + "\n"
    )
    return tmp_path


def test_eval_pure_scores_cp_when_given_cp_path(hexane_files):
    res = eval_pure("hexane", hexane_files / "psat.csv", hexane_files / "rho.csv",
                    params=PARAMS, cp_path=hexane_files / "cp.csv", cp_ig=HEXANE_DIPPR107)
    assert res.metrics["cp"].n == 3
    # the file was generated from these very parameters, so the total-cp AARD is ~0
    assert res.ard_cp == pytest.approx(0.0, abs=1e-3)
    cp_rows = res.residuals().filter(pl.col("property") == "cp")
    assert sorted(cp_rows["T"]) == [300.0, 320.0, 340.0]


def test_eval_pure_without_cp_path_is_unchanged(hexane_files):
    res = eval_pure("hexane", hexane_files / "psat.csv", hexane_files / "rho.csv", params=PARAMS)
    assert res.metrics["cp"].n == 0
    assert res.residuals().filter(pl.col("property") == "cp").is_empty()


def test_eval_pure_cp_path_without_cp_ig_is_an_error(hexane_files):
    with pytest.raises(ValueError, match="cp_ig"):
        eval_pure("hexane", hexane_files / "psat.csv", hexane_files / "rho.csv",
                  params=PARAMS, cp_path=hexane_files / "cp.csv")
