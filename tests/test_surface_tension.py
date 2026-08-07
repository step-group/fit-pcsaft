from pathlib import Path

import numpy as np
import pytest

from fit_pcsaft._csv import load_sft_csv


def test_load_sft_csv_canonical(tmp_path: Path):
    p = tmp_path / "g.csv"
    p.write_text("T,sft\n300.0,71.7\n350.0,63.2\n")
    T, g = load_sft_csv(p)
    assert np.allclose(T, [300.0, 350.0])
    assert np.allclose(g, [71.7, 63.2])


@pytest.mark.parametrize("header", ["gamma", "surface_tension", "sigma_st", "st"])
def test_load_sft_csv_aliases(tmp_path: Path, header: str):
    p = tmp_path / "g.csv"
    p.write_text(f"T,{header}\n300.0,71.7\n")
    T, g = load_sft_csv(p)
    assert np.allclose(T, [300.0]) and np.allclose(g, [71.7])


def test_load_sft_csv_missing_column(tmp_path: Path):
    p = tmp_path / "g.csv"
    p.write_text("T,psat\n300.0,3.5\n")
    with pytest.raises(ValueError):
        load_sft_csv(p)


import feos
import si_units as si

from fit_pcsaft._fit_utils import _build_eos, _build_functional
from fit_pcsaft._pure.surface_tension import (
    SurfaceTensionOptions,
    predict_surface_tension,
)
from fit_pcsaft._types import Compound, ModelSpec, Units

HEXANE = Compound(identifier=feos.Identifier(name="hexane", cas="110-54-3"), mw=86.177)
HEXANE_P = np.array([3.0576, 3.7983, 236.77])          # m, sigma, epsilon_k
HEXANE_SPEC = ModelSpec(mu=0.0, na=None, nb=None, q=0.0)

# Rehner & Gross 2020, Table 1 — water, PC-SAFT, 2B
WATER = Compound(identifier=feos.Identifier(name="water", cas="7732-18-5"), mw=18.015)
WATER_2B = np.array([1.0000, 2.9375, 272.03, 0.044480, 3125.3])
WATER_2B_SPEC = ModelSpec(mu=0.0, na=1, nb=1, q=0.0)


def iapws_sft_mN_m(T):
    """IAPWS R1-76 surface tension of water. Returns mN/m."""
    tau = 1.0 - np.asarray(T, dtype=float) / 647.096
    return 235.8 * tau ** 1.256 * (1.0 - 0.625 * tau)


def test_puredata_defaults_empty_sft():
    from fit_pcsaft._types import FitConfig, PureData, Units
    d = PureData(
        T_psat=np.array([300.0]), p_psat=np.array([3.5]),
        T_rho=np.array([300.0]), rho=np.array([789.0]),
    )
    assert d.T_sft.size == 0 and d.sft.size == 0
    assert Units().surface_tension is not None
    assert FitConfig().w_sft == 1.0
    assert FitConfig().f_scale["sft"] == 1.0


def test_hexane_reference_value():
    func = _build_functional(HEXANE_P, HEXANE, HEXANE_SPEC)
    g = predict_surface_tension(func, np.array([300.0]), Units())
    assert g[0] == pytest.approx(17.598992, abs=0.01)


def test_returns_nan_above_critical():
    func = _build_functional(HEXANE_P, HEXANE, HEXANE_SPEC)
    g = predict_surface_tension(func, np.array([300.0, 2000.0]), Units())
    assert np.isfinite(g[0]) and np.isnan(g[1])


def test_grid_independence():
    func = _build_functional(HEXANE_P, HEXANE, HEXANE_SPEC)
    coarse = predict_surface_tension(func, np.array([300.0]), Units(),
                                     SurfaceTensionOptions(n_grid=256))
    fine = predict_surface_tension(func, np.array([300.0]), Units(),
                                   SurfaceTensionOptions(n_grid=1024))
    assert coarse[0] == pytest.approx(fine[0], rel=1e-5)


def test_functional_matches_eos_bulk():
    """The DFT functional must reproduce the EOS vapour pressure in the bulk limit."""
    eos = _build_eos(HEXANE_P, HEXANE, HEXANE_SPEC)
    func = _build_functional(HEXANE_P, HEXANE, HEXANE_SPEC)
    p_eos = feos.PhaseEquilibrium.vapor_pressure(eos, 300.0 * si.KELVIN)[0] / si.PASCAL
    p_fun = feos.PhaseEquilibrium.vapor_pressure(func, 300.0 * si.KELVIN)[0] / si.PASCAL
    assert p_eos == pytest.approx(p_fun, rel=1e-8)


def test_water_2b_forward_check_against_paper():
    """Rehner & Gross Table 2: PC-SAFT 2B water, AAD_DFT = 1.59 mN/m.

    Measured here: 1.60 mN/m. The bound is 2.5 rather than 1.6 because our
    T-grid differs from the paper's and feos hardcodes psi_dft = 1.3862.
    """
    func = _build_functional(WATER_2B, WATER, WATER_2B_SPEC)
    T = np.linspace(280.0, 630.0, 15)
    model = predict_surface_tension(func, T, Units())
    exp = iapws_sft_mN_m(T)
    ok = np.isfinite(model)
    assert ok.sum() >= 12, "surface tension failed at too many temperatures"
    aad = float(np.mean(np.abs(model[ok] - exp[ok])))
    print(f"\nwater 2B PC-SAFT AAD_DFT = {aad:.2f} mN/m (paper: 1.59)")
    assert aad < 2.5


def _sft_config(w_sft=1.0):
    from fit_pcsaft._types import FitConfig
    return FitConfig(w_sft=w_sft,
                     f_scale={"psat": 1.0, "rho": 1.0, "hvap": 1.0, "sft": 1.0})


def _hexane_data_with_sft():
    from fit_pcsaft._types import PureData
    T = np.array([300.0, 320.0, 340.0])
    func = _build_functional(HEXANE_P, HEXANE, HEXANE_SPEC)
    gamma = predict_surface_tension(func, T, Units())
    return PureData(
        T_psat=np.array([300.0, 320.0]),
        p_psat=np.array([21.9, 43.9]),
        T_rho=np.array([300.0, 320.0]),
        rho=np.array([654.9, 635.6]),
        T_sft=T,
        sft=gamma,
    )


def test_cost_fn_length_includes_sft():
    from fit_pcsaft._fit_utils import _make_cost_fn
    data = _hexane_data_with_sft()
    cost = _make_cost_fn(data, HEXANE, HEXANE_SPEC, Units(), _sft_config())
    r = cost(np.sqrt(HEXANE_P))
    assert r.size == 2 + 2 + 0 + 3


def test_sft_residuals_vanish_on_self_consistent_data():
    """Feeding back the model's own gamma must zero the sft block."""
    from fit_pcsaft._fit_utils import _make_cost_fn
    data = _hexane_data_with_sft()
    cost = _make_cost_fn(data, HEXANE, HEXANE_SPEC, Units(), _sft_config())
    r = cost(np.sqrt(HEXANE_P))
    assert np.allclose(r[-3:], 0.0, atol=1e-9)


def test_analytical_jacobian_refuses_sft():
    from fit_pcsaft._pure.jacobian import _make_f_and_df
    data = _hexane_data_with_sft()
    with pytest.raises(ValueError, match="surface tension"):
        _make_f_and_df(data, HEXANE, HEXANE_SPEC, Units(), _sft_config())
