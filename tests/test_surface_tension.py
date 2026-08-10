import os
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


def water_reference_sft():
    """Reference gamma(T) for water: the committed IAPWS R1-76 quasi-data.

    Regenerate with ``examples/data/generate_water_reference.py``.
    """
    d = Path(__file__).parent.parent / "examples" / "data" / "surface_tension"
    return load_sft_csv(d / "water.csv")


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
    T, exp = water_reference_sft()
    model = predict_surface_tension(func, T, Units())
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


def test_metrics_include_sft_aad():
    from fit_pcsaft.result import _compute_pure_metrics

    data = _hexane_data_with_sft()
    eos = _build_eos(HEXANE_P, HEXANE, HEXANE_SPEC)
    func = _build_functional(HEXANE_P, HEXANE, HEXANE_SPEC)
    metrics = _compute_pure_metrics(eos, data, Units(), functional=func)
    assert "sft" in metrics
    # self-consistent data -> zero deviation
    assert metrics["sft"].mae == pytest.approx(0.0, abs=1e-9)
    assert metrics["sft"].n == 3


def test_metrics_sft_empty_without_functional():
    from fit_pcsaft.result import _compute_pure_metrics
    data = _hexane_data_with_sft()
    eos = _build_eos(HEXANE_P, HEXANE, HEXANE_SPEC)
    metrics = _compute_pure_metrics(eos, data, Units())
    assert metrics["sft"].n == 0


def test_residuals_df_carries_sft_rows():
    from fit_pcsaft.result import _compute_per_point_rd

    data = _hexane_data_with_sft()
    eos = _build_eos(HEXANE_P, HEXANE, HEXANE_SPEC)
    func = _build_functional(HEXANE_P, HEXANE, HEXANE_SPEC)
    df = _compute_per_point_rd(eos, data, Units(), functional=func)
    assert "sft" in set(df["property"].to_list())


def test_public_entry_points_accept_sft():
    import inspect

    from fit_pcsaft import eval_pure, fit_pure, fit_pure_de

    for fn in (fit_pure, fit_pure_de, eval_pure):
        params = inspect.signature(fn).parameters
        assert "sft_path" in params, f"{fn.__name__} missing sft_path"
        assert "surface_tension_unit" in params, f"{fn.__name__} missing surface_tension_unit"
    for fn in (fit_pure, fit_pure_de):
        assert "sft_weight" in inspect.signature(fn).parameters


@pytest.mark.skipif(
    bool(os.environ.get("NO_NETWORK")), reason="_fetch_compound hits PubChem"
)
def test_setup_pure_fit_picks_numerical_jacobian_for_sft(tmp_path: Path, capsys):
    """_setup_pure_fit must not hand back the analytical Jacobian when sft is present."""
    from fit_pcsaft._pure.fit import _setup_pure_fit

    psat = tmp_path / "psat.csv"
    psat.write_text("T,psat\n300.0,21.9\n320.0,43.9\n340.0,79.5\n")
    rho = tmp_path / "rho.csv"
    rho.write_text("T,rho\n300.0,654.9\n320.0,635.6\n340.0,615.4\n")
    sft = tmp_path / "sft.csv"
    sft.write_text("T,sft\n300.0,17.60\n320.0,15.34\n")

    out = _setup_pure_fit(
        id="hexane",
        psat_path=psat, density_path=rho, hvap_path=None, sft_path=sft,
        mu=0.0, q=0.0, na=None, nb=None,
        psat_weight=3.0, density_weight=2.0, hvap_weight=1.0, sft_weight=1.0,
        extrapolate_psat=False,
        pressure_unit=si.KILO * si.PASCAL,
        temperature_unit=si.KELVIN,
        density_unit=si.KILOGRAM / (si.METER**3),
        enthalpy_unit=si.KILO * si.JOULE / si.MOL,
        surface_tension_unit=si.MILLI * si.NEWTON / si.METER,
    )
    data, cost_fn = out[0], out[5]
    assert data.T_sft.size == 2
    assert cost_fn(np.sqrt(HEXANE_P)).size == 3 + 3 + 2
    assert "surface tension" in capsys.readouterr().out.lower()
