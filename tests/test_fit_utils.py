"""Tests for src/fit_pcsaft/_fit_utils.py: _fetch_compound's cache, the parameter layout.

No network here: pcp.get_compounds/get_synonyms are monkeypatched, unlike the
skipif(NO_NETWORK) live-PubChem tests elsewhere (e.g. test_surface_tension.py).
"""

import pytest

from fit_pcsaft import _fit_utils


class _FakeCompound:
    cid = 962
    iupac_name = "oxidane"
    preferred_iupac_name = None
    smiles = "O"
    inchi = "InChI=1S/H2O/h1H2"
    molecular_formula = "H2O"
    molecular_weight = 18.015


@pytest.fixture(autouse=True)
def _clear_fetch_compound_cache():
    """_fetch_compound is lru_cache'd at module scope -- an unmocked call in
    another test module (real PubChem lookups keyed on the same id string,
    e.g. "water") must never see a fake compound left behind by this file,
    and vice versa. Clear on both sides regardless of test order.
    """
    _fit_utils._fetch_compound.cache_clear()
    yield
    _fit_utils._fetch_compound.cache_clear()


def test_fetch_compound_caches_one_lookup_per_id(monkeypatch):
    """Two calls for the same id hit PubChem once; a different id hits again."""
    calls = []

    def fake_get_compounds(id_str, namespace):
        calls.append((id_str, namespace))
        return [_FakeCompound()] if namespace == "name" else []

    def fake_get_synonyms(cid, namespace):
        return [{"Synonym": ["7732-18-5"]}]

    monkeypatch.setattr(_fit_utils.pcp, "get_compounds", fake_get_compounds)
    monkeypatch.setattr(_fit_utils.pcp, "get_synonyms", fake_get_synonyms)

    ident1, mw1 = _fit_utils._fetch_compound("water")
    ident2, mw2 = _fit_utils._fetch_compound("water")

    assert calls == [("water", "name")], "second call must be a cache hit, not a new lookup"
    assert ident1 is ident2
    assert mw1 == mw2 == 18.015

    _fit_utils._fetch_compound("ethanol")
    assert calls == [("water", "name"), ("ethanol", "name")], "a different id must still look up"


@pytest.fixture
def _instant_backoff(monkeypatch):
    """Same number of attempts, no wall-clock wait."""
    monkeypatch.setattr(_fit_utils, "_RETRY_BACKOFF", (0.0, 0.0))


def test_fetch_compound_error_message_includes_last_exception(monkeypatch, _instant_backoff):
    """A total lookup failure names the swallowed exception, not just 'not found'."""

    def raises(id_str, namespace):
        raise RuntimeError(f"pubchem down ({namespace})")

    monkeypatch.setattr(_fit_utils.pcp, "get_compounds", raises)

    with pytest.raises(ValueError, match="pubchem down"):
        _fit_utils._fetch_compound("nonexistent-compound-xyz")


def test_fetch_compound_retries_an_erroring_lookup(monkeypatch, _instant_backoff):
    """A flap on the first pass is retried; the compound is found on the third."""
    calls = []

    def flaky(id_str, namespace):
        calls.append(namespace)
        if len(calls) < 7:  # two full passes of three namespaces fail
            raise RuntimeError("HTTP 000")
        return [_FakeCompound()] if namespace == "name" else []

    monkeypatch.setattr(_fit_utils.pcp, "get_compounds", flaky)
    monkeypatch.setattr(
        _fit_utils.pcp, "get_synonyms", lambda cid, ns: [{"Synonym": ["7732-18-5"]}]
    )

    ident, _ = _fit_utils._fetch_compound("water")
    assert ident.cas == "7732-18-5"
    assert calls == ["name", "smiles", "inchi"] * 2 + ["name"]


def test_fetch_compound_does_not_retry_a_genuine_miss(monkeypatch, _instant_backoff):
    """No exception, just nothing found -> fail fast, do not sleep through 3 passes."""
    calls = []

    def empty(id_str, namespace):
        calls.append(namespace)
        return []

    monkeypatch.setattr(_fit_utils.pcp, "get_compounds", empty)

    with pytest.raises(ValueError, match="not found in PubChem"):
        _fit_utils._fetch_compound("nonexistent-compound-xyz")
    assert calls == ["name", "smiles", "inchi"]


# ---------------------------------------------------------------------------
# Parameter layout: one rule, every consumer
# ---------------------------------------------------------------------------

import feos
import numpy as np

from fit_pcsaft._fit_utils import (
    _build_eos,
    ad_model,
    ad_rows,
    param_names,
    params_dict,
)
from fit_pcsaft._types import Compound, ModelSpec

_ETHANOL = Compound(identifier=feos.Identifier(cas="64-17-5", name="ethanol"), mw=46.07)

# (spec, optimiser vector, feos AD row) -- the four layouts jacobian.py's old
# _setup_nonassoc / _setup_nonassoc_mu / _setup_assoc / _setup_assoc_mu produced.
_LAYOUTS = [
    (ModelSpec(mu=0.0), [2.38, 3.18, 198.2], [2.38, 3.18, 198.2, 0.0]),
    (ModelSpec(mu=None), [2.38, 3.18, 198.2, 1.7], [2.38, 3.18, 198.2, 1.7]),
    (
        ModelSpec(mu=0.0, na=1, nb=1),
        [2.38, 3.18, 198.2, 0.032, 2653.4],
        [2.38, 3.18, 198.2, 0.0, 0.032, 2653.4, 1.0, 1.0],
    ),
    (
        ModelSpec(mu=None, na=1, nb=2),
        [2.38, 3.18, 198.2, 1.7, 0.032, 2653.4],
        [2.38, 3.18, 198.2, 1.7, 0.032, 2653.4, 1.0, 2.0],
    ),
]


@pytest.mark.parametrize("spec,vec,row", _LAYOUTS)
def test_ad_rows_reproduce_the_four_old_make_row_layouts(spec, vec, row):
    assert ad_rows(np.array(vec), spec).tolist() == [row]
    assert ad_rows(np.array([vec, vec]), spec).shape == (2, len(row))
    assert len(param_names(spec)) == len(vec)
    expected_model = (
        feos.EquationOfStateAD.PcSaftFull if spec.na else feos.EquationOfStateAD.PcSaftNonAssoc
    )
    assert ad_model(spec) is expected_model


@pytest.mark.parametrize("spec,vec,row", _LAYOUTS)
def test_build_record_and_ad_rows_read_the_vector_the_same_way(spec, vec, row):
    """The EoS built from an optimiser vector and feos's AD model fed ad_rows of the
    same vector must give the same vapour pressure: one layout, two consumers."""
    T = np.array([[350.0]])
    p_eos, ok = feos.Property.vapor_pressure(_build_eos(np.array(vec), _ETHANOL, spec), T)
    p_ad, _, ok_ad = feos.Property.vapor_pressure_derivatives(
        ad_model(spec), [], ad_rows(np.array(vec), spec), T
    )
    assert bool(ok[0]) and bool(ok_ad[0])
    assert float(p_ad[0]) == pytest.approx(float(p_eos[0]), rel=1e-10)


def test_params_dict_is_the_old_extract_params_dict():
    spec = ModelSpec(mu=None, na=1, nb=1)
    assert params_dict([1, 2, 3, 4, 5, 6], spec) == {
        "m": 1.0, "sigma": 2.0, "epsilon_k": 3.0, "mu": 4.0, "kappa_ab": 5.0, "epsilon_k_ab": 6.0,
    }
    assert list(params_dict([1, 2, 3], ModelSpec(mu=0.0))) == ["m", "sigma", "epsilon_k"]
    with pytest.raises(ValueError):  # a vector of the wrong length is a bug, not a truncation
        params_dict([1, 2, 3], spec)


# ---------------------------------------------------------------------------
# predict_bulk: one vectorised feos call per property
# ---------------------------------------------------------------------------


def test_predict_bulk_matches_the_per_point_solve():
    """Same pure_t solver, one rayon-parallel call per property: point-for-point equal to
    the PhaseEquilibrium loop it replaces, NaN where that loop got an exception."""
    import si_units as si

    from fit_pcsaft._fit_utils import predict_bulk
    from fit_pcsaft._types import PureData, Units
    from tests.test_surface_tension import (
        HEXANE,
        HEXANE_P,
        HEXANE_SPEC,
        WATER,
        WATER_2B,
        WATER_2B_SPEC,
    )

    data = PureData(
        T_psat=np.array([300.0, 320.0, 2000.0]), p_psat=np.zeros(3),
        T_rho=np.array([300.0, 320.0, 2000.0]), rho=np.zeros(3),
        T_hvap=np.array([300.0, 2000.0]), hvap=np.zeros(2),
    )
    units = Units()
    for vec, comp, spec in ((HEXANE_P, HEXANE, HEXANE_SPEC), (WATER_2B, WATER, WATER_2B_SPEC)):
        eos = _build_eos(vec, comp, spec)

        def ref(T, f):
            out = np.full(len(T), np.nan)
            for i, t in enumerate(T):
                try:
                    out[i] = float(f(float(t) * si.KELVIN))
                except Exception:
                    pass
            return out

        def hvap(T):
            vle = feos.PhaseEquilibrium.pure(eos, T)
            return (
                vle.vapor.molar_enthalpy(feos.Contributions.Residual)
                - vle.liquid.molar_enthalpy(feos.Contributions.Residual)
            ) / units.enthalpy

        want = {
            "psat": ref(data.T_psat, lambda T: feos.PhaseEquilibrium.vapor_pressure(eos, T)[0] / units.pressure),
            "rho": ref(data.T_rho, lambda T: feos.PhaseEquilibrium.pure(eos, T).liquid.mass_density() / units.density),
            "hvap": ref(data.T_hvap, hvap),
        }
        got = predict_bulk(eos, comp.mw, data, units)
        for k in want:
            np.testing.assert_allclose(got[k], want[k], rtol=1e-12, equal_nan=True)
        assert np.isnan(got["psat"][-1]) and np.isnan(got["rho"][-1]) and np.isnan(got["hvap"][-1])
        assert np.isfinite(got["psat"][:2]).all() and np.isfinite(got["rho"][:2]).all()

    empty = PureData(T_psat=np.array([]), p_psat=np.array([]), T_rho=np.array([]), rho=np.array([]))
    assert all(v.size == 0 for v in predict_bulk(eos, comp.mw, empty, units).values())


# ---------------------------------------------------------------------------
# Liquid heat capacity: loader, ideal-gas cp, predict_bulk, reporting
# ---------------------------------------------------------------------------

# DIPPR 107 for water, in DIPPR units J/(kmol K) -- reproduces the IAPWS-95
# ideal-gas cp to 0.03% at 300 K.
WATER_DIPPR107 = [33363.0, 26790.0, 2610.5, 8896.0, 1169.0]


def _aly_lee(T, c):
    A, B, C, D, E = c
    T = np.asarray(T, dtype=float)
    return (A + B * ((C / T) / np.sinh(C / T)) ** 2 + D * ((E / T) / np.cosh(E / T)) ** 2) / 1000.0


def test_load_cp_csv_pressure_column_is_optional(tmp_path):
    """No P column -> all-NaN P of the right length; a blank P cell -> NaN in that
    row only, and load_csv must not raise (the NaN check covers required columns)."""
    from fit_pcsaft._csv import load_cp_csv

    p = tmp_path / "a.csv"
    p.write_text("T,cp\n300.0,75.3\n350.0,75.9\n")
    T, P, cp = load_cp_csv(p)
    assert T.tolist() == [300.0, 350.0] and cp.tolist() == [75.3, 75.9]
    assert P.shape == (2,) and np.isnan(P).all()

    p.write_text("T,P,cp\n300.0,101.325,75.3\n350.0,,75.9\n")
    T, P, cp = load_cp_csv(p)
    assert P[0] == 101.325 and np.isnan(P[1])


@pytest.mark.parametrize("header", ["heat_capacity", "c_p", "Cp", "cp_J_mol_K"])
def test_load_cp_csv_aliases(tmp_path, header):
    from fit_pcsaft._csv import load_cp_csv

    p = tmp_path / "a.csv"
    p.write_text(f"T,{header}\n300.0,75.3\n")
    _, _, cp = load_cp_csv(p)
    assert cp.tolist() == [75.3]


def test_puredata_and_units_carry_cp():
    from fit_pcsaft._types import PureData, Units

    d = PureData(T_psat=np.array([300.0]), p_psat=np.array([3.5]),
                 T_rho=np.array([300.0]), rho=np.array([996.0]))
    assert d.T_cp.size == d.P_cp.size == d.cp.size == d.cp_ig.size == 0
    import si_units as si
    assert Units().heat_capacity / (si.JOULE / (si.MOL * si.KELVIN)) == 1.0


def test_ideal_gas_cp_matches_aly_lee():
    """feos's DIPPR-107 model, evaluated through an ideal-gas-only EoS, is the
    closed-form Aly-Lee equation. feos takes the coefficients in DIPPR units."""
    from fit_pcsaft._fit_utils import ideal_gas_cp
    from tests.test_surface_tension import WATER

    T = np.array([300.0, 400.0, 500.0])
    got = ideal_gas_cp(WATER, WATER_DIPPR107, T)
    np.testing.assert_allclose(got, _aly_lee(T, WATER_DIPPR107), rtol=1e-9)
    assert got[0] == pytest.approx(33.5859, abs=1e-3)


def test_ideal_gas_cp_rejects_a_bad_coefficient_list():
    from fit_pcsaft._fit_utils import ideal_gas_cp
    from tests.test_surface_tension import WATER

    with pytest.raises(ValueError, match="cp_ig"):
        ideal_gas_cp(WATER, [1.0, 2.0], np.array([300.0]))


def test_predict_bulk_cp_is_the_residual_plus_the_tabulated_ideal_gas():
    """NaN pressure -> saturated liquid; explicit pressure -> the liquid root at
    (T, P); a temperature where psat does not converge -> NaN. cp_ig is whatever
    PureData carries -- predict_bulk only adds it."""
    import si_units as si

    from fit_pcsaft._fit_utils import predict_bulk
    from fit_pcsaft._types import PureData, Units
    from tests.test_surface_tension import (
        HEXANE,
        HEXANE_P,
        HEXANE_SPEC,
        WATER,
        WATER_2B,
        WATER_2B_SPEC,
    )

    units = Units()
    J_molK = si.JOULE / (si.MOL * si.KELVIN)
    T = np.array([300.0, 320.0, 2000.0])
    P = np.array([np.nan, 500.0, np.nan])          # kPa; NaN = saturation
    cp_ig = np.array([30.0, 31.0, 32.0])
    data = PureData(
        T_psat=np.array([]), p_psat=np.array([]), T_rho=np.array([]), rho=np.array([]),
        T_cp=T, P_cp=P, cp=np.zeros(3), cp_ig=cp_ig,
    )
    for vec, comp, spec in ((HEXANE_P, HEXANE, HEXANE_SPEC), (WATER_2B, WATER, WATER_2B_SPEC)):
        eos = _build_eos(vec, comp, spec)
        sat = feos.PhaseEquilibrium.pure(eos, 300.0 * si.KELVIN).liquid
        at_p = feos.State(eos, temperature=320.0 * si.KELVIN, pressure=500.0 * si.KILO * si.PASCAL,
                          density_initialization="liquid")
        want = np.array([
            sat.molar_isobaric_heat_capacity(feos.Contributions.Residual) / J_molK + 30.0,
            at_p.molar_isobaric_heat_capacity(feos.Contributions.Residual) / J_molK + 31.0,
            np.nan,
        ])
        got = predict_bulk(eos, comp.mw, data, units)["cp"]
        np.testing.assert_allclose(got, want, rtol=1e-10, equal_nan=True)

    empty = PureData(T_psat=np.array([]), p_psat=np.array([]), T_rho=np.array([]), rho=np.array([]))
    assert predict_bulk(eos, comp.mw, empty, units)["cp"].size == 0


def test_fit_result_reports_cp_only_when_cp_data_is_present():
    from types import SimpleNamespace

    from fit_pcsaft._types import PureData, Units
    from fit_pcsaft.result import FitResult, _compute_pure_metrics
    from tests.test_surface_tension import HEXANE, HEXANE_P, HEXANE_SPEC

    eos = _build_eos(HEXANE_P, HEXANE, HEXANE_SPEC)
    bulk = PureData(T_psat=np.array([300.0]), p_psat=np.array([21.7]),
                    T_rho=np.array([300.0]), rho=np.array([655.0]))
    with_cp = PureData(T_psat=bulk.T_psat, p_psat=bulk.p_psat, T_rho=bulk.T_rho, rho=bulk.rho,
                       T_cp=np.array([300.0]), P_cp=np.array([np.nan]),
                       cp=np.array([195.0]), cp_ig=np.array([143.0]))

    def result(data):
        metrics = _compute_pure_metrics(eos, HEXANE.mw, data, Units())
        return FitResult(
            params={"m": 3.0576, "sigma": 3.7983, "epsilon_k": 236.77}, eos=eos, data=data,
            compound=HEXANE, spec=HEXANE_SPEC, units=Units(), metrics=metrics,
            scipy_result=SimpleNamespace(cost=0.0, fun=np.zeros(1), success=True, nfev=1),
            time_elapsed=0.0,
        )

    without = result(bulk)
    assert "Heat capacity" not in str(without)
    assert without.metrics["cp"].n_total == 0

    with_ = result(with_cp)
    assert "Heat capacity" in str(with_)
    assert np.isfinite(with_.ard_cp)
    table = with_.metrics_table()
    assert table.filter(table["property"] == "cp")["n"].item() == 1


def test_water_cp_quasi_data_is_saturated_liquid_total_cp():
    """examples/data/heat_capacity/water.csv: IAPWS-95 total cp of the saturated
    liquid on the same 23-point 280-620 K grid as psat/rho, no P column (the data
    *is* at saturation, so the file exercises the NaN-P path). Regenerate with
    examples/data/generate_water_reference.py."""
    from pathlib import Path

    from fit_pcsaft._csv import load_cp_csv

    d = Path(__file__).parent.parent / "examples" / "data"
    T, P, cp = load_cp_csv(d / "heat_capacity" / "water.csv")
    T_psat, _ = np.loadtxt(d / "psat" / "water.csv", delimiter=",", skiprows=1, unpack=True)
    np.testing.assert_allclose(T, T_psat)
    assert np.isnan(P).all()
    assert cp[0] == pytest.approx(75.69, abs=0.05)    # IAPWS-95, 280 K
    assert cp.argmin() == 2                            # liquid water's cp minimum, ~310 K
    assert 2.0 * cp[0] < cp[-1] < 200.0                # climbing toward the critical point
