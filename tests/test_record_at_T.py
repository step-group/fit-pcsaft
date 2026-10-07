"""record_at_T: an opt-in hook that rebuilds a pure record per temperature.

feos holds sigma constant; a water model such as Cameretti & Sadowski (2008)
has sigma(T). The hook is applied at every binary EOS build that knows its
temperature, and stored on the result so re-predictions use it too.
"""

import json
from pathlib import Path

import feos
import pytest
import si_units as si
from fit_pcsaft import fit_kij_henry, fit_kij_lle, fit_kij_sle, fit_kij_vle, fit_kij_vlle
from fit_pcsaft._binary._utils import _build_binary_eos

DATA = Path(__file__).parent.parent / "examples" / "data"

WATER_2B = {
    "identifier": {"name": "water"},
    "molarweight": 18.015,
    "m": 1.0656,
    "sigma": 3.0007,
    "epsilon_k": 366.51,
    "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.034868, "epsilon_k_ab": 2500.7}],
}
KETONE = {
    "identifier": {"name": "ketone"},
    "molarweight": 58.08,
    "m": 2.7447,
    "sigma": 3.2742,
    "epsilon_k": 232.99,
    "association_sites": [{"na": 0.0, "nb": 1.0}],
}
LLE_T = (298.15, 313.15)

# Recorded on fit-pcsaft before record_at_T existed.
LLE_KIJ = -0.050613523052819666
LLE_MODEL = [0.02064465813539141, 0.9680090804837835, 0.013829906079903379, 0.9259148574242361]
SLE_KIJ = 0.0011629679502175385
SLE_ARD = 1.012643006088236


def _identity(record, T):
    return record


def _water_sigma_grows(record, T):
    d = record.to_dict()
    if d["identifier"]["name"] != "water":
        return record
    d["sigma"] += 0.002 * (T - 298.15)
    return feos.PureRecord.from_json_str(json.dumps(d))


@pytest.fixture
def lle_inputs(tmp_path):
    params = tmp_path / "pure.json"
    params.write_text(json.dumps([KETONE, WATER_2B]))
    csv = tmp_path / "lle.csv"
    csv.write_text("temperature_K,x1_I,x1_II\n298.15,0.02,0.75\n313.15,0.025,0.72\n")
    return csv, params


def _lle(lle_inputs, **kw):
    csv, params = lle_inputs
    return fit_kij_lle(
        "ketone", "water", csv, params,
        kij_order=0, log_residuals=True, require_both_phases=False, **kw,
    )


def _sle(**kw):
    return fit_kij_sle(
        "tetrachloromethane", "2-undecanone",
        DATA / "sle" / "ccl4_2-undecanone.csv",
        DATA / "parameters" / "binary_params.json",
        tm=250.77 * si.KELVIN, delta_hfus=3.273 * si.KILO * si.JOULE / si.MOL, solid_index=0,
        tm2=285.84 * si.KELVIN, delta_hfus2=34.544 * si.KILO * si.JOULE / si.MOL,
        kij_order=0, kij_t_ref=298.0, kij_bounds=(-0.2, 0.2), **kw,
    )


def test_default_reproduces_the_numbers_from_before_the_hook(lle_inputs):
    lle = _lle(lle_inputs)
    assert lle._record_at_T is None
    assert lle.kij_coeffs[0] == pytest.approx(LLE_KIJ, rel=1e-6)
    assert lle.residuals()["model"].to_list() == pytest.approx(LLE_MODEL, rel=1e-6)
    sle = _sle()
    assert sle.kij_coeffs[0] == pytest.approx(SLE_KIJ, rel=1e-6)
    assert sle.ard == pytest.approx(SLE_ARD, rel=1e-6)


def test_identity_hook_changes_nothing(lle_inputs):
    for fit in (lambda **kw: _lle(lle_inputs, **kw), _sle):
        plain, hooked = fit(), fit(record_at_T=_identity)
        assert hooked._record_at_T is _identity
        assert list(hooked.kij_coeffs) == list(plain.kij_coeffs)
        assert hooked.residuals().equals(plain.residuals())


def test_a_temperature_dependent_sigma_moves_the_fit(lle_inputs):
    plain, hooked = _lle(lle_inputs), _lle(lle_inputs, record_at_T=_water_sigma_grows)
    assert not hooked.residuals().equals(plain.residuals())
    assert hooked.kij_coeffs[0] != plain.kij_coeffs[0]


def test_the_hook_without_a_temperature_is_refused():
    water = feos.PureRecord.from_json_str(json.dumps(WATER_2B))
    ketone = feos.PureRecord.from_json_str(json.dumps(KETONE))
    with pytest.raises(ValueError, match="T_K"):
        _build_binary_eos(ketone, water, 0.0, record_at_T=_identity)


def test_residuals_rebuild_the_records_at_each_data_temperature(lle_inputs):
    seen = []

    def spy(record, T):
        seen.append(T)
        return _water_sigma_grows(record, T)

    result = _lle(lle_inputs, record_at_T=spy)
    seen.clear()
    resid = result.residuals()
    # Both records at each of the two rows, nothing at any other temperature.
    assert sorted(seen) == sorted(2 * LLE_T)
    # And what it scored is the hooked model, built by hand at each T.
    by_hand = []
    for T in LLE_T:
        water = _water_sigma_grows(result._record2, T)
        eos = _build_binary_eos(result._record1, water, result.kij_at(T))
        pe = feos.State(
            eos, T * si.KELVIN, pressure=result.lle_pressure_bar * si.BAR,
            composition=[0.4, 0.6], density_initialization="liquid",
        ).tp_flash(max_iter=1000)
        by_hand.append(sorted(float(s.molefracs[0]) for s in (pe.liquid, pe.vapor)))
    model = resid["model"].to_list()
    assert model == pytest.approx([x for pair in by_hand for x in pair], rel=1e-6)


@pytest.mark.parametrize("fit", [fit_kij_vle, fit_kij_vlle, fit_kij_henry])
def test_the_other_fitters_refuse_the_hook(fit):
    with pytest.raises(NotImplementedError, match="record_at_T"):
        fit("a", "b", "missing.csv", "missing.json", record_at_T=_identity)
