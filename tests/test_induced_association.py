"""The induced-association rule: which sites the solvating component gets.

Rehner, Bardow & Gross, Int. J. Thermophys. 44, 179 (2023): a ketone carries one
acceptor site (na = 0, nb = 1), kappa is copied from the self-associating partner,
and the site energy is the fitted cross parameter through feos's arithmetic-mean
combining rule. The 2B override is Kleiner & Sadowski (2007)'s "one acceptor and
one donor site for the polar component", chosen there for simplicity; it stays the
default.
"""

import inspect
import json
import warnings

import feos
import pytest
import si_units as si
from fit_pcsaft import (
    fit_kij_henry,
    fit_kij_lle,
    fit_kij_vle,
    fit_kij_vle_lle,
    fit_kij_vlle,
)
from fit_pcsaft._binary._utils import _apply_induced_association, _build_binary_eos
from fit_pcsaft._binary.fitter import BinaryKijFitter

WATER_2B = {
    "identifier": {"name": "water"},
    "molarweight": 18.015,
    "m": 1.0656,
    "sigma": 3.0007,
    "epsilon_k": 366.51,
    "association_sites": [
        {"na": 1.0, "nb": 1.0, "kappa_ab": 0.034868, "epsilon_k_ab": 2500.7}
    ],
}
KETONE = {  # acetone-like, one acceptor, no association parameters
    "identifier": {"name": "ketone"},
    "molarweight": 58.08,
    "m": 2.7447,
    "sigma": 3.2742,
    "epsilon_k": 232.99,
    "association_sites": [{"na": 0.0, "nb": 1.0}],
}
BARE = {  # no sites declared at all
    "identifier": {"name": "bare"},
    "molarweight": 58.08,
    "m": 2.7447,
    "sigma": 3.2742,
    "epsilon_k": 232.99,
}


def _rec(d):
    return feos.PureRecord.from_json_str(json.dumps(d))


def _site(record):
    return record.to_dict()["association_sites"][0]


def test_default_is_the_2b_override():
    ketone, water = _apply_induced_association(_rec(KETONE), _rec(WATER_2B))
    site = _site(ketone)
    assert site["na"] == 1.0 and site["nb"] == 1.0
    assert site["kappa_ab"] == pytest.approx(0.034868)
    assert site.get("epsilon_k_ab", 0.0) == 0.0
    assert _site(water)["epsilon_k_ab"] == pytest.approx(2500.7)


def test_own_keeps_the_declared_acceptor_only_site():
    ketone, _ = _apply_induced_association(_rec(KETONE), _rec(WATER_2B), sites="own")
    site = _site(ketone)
    assert site.get("na", 0.0) == 0.0 and site["nb"] == 1.0
    assert site["kappa_ab"] == pytest.approx(0.034868)


def test_own_writes_the_requested_epsilon():
    ketone, _ = _apply_induced_association(
        _rec(KETONE), _rec(WATER_2B), sites="own", epsilon_k_ab=859.1
    )
    assert _site(ketone)["epsilon_k_ab"] == pytest.approx(859.1)


def test_epsilon_on_the_one_site_record_is_a_live_knob():
    """The cross energy is (eps_ketone + eps_water)/2, so raising the ketone's
    site energy must lower ln phi of the ketone in the mixture."""
    T, p, z = 298.15 * si.KELVIN, 1.01325 * si.BAR, [0.5, 0.5]

    def ln_phi(eps):
        r1, r2 = _apply_induced_association(
            _rec(KETONE), _rec(WATER_2B), sites="own", epsilon_k_ab=eps
        )
        state = feos.State(
            _build_binary_eos(r1, r2, 0.0),
            temperature=T,
            pressure=p,
            composition=z,
            density_initialization="liquid",
        )
        return float(state.ln_phi()[0])

    assert ln_phi(859.1) < ln_phi(0.0) - 0.02


def test_own_with_no_declared_sites_falls_back_to_2b_and_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bare, _ = _apply_induced_association(_rec(BARE), _rec(WATER_2B), sites="own")
    assert any("2B" in str(x.message) for x in w)
    site = _site(bare)
    assert site["na"] == 1.0 and site["nb"] == 1.0


def test_unknown_sites_value_is_refused():
    with pytest.raises(ValueError, match="sites"):
        _apply_induced_association(_rec(KETONE), _rec(WATER_2B), sites="3B")


def test_both_associating_is_still_a_no_op_with_a_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a, b = _apply_induced_association(_rec(WATER_2B), _rec(WATER_2B), sites="own")
    assert w and "no effect" in str(w[0].message)
    assert _site(a) == _site(_rec(WATER_2B))


@pytest.mark.parametrize(
    "fn", [fit_kij_lle, fit_kij_vle, fit_kij_henry, fit_kij_vlle, fit_kij_vle_lle]
)
def test_every_induced_fitter_forwards_sites_and_epsilon(fn):
    params = inspect.signature(fn).parameters
    assert params["induced_sites"].default == "2B"
    assert params["induced_epsilon_k_ab"].default == 0.0


def test_the_fluent_fitter_forwards_them_too():
    params = inspect.signature(BinaryKijFitter.__init__).parameters
    assert params["induced_sites"].default == "2B"
    assert params["induced_epsilon_k_ab"].default == 0.0


def test_lle_source_forwards_both_to_the_helper():
    import fit_pcsaft._binary.lle as lle

    text = inspect.getsource(lle)
    assert "sites=induced_sites, epsilon_k_ab=induced_epsilon_k_ab" in text
