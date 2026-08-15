"""Tests for src/fit_pcsaft/_fit_utils.py -- currently just _fetch_compound's cache.

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
