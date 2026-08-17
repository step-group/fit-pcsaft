"""Configuration tests for the transfer-test problems.

These are cheap (no feos, no pymoo, no fitting) and exist because the expensive
failures in this area were all configuration: an overlay figure named after the
wrong compound, and a 70-minute ladder that would have died on a missing CSV.
"""

import pytest


def test_camphor_and_carvone_configs_survive_the_refactor_unchanged():
    """benchmarks/{camphor,carvone,carvone-rho6}-transfer/ are committed artefacts
    produced by the pre-registry BASE dict. If the move to _problems.py changes one
    of these values, those artefacts stop describing the code that made them."""
    from benchmarks._problems import PROBLEMS

    assert PROBLEMS["camphor"].fit == {
        "id": "camphor", "mu": 2.9, "na": 0, "nb": 0,
        "objectives": ("psat", "rho"), "bounds": None,
    }
    assert PROBLEMS["carvone"].fit == {
        "id": "carvone", "mu": 2.8, "na": 0, "nb": 1,
        "objectives": ("psat", "rho"), "bounds": None,
    }
    assert PROBLEMS["carvone-rho6"].fit == PROBLEMS["carvone"].fit
    assert PROBLEMS["carvone-rho6"].density_stem == "carvone_rho6"
    assert PROBLEMS["carvone-rho6"].psat_stem == "carvone"
    for name in ("camphor", "carvone", "carvone-rho6"):
        assert PROBLEMS[name].axis_labels == ("AARD_psat [%]", "AARD_rho [%]")
        assert PROBLEMS[name].rescore_cap == (10.0, 10.0)


def test_thymol_matches_the_production_fit_in_tesis():
    """Copied from TESIS analysis/scripts/pure/s04_fit_thymol.py. The point of the
    benchmark is that it measures TESIS's problem and not a convenient neighbour of
    it, so every one of these is checked rather than eyeballed."""
    from benchmarks._problems import PROBLEMS

    p = PROBLEMS["thymol"]
    assert p.fit["id"] == "thymol"
    assert p.fit["na"] == 1 and p.fit["nb"] == 1          # 2B, one phenolic -OH
    assert p.fit["objectives"] == ("vle", "sft")
    assert p.fit["refs"] == (2.0, 3.0)                    # REF_VLE, REF_SFT
    assert p.fit["bounds"] == [
        (2.0, 4.5), (3.9, 4.9), (260.0, 400.0), (1.0e-4, 0.4), (2000.0, 3600.0),
    ]
    assert p.fit["sft_options"].critical_temperature_k == pytest.approx(698.3)
    assert p.axis_labels == ("AAD_vle [%]", "AAD_sft [mN/m]")
    assert p.rescore_cap == (5.0, None)                   # AAD_sft is not clipped
    assert p.select_refs == (2.0, 3.0)
    assert p.baseline_eq32 == pytest.approx(0.561)


def test_every_problem_points_at_data_that_exists():
    """A missing CSV otherwise surfaces ~70 minutes into a ladder."""
    from benchmarks._problems import PROBLEMS

    for name, p in PROBLEMS.items():
        for key, path in p.paths().items():
            assert path.exists(), f"{name}: {key} -> {path} does not exist"


def test_an_sft_objective_implies_an_sft_path():
    """objectives=('vle','sft') without sft_path raises deep inside the fit, after
    the worker pool has spawned. Catch it as configuration instead."""
    from benchmarks._problems import PROBLEMS

    for name, p in PROBLEMS.items():
        if "sft" in p.fit["objectives"]:
            assert p.sft_stem is not None, f"{name} has an sft objective but no sft_stem"
            assert "sft_path" in p.paths()


def test_no_unpublished_thesis_data_reaches_examples():
    """fit-pcsaft is still the public step-group/fit-pcsaft. TESIS's processed CSVs
    carry a `source` column, and rows reading "This work (2026)" are unpublished
    thesis measurements -- carvone's five were dropped before copying. This makes the
    guard mechanical instead of remembered."""
    from benchmarks._problems import DATA

    offenders = sorted(
        p.relative_to(DATA).as_posix()
        for p in DATA.rglob("*.csv")
        if "This work" in p.read_text()
    )
    assert offenders == []
