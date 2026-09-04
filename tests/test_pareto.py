import os
from pathlib import Path

import numpy as np
import pytest

from fit_pcsaft._pure.pareto import (
    _BIG,
    _argmin_scalarized,
    aad_objectives,
    non_dominated,
)
from fit_pcsaft._types import Units
from tests.test_surface_tension import (
    HEXANE,
    HEXANE_P,
    HEXANE_SPEC,
    _hexane_data_with_sft,
)

FORTE = ("psat", "rho")


class _NoDft(BaseException):
    """Not an Exception on purpose: _evaluate_point's `except Exception` around
    _build_functional would swallow one and the test would silently pass."""


def test_non_dominated_keeps_only_the_front():
    F = np.array([
        [1.0, 5.0],   # on front
        [2.0, 2.0],   # on front
        [5.0, 1.0],   # on front
        [3.0, 3.0],   # dominated by (2,2)
        [2.0, 2.0],   # duplicate of a front point
    ])
    mask = non_dominated(F)
    assert mask[0] and mask[1] and mask[2]
    assert not mask[3]
    assert not mask[4]


def test_non_dominated_single_point():
    assert non_dominated(np.array([[1.0, 1.0]])).tolist() == [True]


def test_coverage_is_zero_between_a_front_and_itself():
    """No point on a real front dominates another, in either direction."""
    from fit_pcsaft._pure.pareto import coverage

    A = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0]])
    assert coverage(A, A) == 0.0


def test_coverage_is_one_when_every_point_is_beaten():
    from fit_pcsaft._pure.pareto import coverage

    A = np.array([[1.0, 1.0], [2.0, 0.5]])
    B = np.array([[3.0, 3.0], [5.0, 2.0], [2.0, 4.0]])
    assert coverage(A, B) == 1.0
    assert coverage(B, A) == 0.0


def test_coverage_is_asymmetric_on_crossing_fronts():
    """The metric only means something reported both ways round.

    This is the case the docstring's extent comparison could not see: two
    fronts of similar span where one sits behind the other over part of its
    length and ahead over the rest.
    """
    from fit_pcsaft._pure.pareto import coverage

    A = np.array([[1.0, 9.0], [2.0, 4.0], [9.0, 1.0]])
    B = np.array([[3.0, 5.0], [4.0, 4.5], [8.0, 2.0]])
    assert coverage(A, B) == pytest.approx(2 / 3)
    assert coverage(B, A) == 0.0


def test_objectives_zero_on_self_consistent_sft():
    data = _hexane_data_with_sft()
    aad_vle, aad_sft = aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, data, Units())
    assert aad_sft == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(aad_vle)


def test_objectives_penalize_impossible_parameters():
    data = _hexane_data_with_sft()
    bad = np.array([1.0e-6, 1.0e-6, 1.0e-6])
    assert aad_objectives(bad, HEXANE, HEXANE_SPEC, data, Units()) == (_BIG, _BIG)


def test_hvap_is_reported_but_never_optimized():
    """`fit_pure_pareto` accepts hvap_path, and the front must ignore it.

    Enthalpy of vaporization is loaded into `data` and reported in the metrics
    of whatever `.select()` returns, which makes it easy to assume it is being
    fitted. It is not: eq 30 is psat and rho only. Absurd hvap values must move
    neither objective by a single ulp.
    """
    import dataclasses

    base = _hexane_data_with_sft()
    with_hvap = dataclasses.replace(
        base,
        T_hvap=np.array([300.0, 320.0]),
        hvap=np.array([1.0e6, -5.0e6]),   # nonsense on purpose
    )
    assert aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, base, Units()) == (
        aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, with_hvap, Units())
    )


def test_forte_objectives_are_the_two_aards_and_average_to_aad_vle():
    """Forte et al. 2018 splits Rehner & Gross's eq-30 mean into its two terms.

    The arithmetic identity is the check: whatever ('psat','rho') reports must
    average back to what ('vle','sft') reports on axis 0, exactly. If it does
    not, one of the two branches is computing a different quantity.
    """
    from fit_pcsaft._types import ModelSpec

    data, water = _water_data()
    spec = ModelSpec(mu=0.0, na=1, nb=1, q=0.0)
    params = np.array(WATER_SCHEMES[0][1])          # 2B, Table 1

    aad_psat, aad_rho = aad_objectives(
        params, water, spec, data, Units(), objectives=FORTE
    )
    aad_vle, _ = aad_objectives(params, water, spec, data, Units())

    assert 0.5 * (aad_psat + aad_rho) == pytest.approx(aad_vle, rel=1e-12)
    # the two terms of the module docstring's own reconciliation argument
    assert aad_psat == pytest.approx(4.56, abs=0.05)
    assert aad_rho == pytest.approx(1.76, abs=0.05)


def test_forte_mode_never_builds_a_functional(monkeypatch):
    """The whole performance case for the mode: the DFT solve dominates the
    cost of an evaluation and is skipped entirely, because there is no
    interface solve at all.

    Both names are module-level imports in pareto.py, so patching them *there*
    is what _evaluate_point actually resolves.
    """
    from fit_pcsaft._pure import pareto

    def _boom(*args, **kwargs):
        raise _NoDft("the DFT path was entered")

    monkeypatch.setattr(pareto, "_build_functional", _boom)
    monkeypatch.setattr(pareto, "predict_surface_tension", _boom)

    data = _hexane_data_with_sft()      # sft data present, and irrelevant
    f = pareto.aad_objectives(
        HEXANE_P, HEXANE, HEXANE_SPEC, data, Units(), objectives=FORTE
    )
    assert all(np.isfinite(v) for v in f)
    assert f != (_BIG, _BIG)

    # negative control: without it, a typo in the patch target passes silently
    with pytest.raises(_NoDft):
        pareto.aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, data, Units())


def test_sft_is_reported_but_never_optimized_in_forte_mode():
    """The `hvap_path` bullet, one property over.

    Under ('psat','rho') surface tension is still loaded into `data` and still
    shows up in the metrics of whatever `.select()` returns, which makes it easy
    to assume it is being fitted. Absurd gamma must move neither objective by a
    single ulp -- and must move both under the default pair.
    """
    import dataclasses

    base = _hexane_data_with_sft()
    absurd = dataclasses.replace(base, sft=base.sft * 3.0 + 50.0)

    assert aad_objectives(
        HEXANE_P, HEXANE, HEXANE_SPEC, base, Units(), objectives=FORTE
    ) == aad_objectives(
        HEXANE_P, HEXANE, HEXANE_SPEC, absurd, Units(), objectives=FORTE
    )
    assert aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, base, Units()) != (
        aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, absurd, Units())
    )


def test_refs_default_per_mode():
    """(2, 0.7) for Rehner & Gross, (2, 2) for Forte -- and they pick different
    points on the same front, which is the only way to test a default that is
    otherwise just a table entry.
    """
    from fit_pcsaft._pure.pareto import _DEFAULT_REFS, _argmin_scalarized

    F = np.array([[1.0, 4.0], [4.0, 2.0]])
    assert _DEFAULT_REFS[("vle", "sft")] == (2.0, 0.7)
    assert _DEFAULT_REFS[("psat", "rho")] == (2.0, 2.0)
    assert _argmin_scalarized(F, _DEFAULT_REFS[("vle", "sft")]) == 1
    assert _argmin_scalarized(F, _DEFAULT_REFS[("psat", "rho")]) == 0


def test_sft_objectives_without_sft_data_is_rejected():
    """The bug this guard replaces: with no gamma, the old code returned a
    constant 0.0 second objective and the whole front degenerated onto one axis.

    Raised before _setup_pure_fit, so it never reaches the PubChem lookup --
    which is also why this test passes offline with paths that do not exist.
    """
    from fit_pcsaft import fit_pure_pareto

    with pytest.raises(ValueError, match="sft_path"):
        fit_pure_pareto(id="water", psat_path="nope.csv", density_path="nope.csv")


def test_an_unsupported_objective_pair_is_rejected():
    from fit_pcsaft import fit_pure_pareto

    for pair in [("psat", "sft"), ("vle", "rho"), ("sft", "vle"), ("vle", "hvap")]:
        with pytest.raises(ValueError, match="objectives"):
            fit_pure_pareto(
                id="water", psat_path="nope.csv", density_path="nope.csv",
                sft_path="nope.csv", objectives=pair,
            )


def test_objectives_cross_the_process_boundary():
    """_worker_evaluate reads `objectives` from _WORKER, so it has to be in
    _worker_init's initargs: a spawned worker re-imports the module and inherits
    nothing else. Two plain strings pickle; feos.Identifier does not, which is
    why nothing else can ride along.

    Under ("vle", "sft"), the pair that still runs through the pool: the bulk
    pair is batched in-process and never reaches a worker.
    """
    from fit_pcsaft._pure.pareto import _map_evaluate, _worker_pool

    data = _hexane_data_with_sft()
    units = Units()
    expected = aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, data, units)
    with _worker_pool(
        2, HEXANE, HEXANE_SPEC, data, units, None, False, ("vle", "sft")
    ) as pool:
        got = _map_evaluate(
            [HEXANE_P], pool, HEXANE, HEXANE_SPEC, data, units, None, ("vle", "sft")
        )
    assert got[0][:2] == pytest.approx(expected, abs=1e-9)


# Rehner & Gross 2020, water, PC-SAFT: parameters from Table 1, AADs from Table 2.
# (scheme, params, na, nb, paper AAD_vle %, paper AAD_DFT mN/m)
WATER_SCHEMES = [
    ("2B", [1.0000, 2.9375, 272.03, 0.044480, 3125.3], 1, 1, 2.14, 1.59),
    ("3B", [1.6330, 2.4570, 238.32, 0.037807, 2749.0], 2, 1, 3.06, 1.14),
    ("4C", [1.8668, 2.3950, 169.78, 0.133738, 1772.0], 2, 2, 1.84, 1.81),
]


def _water_data():
    import feos

    from fit_pcsaft._csv import load_density_csv, load_psat_csv, load_sft_csv
    from fit_pcsaft._types import Compound, PureData

    d = Path(__file__).parent.parent / "examples" / "data"
    T1, p1 = load_psat_csv(d / "psat" / "water.csv")
    T2, r2 = load_density_csv(d / "density" / "water.csv")
    T3, g3 = load_sft_csv(d / "surface_tension" / "water.csv")
    data = PureData(T_psat=T1, p_psat=p1, T_rho=T2, rho=r2, T_sft=T3, sft=g3)
    water = Compound(
        identifier=feos.Identifier(name="water", cas="7732-18-5"), mw=18.015
    )
    return data, water


@pytest.mark.parametrize(
    "scheme,params,na,nb,paper_vle,paper_sft", WATER_SCHEMES,
    ids=[c[0] for c in WATER_SCHEMES],
)
def test_water_forward_check_against_paper(scheme, params, na, nb, paper_vle, paper_sft):
    """Rehner & Gross Tables 1 and 2, forward-evaluated through our objectives.

    Surface tension reproduces the paper's AAD_DFT column closely (within
    0.05 mN/m for all three schemes). AAD_vle runs systematically higher: for
    water the paper also fitted liquid and supercritical densities over
    273-1073 K, whereas this data set is saturation-only over 280-620 K. The
    ranking across schemes (4C < 2B < 3B) is reproduced either way.

    The AAD_vle band is two-sided on purpose. Measured ratios to the paper are
    1.48 / 1.56 / 1.25 for 2B / 3B / 4C -- systematic, as a data-set difference
    should be. The upper edge is the half with teeth: implementing eq 30 as the
    *sum* of the two AARDs rather than their *mean* doubles every ratio to
    2.5-3.1, and a one-sided "is it small enough" assertion would not notice.
    """
    from fit_pcsaft._types import ModelSpec

    data, water = _water_data()
    spec = ModelSpec(mu=0.0, na=na, nb=nb, q=0.0)
    aad_vle, aad_sft = aad_objectives(np.array(params), water, spec, data, Units())
    print(
        f"\n{scheme}: AAD_vle={aad_vle:.2f}% (paper {paper_vle})  "
        f"AAD_sft={aad_sft:.2f} mN/m (paper {paper_sft})"
    )
    assert aad_sft == pytest.approx(paper_sft, abs=0.05)
    ratio = aad_vle / paper_vle
    assert 1.0 < ratio < 2.0, (
        f"AAD_vle/paper = {ratio:.2f}, outside the 1-2 band; above 2 usually "
        "means eq 30 has been turned into a sum of the two AARDs"
    )


def test_water_scheme_ranking_matches_paper():
    """4C gives the best bulk AAD and 3B the best surface tension, as in Table 2."""
    from fit_pcsaft._types import ModelSpec

    data, water = _water_data()
    res = {}
    for scheme, params, na, nb, _, _ in WATER_SCHEMES:
        res[scheme] = aad_objectives(
            np.array(params), water, ModelSpec(mu=0.0, na=na, nb=nb, q=0.0),
            data, Units(),
        )
    assert min(res, key=lambda k: res[k][0]) == "4C"   # best bulk
    assert min(res, key=lambda k: res[k][1]) == "3B"   # best surface tension


def test_violation_is_graded_not_flat():
    """Infeasible sets must be rankable, not collapsed onto one penalty value."""
    from fit_pcsaft._pure.pareto import _evaluate_point

    data = _hexane_data_with_sft()
    good = _evaluate_point(HEXANE_P, HEXANE, HEXANE_SPEC, data, Units())
    hopeless = _evaluate_point(
        np.array([1e-6, 1e-6, 1e-6]), HEXANE, HEXANE_SPEC, data, Units()
    )
    assert good[2] <= 0.0, "a good parameter set must be feasible"
    assert hopeless[2] > 0.0, "a hopeless set must violate"
    # bounded above by the worst possible violation, so it stays comparable
    assert hopeless[2] <= 1.0


def test_penalty_is_zero_for_feasible_points():
    """A feasible row is the plain scaling with nothing added to it."""
    from fit_pcsaft._pure.pareto import _penalize

    R = np.array([[4.0, 1.4, -0.1], [2.0, 0.7, 0.0]])
    out = _penalize(R, (2.0, 0.7))
    assert out == pytest.approx(np.array([[2.0, 2.0], [1.0, 1.0]]))


def test_penalty_is_graded_and_loses_to_every_feasible_point():
    """MOEA/D has no constraint handling, so the grading has to live in F.

    Same objectives, three feasibility levels: the ordering between them is the
    only signal the optimizer gets about how badly a set failed.
    """
    from fit_pcsaft._pure.pareto import _penalize

    R = np.array([
        [4.0, 1.4, -0.1],   # feasible
        [4.0, 1.4, 0.02],   # missed the threshold by one point in twenty-five
        [4.0, 1.4, 1.00],   # nothing evaluated at all
    ])
    out = _penalize(R, (2.0, 0.7))
    assert np.all(out[0] < out[1]), "feasible must beat infeasible"
    assert np.all(out[1] < out[2]), "violation must stay graded, not flat"


def test_objective_scale_uses_the_observed_spans():
    """Tchebicheff responds to spans, so the spans are what must be equalized."""
    from fit_pcsaft._pure.pareto import _objective_scale

    R = np.array([
        [1.0, 2.0, -0.1],
        [5.0, 2.5, -0.1],
        [9.0, 3.0, -0.1],
        [11.0, 3.5, -0.1],
    ])
    s_vle, s_sft = _objective_scale(R, (2.0, 0.7))
    # spans of the feasible rows, not the eq-32 references
    assert s_vle == pytest.approx(9.0, rel=0.2)
    assert s_sft == pytest.approx(1.35, rel=0.2)


def test_objective_scale_ignores_infeasible_and_degenerate_rows():
    from fit_pcsaft._pure.pareto import _objective_scale

    clean = np.array([[1.0, 2.0, -0.1], [3.0, 2.4, -0.1],
                      [5.0, 2.8, -0.1], [7.0, 3.2, -0.1]])
    polluted = np.vstack([clean, np.array([
        [500.0, 90.0, 0.4],        # infeasible
        [_BIG, _BIG, -0.1],        # feasible but degenerate
    ])])
    assert _objective_scale(polluted, (2.0, 0.7)) == pytest.approx(
        _objective_scale(clean, (2.0, 0.7))
    )


def test_objective_scale_is_robust_to_one_wild_feasible_point():
    """A single terrible-but-feasible set must not set the scale by itself."""
    from fit_pcsaft._pure.pareto import _objective_scale

    R = np.array([
        [1.0, 2.0, -0.1], [2.0, 2.2, -0.1], [3.0, 2.4, -0.1],
        [4.0, 2.6, -0.1], [5.0, 2.8, -0.1], [6.0, 3.0, -0.1],
    ])
    tame = _objective_scale(R, (2.0, 0.7))
    wild = _objective_scale(
        np.vstack([R, [[5000.0, 900.0, -0.1]]]), (2.0, 0.7)
    )
    assert wild[0] < 10 * tame[0], "one outlier hijacked the AAD_vle scale"
    assert wild[1] < 10 * tame[1], "one outlier hijacked the AAD_sft scale"


def test_objective_scale_falls_back_when_almost_nothing_is_feasible():
    """Gen zero can be nearly all infeasible for an associating fluid."""
    from fit_pcsaft._pure.pareto import _objective_scale

    R = np.array([[1.0, 2.0, -0.1], [4.0, 9.0, 0.5], [_BIG, _BIG, 1.0]])
    assert _objective_scale(R, (2.0, 0.7)) == (2.0, 0.7)


def test_lhs_sampling_selected():
    """Generation zero must use Latin hypercube coverage by default."""
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    from fit_pcsaft._pure.pareto import _initial_sampling

    assert isinstance(_initial_sampling(True), LatinHypercubeSampling)
    assert isinstance(_initial_sampling(False), FloatRandomSampling)


def test_ref_dirs_give_one_weight_vector_per_population_slot():
    """MOEA/D's population size *is* its number of weight vectors.

    Das-Dennis on two objectives returns n_partitions + 1 vectors, so pop_size
    has to map to n_partitions - 1 for the argument to keep its meaning.
    """
    from fit_pcsaft._pure.pareto import _ref_dirs

    W = _ref_dirs(60)
    assert W.shape == (60, 2)
    assert W.sum(axis=1) == pytest.approx(np.ones(60))


def test_ref_dirs_rejects_a_degenerate_population():
    from fit_pcsaft._pure.pareto import _ref_dirs

    with pytest.raises(ValueError, match="at least 2"):
        _ref_dirs(1)


def test_algorithm_is_the_parallel_moead_variant():
    """Plain MOEAD is loopwise and would starve the worker pool.

    Its _next yields one offspring at a time, so the pool would get batches of
    one and the measured 4.3x on 14 workers would be gone. ParallelMOEAD emits
    a whole population per generation.
    """
    from pymoo.algorithms.moo.moead import ParallelMOEAD

    from fit_pcsaft._pure.pareto import _make_algorithm

    algorithm = _make_algorithm(pop_size=20, lhs=True)
    assert isinstance(algorithm, ParallelMOEAD)
    assert algorithm.pop_size == 20
    assert algorithm.n_offsprings == 20, "one full population per generation"


def test_replacement_keeps_everything_under_the_cap():
    from fit_pcsaft._pure.pareto import _capped_replacement

    I = np.array([3, 7])
    out = _capped_replacement(I, 2, np.random.default_rng(0))
    assert out.tolist() == [3, 7]


def test_replacement_is_capped_when_an_offspring_beats_many_neighbours():
    """One solution flooding its whole neighbourhood is how the front collapses."""
    from fit_pcsaft._pure.pareto import _capped_replacement

    I = np.arange(20)
    out = _capped_replacement(I, 2, np.random.default_rng(0))
    assert len(out) == 2
    assert set(out.tolist()) <= set(I.tolist())


def test_replacement_does_not_always_take_the_same_slots():
    """Li and Zhang scan the neighbourhood in random order.

    Taking the first n, or greedily the n it improves most, reintroduces exactly
    the bias towards one region that the cap exists to remove.
    """
    from fit_pcsaft._pure.pareto import _capped_replacement

    rng = np.random.default_rng(0)
    I = np.arange(20)
    seen: set = set()
    for _ in range(50):
        seen.update(_capped_replacement(I, 2, rng).tolist())
    assert len(seen) > 4, "selection is not spread across the neighbourhood"


def test_algorithm_caps_neighbour_replacement():
    """pymoo's own _replace has no cap; pagmo's limit=2 is what the paper used."""
    from fit_pcsaft._pure.pareto import _N_REPLACE, _make_algorithm

    algorithm = _make_algorithm(pop_size=20, lhs=True)
    assert algorithm.n_replace == _N_REPLACE == 2


def test_moead_settings_that_still_match_pymoo_stock():
    """Four of the six settings exposed on this branch are still pymoo's own.
    Pinned so a pymoo upgrade that moves one is loud rather than silent; the two
    that are deliberately different have their own test below."""
    from pymoo.decomposition.tchebicheff import Tchebicheff
    from pymoo.operators.mutation.pm import PM

    from fit_pcsaft._pure.pareto import _make_algorithm

    algo = _make_algorithm(20, True)
    assert algo.n_neighbors == 20
    assert algo.selection.prob.value == pytest.approx(0.9)
    assert isinstance(algo.mating.mutation, PM)
    assert algo.mating.mutation.eta.value == pytest.approx(20)
    # decomposition is resolved in _setup when left as None; ours is set eagerly
    assert isinstance(algo.decomposition, Tchebicheff)


def test_moead_settings_deliberately_moved_from_pymoo_stock():
    """Two, and only two.

    ``n_replace=2`` is Li & Zhang (2009)'s MOEA/D-DE replacement cap n_r, where
    pymoo's stock MOEA/D lets one offspring take over every neighbour it beats.
    It predates this branch.

    The crossover is what this branch moved: pymoo ships ``SBX(eta=20)`` and the
    default is now DE/rand/1/bin. The two belong together -- n_r=2 was adopted
    while pymoo's SBX stayed, which is half of MOEA/D-DE; this pairs the cap with
    the operator it was designed for. Evidence in CLAUDE.md and
    ``benchmarks/*-transfer/``.
    """
    from pymoo.operators.crossover.dex import DEX

    from fit_pcsaft._pure.pareto import _make_algorithm

    algo = _make_algorithm(20, True)
    assert algo.n_replace == 2
    assert isinstance(algo.mating.crossover, DEX)
    assert algo.mating.crossover.F == pytest.approx(0.5)
    assert algo.mating.crossover.CR == pytest.approx(1.0)


def test_sbx_variant_is_still_reachable():
    """The old default has to stay available and stay SBX: TESIS's committed
    parameters were produced under it, so reproducing them means passing
    ``variant="sbx"`` explicitly."""
    from pymoo.operators.crossover.sbx import SBX

    from fit_pcsaft._pure.pareto import _make_algorithm

    algo = _make_algorithm(20, True, variant="sbx")
    assert isinstance(algo.mating.crossover, SBX)
    assert algo.mating.crossover.eta.value == pytest.approx(20)


def test_de_variant_swaps_the_crossover_operator():
    """n_r=2 is MOEA/D-DE's replacement cap (Li & Zhang 2009), adopted here while
    pymoo's SBX stayed. The DE variant is what that cap was designed to pair with."""
    from pymoo.operators.crossover.dex import DEX

    from fit_pcsaft._pure.pareto import _make_algorithm

    algo = _make_algorithm(20, True, variant="de")
    assert isinstance(algo.mating.crossover, DEX)


def test_dex_do_pinned_to_vendored_pymoo_version():
    """pareto.py's ``_SeededDEX`` (inside ``_make_algorithm``, variant="de")
    vendors ``DEX.do``'s body verbatim from pymoo 0.6.2, with one line fixed:
    the ``repair_random_init`` call upstream makes with no ``random_state``,
    which silently fabricated a fresh unseeded RNG and made ``variant="de"``
    non-reproducible even at a fixed seed.

    This test pins ``pymoo.__version__`` on purpose, so that upgrading pymoo
    fails this test loudly instead of letting the vendored copy silently drift
    out of sync with upstream's ``DEX.do``. If this test fails after a pymoo
    bump: re-diff ``pymoo/operators/crossover/dex.py``'s ``DEX.do`` against
    ``_SeededDEX.do`` in pareto.py by hand (has upstream fixed the missing
    ``random_state``? did anything else in ``do`` change?) before touching
    this pin -- do not just bump the version and move on.
    """
    import pymoo

    assert pymoo.__version__ == "0.6.2"


def test_de_variant_is_reproducible():
    """The bug this whole vendored override exists for: variant="de" used to
    give a different front on every run at an identical config and seed,
    because DEX.do's out-of-bounds repair (see _SeededDEX's docstring in
    pareto.py) drew from a fresh unseeded RNG every time it fired -- and it
    fires often, because DE extrapolates outside the parameter box. variant
    "sbx" never hit this (interpolates within the parent hull), which is why
    only "de" needed the fix.

    Confirmed against the unfixed code before writing this test: reverting
    _SeededDEX's one corrected line (dropping ``random_state=random_state``
    from the ``repair_random_init`` call, i.e. upstream's own bug) makes this
    test fail -- two runs below returned different front shapes/objectives.
    With the fix in place they are identical.
    """
    if bool(os.environ.get("NO_NETWORK")):
        pytest.skip("fit_pure_pareto -> _fetch_compound hits PubChem")

    from fit_pcsaft import fit_pure_pareto

    d = Path(__file__).parent.parent / "examples" / "data"
    kwargs = dict(
        id="water",
        psat_path=d / "psat" / "water.csv",
        density_path=d / "density" / "water.csv",
        na=1, nb=1,
        objectives=("psat", "rho"),
        bounds=[(0.8, 3.0), (2.0, 3.5), (150.0, 400.0), (1e-3, 0.35), (1500.0, 4000.0)],
        pop_size=20, n_gen=10, n_restarts=1, refine=0,
        seed=7, variant="de", n_jobs=1, verbose=False,
    )
    r1 = fit_pure_pareto(**kwargs)
    r2 = fit_pure_pareto(**kwargs)
    assert r1.X.shape == r2.X.shape
    assert np.array_equal(r1.X, r2.X)
    assert np.array_equal(r1.F, r2.F)


def test_pbi_decomposition_is_selectable_and_carries_theta():
    from pymoo.decomposition.pbi import PBI

    from fit_pcsaft._pure.pareto import _make_algorithm

    algo = _make_algorithm(20, True, decomposition="pbi", pbi_theta=5.0)
    assert isinstance(algo.decomposition, PBI)
    assert algo.decomposition.theta == pytest.approx(5.0)


def test_unknown_knob_values_are_rejected():
    from fit_pcsaft._pure.pareto import _make_algorithm

    for kw in ({"decomposition": "weighted-sum"}, {"variant": "pso"}):
        with pytest.raises(ValueError):
            _make_algorithm(20, True, **kw)


def test_de_variant_rejects_too_few_neighbors():
    """DEX draws 3 parents per neighbourhood without replacement -- fewer than
    3 neighbours would otherwise fail deep inside numpy's random.choice."""
    from fit_pcsaft._pure.pareto import _make_algorithm

    with pytest.raises(ValueError, match="n_neighbors"):
        _make_algorithm(20, True, variant="de", n_neighbors=2)


def test_problem_declares_no_constraints():
    """pymoo's MOEA/D asserts on any constrained problem in its _setup."""
    from fit_pcsaft._pure.pareto import _make_problem

    problem = _make_problem(
        HEXANE, HEXANE_SPEC, _hexane_data_with_sft(), Units(),
        [(1.0, 5.0), (3.0, 4.5), (150.0, 350.0)], None,
    )
    assert not problem.has_constraints(), "MOEA/D would abort at setup"
    assert problem.n_obj == 2
    assert problem.n_var == 3


def test_problem_freezes_its_scale_after_the_first_population():
    """A drifting scale would corrupt MOEA/D's replacement comparisons.

    _replace weighs an offspring's fresh F against F stored on the population
    when it was evaluated. Those are only comparable if the same divisor
    produced both, so the scale is estimated once and then held.
    """
    from fit_pcsaft._pure.pareto import _make_problem

    problem = _make_problem(
        HEXANE, HEXANE_SPEC, _hexane_data_with_sft(), Units(),
        [(1.0, 5.0), (3.0, 4.5), (150.0, 350.0)], None,
    )
    assert problem.scale is None, "nothing to estimate from yet"

    out: dict = {}
    problem._evaluate(np.array([HEXANE_P, HEXANE_P * 1.02]), out)
    frozen = problem.scale
    assert frozen is not None, "first population must set the scale"

    problem._evaluate(np.array([HEXANE_P * 1.05]), out)
    assert problem.scale == frozen, "scale must never be re-estimated"


def test_front_keeps_only_feasible_non_dominated_rows_in_order():
    """The driver, not pymoo, is now what keeps infeasible points off the front."""
    from fit_pcsaft._pure.pareto import _front_from

    X = np.array([
        [1.0, 3.0, 200.0],   # feasible, on the front
        [2.0, 3.0, 200.0],   # feasible, dominated by row 0
        [3.0, 3.0, 200.0],   # feasible, on the front
        [4.0, 3.0, 200.0],   # best objectives of all, but infeasible
        [5.0, 3.0, 200.0],   # feasible but degenerate
    ])
    R = np.array([
        [2.0, 5.0, -0.1],
        [3.0, 6.0, -0.1],
        [5.0, 1.0, -0.1],
        [0.1, 0.1, 0.5],
        [_BIG, _BIG, -0.1],
    ])
    Xf, F = _front_from(X, R)

    assert F.tolist() == [[2.0, 5.0], [5.0, 1.0]]
    assert Xf.tolist() == [[1.0, 3.0, 200.0], [3.0, 3.0, 200.0]]


def test_front_raises_when_nothing_is_feasible():
    from fit_pcsaft._pure.pareto import _front_from

    X = np.array([[1.0, 3.0, 200.0], [2.0, 3.0, 200.0]])
    R = np.array([[1.0, 1.0, 0.3], [_BIG, _BIG, 1.0]])
    with pytest.raises(RuntimeError, match="infeasible"):
        _front_from(X, R)


def test_merge_keeps_the_best_of_each_run_and_sorts_by_aad_vle():
    """Two runs that each found one end of the front make one front together.

    This is the measured water case in miniature: run A owns the low-AAD_vle
    basin, run B the low-AAD_sft one, and each has points the other dominates.
    """
    from fit_pcsaft._pure.pareto import _merge_fronts

    X_a = np.array([[1.8, 2.35, 240.0], [1.8, 2.36, 241.0]])
    F_a = np.array([[1.5, 1.5], [6.0, 1.4]])          # second point is poor
    X_b = np.array([[1.2, 2.78, 272.0], [1.0, 2.86, 274.0]])
    F_b = np.array([[2.5, 1.1], [12.8, 0.5]])

    X, F = _merge_fronts([(X_a, F_a), (X_b, F_b)])

    assert F.tolist() == [[1.5, 1.5], [2.5, 1.1], [12.8, 0.5]]
    assert X[0].tolist() == [1.8, 2.35, 240.0]
    assert X[1].tolist() == [1.2, 2.78, 272.0]


def test_merge_of_one_run_is_that_run():
    """n_restarts=1 must be the old code path, point for point."""
    from fit_pcsaft._pure.pareto import _merge_fronts

    X_a = np.array([[1.0, 3.0, 200.0], [2.0, 3.0, 210.0]])
    F_a = np.array([[2.0, 5.0], [5.0, 1.0]])
    X, F = _merge_fronts([(X_a, F_a)])

    assert F.tolist() == F_a.tolist()
    assert X.tolist() == X_a.tolist()


def test_merge_drops_a_run_that_is_entirely_behind_another():
    """The measured case: three of four MOEA/D runs contributed nothing."""
    from fit_pcsaft._pure.pareto import _merge_fronts

    good = (np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 1.0]]))
    behind = (np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]]),
              np.array([[5.0, 5.0], [6.0, 4.0]]))
    X, F = _merge_fronts([good, behind])

    assert F.tolist() == [[1.0, 1.0]]


def test_select_picks_the_tangent_point():
    F = np.array([[1.0, 6.0], [2.0, 2.0], [6.0, 1.0]])
    # refs=(2%, 1) -> costs: 6.5, 3.0, 4.0 -> index 1
    assert _argmin_scalarized(F, (2.0, 1.0)) == 1
    # heavy weight on sft (small refs[1]) -> the low-sft corner wins
    assert _argmin_scalarized(F, (100.0, 0.1)) == 2


def _hexane_data_conflicting():
    """Bulk data from HEXANE_P, gamma from a different sigma.

    The self-consistent fixture cannot exercise densify: both objectives are
    minimised by the same parameter vector, so its "front" is a single point.
    Taking gamma from a shifted parameter set makes the two objectives pull in
    genuinely different directions, which is the case densify exists for.
    """
    import numpy as np

    from fit_pcsaft._fit_utils import _build_functional
    from fit_pcsaft._pure.surface_tension import predict_surface_tension
    from fit_pcsaft._types import PureData, Units
    from tests.test_surface_tension import HEXANE, HEXANE_P, HEXANE_SPEC

    T = np.array([300.0, 320.0, 340.0])
    shifted = HEXANE_P * np.array([1.0, 1.03, 1.0])
    gamma = predict_surface_tension(
        _build_functional(shifted, HEXANE, HEXANE_SPEC), T, Units()
    )
    base = _hexane_data_with_sft()
    return PureData(
        T_psat=base.T_psat, p_psat=base.p_psat,
        T_rho=base.T_rho, rho=base.rho,
        T_sft=T, sft=gamma,
    )


def test_densify_fills_gaps_without_leaving_the_front():
    """Given a real front, densify must keep every point and add more."""
    import numpy as np

    from fit_pcsaft._pure.pareto import _densify, non_dominated
    from fit_pcsaft._types import Units
    from tests.test_surface_tension import HEXANE, HEXANE_P, HEXANE_SPEC

    data = _hexane_data_conflicting()
    units = Units()

    def _obj(x):
        return aad_objectives(np.asarray(x, float), HEXANE, HEXANE_SPEC, data, units)

    # Sweep sigma between the two objectives' optima; the non-dominated subset
    # is the starting front, exactly as the driver builds it.
    cand = np.array(
        [[HEXANE_P[0], s, HEXANE_P[2]]
         for s in np.linspace(HEXANE_P[1], HEXANE_P[1] * 1.03, 6)]
    )
    Fc = np.array([_obj(x) for x in cand])
    keep = non_dominated(Fc) & (Fc[:, 0] < _BIG)
    order = np.argsort(Fc[keep, 0])
    X, F = cand[keep][order], Fc[keep][order]
    assert len(F) >= 2, "sweep did not produce a front to densify"

    Xd, Fd = _densify(X, F, 3, None, HEXANE, HEXANE_SPEC, data, units, None)

    assert len(Fd) > len(F), "densify added no points"
    assert len(Xd) == len(Fd)
    assert np.all(np.diff(Fd[:, 0]) >= 0.0), "front must stay sorted by AAD_vle"
    for f in F:
        assert np.any(np.all(np.isclose(Fd, f), axis=1)), f"lost front point {f}"
    # every returned pair is a real evaluation of its own parameter vector,
    # never a drawn-in interpolation of the objectives
    for x, f in zip(Xd, Fd):
        assert _obj(x) == pytest.approx(tuple(f), rel=1e-9)
    assert non_dominated(Fd).all()


def test_densify_is_a_noop_when_disabled():
    """refine=0 must leave the raw search output untouched."""
    import numpy as np

    from fit_pcsaft._pure.pareto import _densify
    from fit_pcsaft._types import Units
    from tests.test_surface_tension import HEXANE, HEXANE_SPEC, _hexane_data_with_sft

    X = np.array([[3.0576, 3.7983, 236.77], [3.2000, 3.7500, 233.00]])
    F = np.array([[1.0, 2.0], [2.0, 1.0]])
    Xd, Fd = _densify(
        X, F, 0, None, HEXANE, HEXANE_SPEC, _hexane_data_with_sft(), Units(), None
    )
    assert np.array_equal(Xd, X) and np.array_equal(Fd, F)


def test_to_csv_column_names_follow_the_objectives(tmp_path):
    """The header is an on-disk contract: examples/pure/12_water_vs_rehner.py
    reads `aad_vle_pct`/`aad_sft` back off a saved front."""
    import polars as pl

    from fit_pcsaft._pure.pareto import ParetoResult

    def _front(objectives):
        return ParetoResult(
            X=np.array([[3.0576, 3.7983, 236.77]]), F=np.array([[1.0, 2.0]]),
            data=_hexane_data_with_sft(), compound=HEXANE, spec=HEXANE_SPEC,
            units=Units(), fit_mu=False, is_associative=False, time_elapsed=0.0,
            objectives=objectives,
        )

    p = tmp_path / "front.csv"
    _front(("vle", "sft")).to_csv(p)
    assert pl.read_csv(p).columns == ["aad_vle_pct", "aad_sft", "m", "sigma", "epsilon_k"]
    _front(("psat", "rho")).to_csv(p)
    assert pl.read_csv(p).columns == ["aad_psat_pct", "aad_rho_pct", "m", "sigma", "epsilon_k"]


def test_str_is_table_driven_and_labels_each_mode_correctly():
    """A mislabeled __str__ is the bug this task exists to fix: a real Forte
    run printed 'best AAD_vle'/'AAD_sft there' axis labels for psat/rho data."""
    from fit_pcsaft._pure.pareto import ParetoResult

    def _front(objectives):
        return ParetoResult(
            X=np.array([[3.0576, 3.7983, 236.77], [3.10, 3.75, 233.0]]),
            F=np.array([[1.0, 2.0], [2.0, 1.0]]),
            data=_hexane_data_with_sft(), compound=HEXANE, spec=HEXANE_SPEC,
            units=Units(), fit_mu=False, is_associative=False, time_elapsed=0.0,
            input_name="hexane", objectives=objectives,
        )

    # ("vle", "sft") must reproduce the pre-refactor hardcoded string
    # byte-for-byte: "%" attaches to the number with no space, and the
    # sft number never carries a unit at all -- that asymmetry was already
    # in the old hardcoded __str__ (mN/m was simply never printed), and the
    # global constraints require default-mode text to stay bit-identical.
    assert str(_front(("vle", "sft"))) == (
        "Pareto front — hexane\n"
        "  points: 2   time: 0.0 s\n"
        "  best AAD_vle: 1.00% (AAD_sft there: 2.00)\n"
        "  best AAD_sft: 1.00 (AAD_vle there: 2.00%)"
    )
    # ("psat", "rho") shares the same "%" formatting rule for both axes --
    # both are percent quantities, so both numbers now carry "%", with the
    # correct labels instead of the "AAD_vle"/"AAD_sft" mislabeling.
    assert str(_front(("psat", "rho"))) == (
        "Pareto front — hexane\n"
        "  points: 2   time: 0.0 s\n"
        "  best AARD_psat: 1.00% (AARD_rho there: 2.00%)\n"
        "  best AARD_rho: 1.00% (AARD_psat there: 2.00%)"
    )


def test_select_resolves_per_mode_default_refs_and_reports_sft():
    """refs=None -> _DEFAULT_REFS[objectives], and under ("psat", "rho") that
    is what newly reaches select()'s len(data.T_sft) > 0 functional build --
    the "DFT-solves once regardless of objectives" precedent the module and
    fit_pure_pareto docstrings assert. X uses the real, self-consistent
    HEXANE_P so _build_eos/_build_functional succeed."""
    from fit_pcsaft._pure.pareto import ParetoResult

    def _front(objectives):
        return ParetoResult(
            X=np.array([HEXANE_P]), F=np.array([[1.0, 2.0]]),
            data=_hexane_data_with_sft(), compound=HEXANE, spec=HEXANE_SPEC,
            units=Units(), fit_mu=False, is_associative=False, time_elapsed=0.0,
            objectives=objectives,
        )

    rehner = _front(("vle", "sft")).select()
    assert "refs=(2.0, 0.7)" in rehner.scipy_result.message
    assert rehner.scipy_result.success and np.isfinite(rehner.aad_sft)

    forte = _front(("psat", "rho")).select()
    assert "refs=(2.0, 2.0)" in forte.scipy_result.message
    assert forte.scipy_result.success
    # never an objective under this pair, but still reported -- self-consistent
    # fixture makes it exactly 0, not merely finite by accident
    assert forte.aad_sft == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Batched in-process evaluation for the bulk pair
# ---------------------------------------------------------------------------


def test_evaluate_population_matches_evaluate_point_row_for_row():
    """Under ('psat','rho') the search evaluates the whole population with one
    vectorised feos AD call per property. It must reproduce _evaluate_point on
    every row -- feasible, failing at some temperatures, failing at all of them,
    and outside the default box -- and the batch must actually contain those
    kinds, or the failure handling is untested.
    """
    from fit_pcsaft._pure.pareto import (
        _MIN_VALID_FRACTION,
        _evaluate_point,
        _evaluate_population,
    )
    from fit_pcsaft._types import ModelSpec

    water, water_c = _water_data()
    cases = [
        (HEXANE, HEXANE_SPEC, _hexane_data_with_sft(), [
            HEXANE_P,                            # feasible
            np.array([20.0, 6.0, 700.0]),        # inside the default box, no VLE here
            np.array([1e-6, 1e-6, 1e-6]),        # outside it; the old path's "hopeless" row
        ]),
        (water_c, ModelSpec(mu=0.0, na=1, nb=1, q=0.0), water, [
            np.array(WATER_SCHEMES[0][1]),                    # Table 1, feasible
            np.array([1.0, 2.0, 50.0, 1e-4, 500.0]),          # Tc far below every data point
            np.array([1.0, 2.9375, 150.0, 0.0445, 1500.0]),   # 7 of 23 points converge: partial
        ]),
    ]
    seen_partial = False
    for comp, spec, data, rows in cases:
        got = _evaluate_population(np.array(rows), comp, spec, data, Units(), FORTE)
        want = np.array([_evaluate_point(x, comp, spec, data, Units(), None, FORTE) for x in rows])
        assert got.shape == (len(rows), 3)
        np.testing.assert_allclose(got[:, :2], want[:, :2], rtol=1e-9)
        np.testing.assert_array_equal(got[:, 2], want[:, 2])
        v = got[:, 2]
        assert (v <= 0.0).any(), "no feasible row in the batch"
        assert (v == _MIN_VALID_FRACTION).any(), "no all-failed row in the batch"
        seen_partial |= bool(((v > 0.0) & (v < _MIN_VALID_FRACTION)).any())
    assert seen_partial, "no partially-failing row in either batch"


def test_evaluate_population_takes_a_2d_array_and_a_single_row():
    """pymoo hands _evaluate a (pop, n_var) array; polish_front hands one row at a time."""
    from fit_pcsaft._pure.pareto import _evaluate_population

    data = _hexane_data_with_sft()
    one = _evaluate_population(HEXANE_P, HEXANE, HEXANE_SPEC, data, Units(), FORTE)
    two = _evaluate_population(
        np.array([HEXANE_P, HEXANE_P * 1.01]), HEXANE, HEXANE_SPEC, data, Units(), FORTE
    )
    assert one.shape == (1, 3) and two.shape == (2, 3)
    assert np.isfinite(one).all() and np.isfinite(two).all()


def test_map_evaluate_routes_the_bulk_pair_to_the_batched_path(monkeypatch):
    """With sft out of the objectives, _map_evaluate never calls _evaluate_point."""
    from fit_pcsaft._pure import pareto

    def _boom(*args, **kwargs):
        raise _NoDft("the per-point path was entered")

    monkeypatch.setattr(pareto, "_evaluate_point", _boom)
    got = np.asarray(
        pareto._map_evaluate(
            [HEXANE_P], None, HEXANE, HEXANE_SPEC, _hexane_data_with_sft(), Units(), None, FORTE
        ),
        dtype=float,
    )
    assert got.shape == (1, 3) and np.isfinite(got).all()


@pytest.mark.parametrize("spec,objectives", [
    (HEXANE_SPEC, ("vle", "sft")),   # sft needs the DFT: per point, through the pool
    (None, FORTE),                   # q != 0 has no feos AD model: per point
])
def test_sft_mode_and_quadrupole_stay_on_the_per_point_path(monkeypatch, spec, objectives):
    from fit_pcsaft._pure import pareto
    from fit_pcsaft._types import ModelSpec

    if spec is None:
        spec = ModelSpec(mu=0.0, na=None, nb=None, q=1.0)

    def _boom(*args, **kwargs):
        raise _NoDft("the batched path was entered")

    monkeypatch.setattr(pareto, "_evaluate_population", _boom)
    got = pareto._map_evaluate(
        [HEXANE_P], None, HEXANE, spec, _hexane_data_with_sft(), Units(), None, objectives
    )
    assert np.isfinite(got[0][0])


def test_bulk_pair_never_spawns_a_pool(monkeypatch, tmp_path):
    """n_jobs is a no-op for ('psat','rho'): feos's rayon threads do the work
    in-process, and a spawn pool would only add pickling and one-thread workers."""
    from contextlib import contextmanager

    from fit_pcsaft._pure import fit as fit_mod
    from fit_pcsaft._pure import pareto

    seen = []

    @contextmanager
    def fake_pool(n_jobs, *args):
        seen.append(n_jobs)
        yield None

    monkeypatch.setattr(pareto, "_worker_pool", fake_pool)
    monkeypatch.setattr(fit_mod, "_fetch_compound", lambda _id: (HEXANE.identifier, HEXANE.mw))
    (tmp_path / "psat.csv").write_text("T,psat\n300.0,21.9\n320.0,43.9\n340.0,79.5\n")
    (tmp_path / "rho.csv").write_text("T,rho\n300.0,654.9\n320.0,635.6\n340.0,615.4\n")

    res = pareto.fit_pure_pareto(
        id="hexane", psat_path=tmp_path / "psat.csv", density_path=tmp_path / "rho.csv",
        objectives=FORTE, bounds=[(2.0, 4.0), (3.5, 4.0), (200.0, 260.0)],
        pop_size=20, n_gen=2, n_restarts=1, refine=0, n_jobs=4, verbose=False,
    )
    assert seen == [1], "the bulk pair must run in-process, never through workers"
    assert len(res.F) >= 1


@pytest.mark.parametrize("refine", [0, 2])
def test_polished_front_is_densified_again_and_sorted(monkeypatch, refine):
    """polish moves points onto the true front and drops what they now dominate,
    which can leave a handful of anchors (2-phenylethanol, 2026-09-02: 51 -> 6).
    A second _densify pass after it interpolates between those anchors -- real
    parameter sets, re-evaluated -- and whatever runs last, the front comes back
    sorted by F[:, 0], the ParetoResult contract that polish alone broke (five of
    seven TESIS fronts came back unsorted). polish_front is wrapped to hand its
    rows back reversed, so the sort has to be done by the caller to pass.
    """
    from fit_pcsaft._pure import fit as fit_mod
    from fit_pcsaft._pure import pareto, polish

    data, water = _water_data()
    calls = []
    real_densify = pareto._densify

    def spy(X, F, *args, **kwargs):
        calls.append(len(F))
        return real_densify(X, F, *args, **kwargs)

    real_polish = polish.polish_front

    def reversed_polish(X, F, evaluate, bounds, **kwargs):
        X_p, F_p = real_polish(X, F, evaluate, bounds, **kwargs)
        return X_p[::-1], F_p[::-1]

    monkeypatch.setattr(pareto, "_densify", spy)
    # fit_pure_pareto imports polish_front inside the function, so the patch goes
    # on the polish module, where that import looks it up.
    monkeypatch.setattr(polish, "polish_front", reversed_polish)
    monkeypatch.setattr(fit_mod, "_fetch_compound", lambda _id: (water.identifier, water.mw))
    d = Path(__file__).parent.parent / "examples" / "data"
    res = pareto.fit_pure_pareto(
        id="water", psat_path=d / "psat" / "water.csv", density_path=d / "density" / "water.csv",
        na=1, nb=1, objectives=FORTE,
        bounds=[(0.8, 3.0), (2.0, 3.5), (150.0, 400.0), (1e-3, 0.35), (1500.0, 4000.0)],
        pop_size=20, n_gen=30, n_restarts=1, refine=refine, polish=True, seed=7,
        n_jobs=1, verbose=False,
    )
    assert len(res.F) > 1, "the search must return a front for this test to mean anything"
    assert len(calls) == (0 if refine == 0 else 2), "densify runs before AND after polish"
    assert np.all(np.diff(res.F[:, 0]) >= 0), "rows must be sorted by F[:, 0] after polish"


# ---------------------------------------------------------------------------
# n_obj = len(objectives): everything below the objective table is dimension-generic
# ---------------------------------------------------------------------------


def test_ref_dirs_three_objectives_round_pop_size_up_to_the_das_dennis_fan():
    """Das-Dennis on three objectives has (p+1)(p+2)/2 vectors, so pop_size is
    rounded up to the smallest fan that holds it (60 -> 66). Two objectives are
    untouched: exactly pop_size vectors, as before."""
    from fit_pcsaft._pure.pareto import _ref_dirs

    W = _ref_dirs(60, 3)
    assert W.shape == (66, 3)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert _ref_dirs(55, 3).shape == (55, 3)
    assert _ref_dirs(60, 2).shape == (60, 2)
    assert _ref_dirs(60).shape == (60, 2)
    with pytest.raises(ValueError, match="objectives"):
        _ref_dirs(4, 4)


def test_scalarization_penalty_scale_and_front_take_three_objectives():
    """The violation is the last column of an evaluation row, whatever the
    number of objectives before it."""
    from fit_pcsaft._pure.pareto import (
        _BIG,
        _argmin_scalarized,
        _front_from,
        _merge_fronts,
        _objective_scale,
        _penalize,
    )

    F = np.array([[1.0, 6.0, 2.0], [2.0, 2.0, 2.0], [6.0, 1.0, 1.0]])
    assert _argmin_scalarized(F, (2.0, 1.0, 2.0)) == 1      # 7.5, 4.0, 4.5
    assert _argmin_scalarized(F, (100.0, 0.1, 100.0)) == 2  # the low-f2 corner

    R = np.column_stack([F, [0.0, 0.2, -0.1]])
    P = _penalize(R, (1.0, 2.0, 4.0))
    assert P.shape == (3, 3)
    np.testing.assert_allclose(P[0], F[0] / [1.0, 2.0, 4.0])
    np.testing.assert_allclose(P[2], F[2] / [1.0, 2.0, 4.0])
    np.testing.assert_allclose(P[1], F[1] / [1.0, 2.0, 4.0] + _BIG * 0.2)

    feasible = np.column_stack([np.tile(F, (2, 1)), np.zeros(6)])
    scale = _objective_scale(feasible, (9.0, 9.0, 9.0))
    assert len(scale) == 3 and all(s > 0 for s in scale)
    assert _objective_scale(R, (9.0, 9.0, 9.0)) == (9.0, 9.0, 9.0)   # too few feasible rows

    X = np.array([[1.0], [2.0], [3.0]])
    Xf, Ff = _front_from(X, np.column_stack([F, [0.0, 0.5, 0.0]]))
    assert Ff.shape == (2, 3) and Xf[:, 0].tolist() == [1.0, 3.0]
    _, Fm = _merge_fronts([(Xf, Ff), (Xf, Ff)])
    assert Fm.shape == (2, 3) and np.all(np.diff(Fm[:, 0]) >= 0)


def test_pareto_result_renders_three_objectives_and_selects_by_summed_cost(tmp_path):
    """One 'best' line per objective, the others comma-separated in the
    parenthesis; to_csv one column per objective; select() sums F/refs over
    every axis. Uses ("psat", "rho", "sft") purely as three table keys."""
    import polars as pl

    from fit_pcsaft._pure.pareto import ParetoResult

    X = np.array([[3.0576, 3.7983, 236.77], [3.10, 3.75, 233.0], [3.0, 3.8, 240.0]])
    F = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 3.0, 0.5]])
    res = ParetoResult(
        X=X, F=F, data=_hexane_data_with_sft(), compound=HEXANE, spec=HEXANE_SPEC,
        units=Units(), fit_mu=False, is_associative=False, time_elapsed=0.0,
        input_name="hexane", objectives=("psat", "rho", "sft"),
    )
    assert str(res) == (
        "Pareto front — hexane\n"
        "  points: 3   time: 0.0 s\n"
        "  best AARD_psat: 1.00% (AARD_rho there: 2.00%, AAD_sft there: 3.00)\n"
        "  best AARD_rho: 1.00% (AARD_psat there: 2.00%, AAD_sft there: 3.00)\n"
        "  best AAD_sft: 0.50 (AARD_psat there: 3.00%, AARD_rho there: 3.00%)"
    )
    p = tmp_path / "front.csv"
    res.to_csv(p)
    assert pl.read_csv(p).columns == ["aad_psat_pct", "aad_rho_pct", "aad_sft", "m", "sigma", "epsilon_k"]

    fit = res.select(refs=(1.0, 1.0, 1.0))      # costs 6, 6, 6.5 -> index 0
    assert fit.params["m"] == pytest.approx(3.0576)
    assert fit.scipy_result.fun[0] == pytest.approx(6.0)
    assert "refs=(1.0, 1.0, 1.0)" in fit.scipy_result.message
