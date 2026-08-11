from pathlib import Path

import numpy as np
import pytest

from fit_pcsaft._pure.pareto import _BIG, _argmin_scalarized, aad_objectives, non_dominated
from fit_pcsaft._types import Units
from tests.test_surface_tension import (
    HEXANE,
    HEXANE_P,
    HEXANE_SPEC,
    _hexane_data_with_sft,
)


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
    out = _penalize(R, scale_vle=2.0, scale_sft=0.7)
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
    out = _penalize(R, scale_vle=2.0, scale_sft=0.7)
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
    s_vle, s_sft = _objective_scale(R, ref_vle=2.0, ref_sft=0.7)
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
    assert _objective_scale(polluted, 2.0, 0.7) == pytest.approx(
        _objective_scale(clean, 2.0, 0.7)
    )


def test_objective_scale_is_robust_to_one_wild_feasible_point():
    """A single terrible-but-feasible set must not set the scale by itself."""
    from fit_pcsaft._pure.pareto import _objective_scale

    R = np.array([
        [1.0, 2.0, -0.1], [2.0, 2.2, -0.1], [3.0, 2.4, -0.1],
        [4.0, 2.6, -0.1], [5.0, 2.8, -0.1], [6.0, 3.0, -0.1],
    ])
    tame = _objective_scale(R, 2.0, 0.7)
    wild = _objective_scale(
        np.vstack([R, [[5000.0, 900.0, -0.1]]]), 2.0, 0.7
    )
    assert wild[0] < 10 * tame[0], "one outlier hijacked the AAD_vle scale"
    assert wild[1] < 10 * tame[1], "one outlier hijacked the AAD_sft scale"


def test_objective_scale_falls_back_when_almost_nothing_is_feasible():
    """Gen zero can be nearly all infeasible for an associating fluid."""
    from fit_pcsaft._pure.pareto import _objective_scale

    R = np.array([[1.0, 2.0, -0.1], [4.0, 9.0, 0.5], [_BIG, _BIG, 1.0]])
    assert _objective_scale(R, ref_vle=2.0, ref_sft=0.7) == (2.0, 0.7)


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
    # ref_vle=2%, ref_sft=1 -> costs: 6.5, 3.0, 4.0 -> index 1
    assert _argmin_scalarized(F, ref_vle=2.0, ref_sft=1.0) == 1
    # heavy weight on sft (small ref_sft) -> the low-sft corner wins
    assert _argmin_scalarized(F, ref_vle=100.0, ref_sft=0.1) == 2


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
