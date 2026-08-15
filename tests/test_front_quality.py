import numpy as np


def test_igd_plus_prefers_the_front_that_dominates():
    """The metric that earns its place: coverage(B, A) was inverted by one outlier on
    real data here. IGD+ is weakly Pareto-compliant (Ishibuchi et al. 2015), so a
    dominated set can never score better."""
    from fit_pcsaft._pure.front_quality import front_metrics, reference_front

    good = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    bad = good + 0.5                       # strictly dominated, same shape and spread
    ref = reference_front(good, bad)
    assert front_metrics(good, reference=ref)["igd_plus"] < (
        front_metrics(bad, reference=ref)["igd_plus"]
    )


def test_the_region_of_interest_can_flip_the_verdict():
    """The failure that motivated this module, in miniature -- and note what it is NOT.

    A point that dominates everything really is better by every indicator; no metric
    "fixes" that. What went wrong on the real water fronts is that one tail point at
    (7.73, 0.524) dominated 217 of 522 points, so the whole-front C-metric answered a
    question about a region nobody cares about. Restricted to AARD_psat <= 2% the
    reading reversed (0.912 against 0.075). The cure is the clip, not the indicator.
    """
    from fit_pcsaft._pure.front_quality import compare_fronts

    A = np.array([[1.5, 1.90], [8.0, 0.50]])                 # weak head, strong tail
    B = np.array([[0.8, 1.20], [1.4, 1.00],
                  [9.0, 0.95], [20.0, 0.90], [30.0, 0.80]])  # strong head, weak tail

    whole = compare_fronts(A, B)
    assert whole["coverage_ab"] > whole["coverage_ba"]        # A wins on the whole front

    head = compare_fronts(A, B, region=[(0.0, 2.0), None])
    assert head["coverage_ba"] > head["coverage_ab"]          # B wins where it matters
    assert head["a"]["n_points"] == 1 and head["b"]["n_points"] == 2


def test_reference_front_is_the_non_dominated_union():
    from fit_pcsaft._pure.front_quality import reference_front

    a = np.array([[1.0, 4.0], [3.0, 3.0]])
    b = np.array([[2.0, 2.0], [9.0, 9.0]])
    ref = reference_front(a, b)
    # (2,2) dominates both (3,3) and (9,9); (1,4) survives on axis 0
    assert len(ref) == 2
    assert sorted(ref.tolist()) == [[1.0, 4.0], [2.0, 2.0]]


def test_an_empty_region_is_reported_not_crashed():
    """A clip can legitimately empty a front -- the sweep will hit this on cells whose
    every point sits outside the region of interest. n_points == 0 with NaN metrics is
    a usable row in a CSV; an exception loses the whole sweep."""
    from fit_pcsaft._pure.front_quality import front_metrics

    m = front_metrics(np.array([[8.0, 0.5]]), region=[(0.0, 2.0), None])
    assert m["n_points"] == 0
    assert np.isnan(m["hv"])


def test_hv_reference_point_defaults_above_the_worst_point():
    """HV needs a reference point; a default below any point silently returns zero."""
    from fit_pcsaft._pure.front_quality import front_metrics

    F = np.array([[1.0, 4.0], [4.0, 1.0]])
    assert front_metrics(F)["hv"] > 0.0


def test_hv_ranks_a_dominating_front_higher_under_a_shared_ref_point():
    """Task 3 scores ~45 fronts against one shared ref_point specifically so the
    hypervolumes can be read side by side. This pins that comparability directly:
    same ref_point, a strictly dominating front must score strictly higher.

    It does NOT pin pymoo's norm_ref_point flag -- on this pymoo build (0.6.2) that
    flag is a no-op regardless of value (see front_quality.py's module docstring), so
    this test passes identically whether front_metrics passes norm_ref_point=False or
    True. The explicit False there is a defensive default against a future pymoo
    release wiring the flag up for real, not something this test can detect."""
    from fit_pcsaft._pure.front_quality import front_metrics

    good = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    bad = good + 0.5                       # strictly dominated by good, same shape
    ref_point = np.array([10.0, 10.0])

    hv_good = front_metrics(good, ref_point=ref_point)["hv"]
    hv_bad = front_metrics(bad, ref_point=ref_point)["hv"]
    assert hv_good > hv_bad
