import numpy as np
import pytest


def test_polish_moves_a_point_off_the_front_onto_it():
    """A point pushed off a known analytic front must come back to it."""
    from fit_pcsaft._pure.polish import polish_front

    # front: f2 = 1/f1 on f1 in [0.5, 4]; parameterize by theta = f1
    def evaluate(rows):
        t = np.asarray(rows, dtype=float)[:, 0]
        return np.column_stack([t, 1.0 / t, np.zeros_like(t)])

    X = np.array([[1.0], [2.0]])
    F = evaluate(X)[:, :2] + np.array([[0.3, 0.3], [0.3, 0.3]])   # pushed off
    Xp, Fp = polish_front(X, F, evaluate, [(0.5, 4.0)])
    assert np.all(Fp[:, 1] <= F[:, 1] + 1e-9)
    assert np.all(Fp[:, 0] <= F[:, 0] + 1e-9)


def test_polish_never_returns_a_worse_point_than_it_was_given():
    """The guard that makes polish safe to default-on later: a failed or diverging
    local solve must leave the input point untouched, not replace it."""
    from fit_pcsaft._pure.polish import polish_front

    def evaluate(rows):                      # every solve fails
        n = len(np.asarray(rows, dtype=float))
        return np.column_stack([np.full(n, np.nan), np.full(n, np.nan), np.ones(n)])

    X = np.array([[1.0], [2.0]])
    F = np.array([[1.0, 1.0], [2.0, 0.5]])
    Xp, Fp = polish_front(X, F, evaluate, [(0.5, 4.0)])
    assert np.array_equal(Fp, F) and np.array_equal(Xp, X)


def test_polished_front_is_non_dominated():
    """Polishing points independently can leave one dominating another, so the union
    must be re-filtered before it is returned."""
    from fit_pcsaft._pure.pareto import non_dominated
    from fit_pcsaft._pure.polish import polish_front

    def evaluate(rows):
        t = np.asarray(rows, dtype=float)[:, 0]
        return np.column_stack([t, 1.0 / t, np.zeros_like(t)])

    X = np.array([[1.0], [1.5], [2.0], [3.0]])
    F = evaluate(X)[:, :2] + 0.25          # every point pushed off the front
    _, Fp = polish_front(X, F, evaluate, [(0.5, 4.0)])
    assert non_dominated(Fp).all()


def test_warm_start_uses_the_previous_polished_point():
    """The continuation from Graham et al. 2022: points are solved in sorted order and
    each starts from its neighbour's answer. Recording the start points is the only way
    to observe it, so polish_front takes an optional callback."""
    from fit_pcsaft._pure.polish import polish_front

    starts = []

    def evaluate(rows):
        t = np.asarray(rows, dtype=float)[:, 0]
        return np.column_stack([t, 1.0 / t, np.zeros_like(t)])

    X = np.array([[3.0], [1.0], [2.0]])           # deliberately unsorted
    F = evaluate(X)[:, :2] + 0.25
    polish_front(X, F, evaluate, [(0.5, 4.0)], on_start=starts.append)
    # first point solved is the one with the smallest F[:, 0], i.e. theta = 1.0
    assert starts[0][0] == pytest.approx(1.0)
    assert len(starts) == 3
