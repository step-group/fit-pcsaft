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
    # Closed form: minimising f1 s.t. f2 <= cap on f2 = 1/f1 lands exactly at
    # f1 = 1/cap, f2 = cap -- a strict check a no-op (or a sign-flipped or
    # non-converging solver) cannot pass merely by not regressing.
    cap = F[:, 1]
    assert Fp[:, 0] == pytest.approx(1.0 / cap, rel=1e-6)
    assert Fp[:, 1] == pytest.approx(cap, rel=1e-6)


def test_polish_with_an_analytic_gradient_reaches_the_same_closed_form():
    """``evaluate_grad`` replaces the finite-difference gradient of both the
    objective and the constraint. On the ``f2 = 1/f1`` front it must land on the
    same closed form as the finite-difference solve, and ``evaluate`` must be
    called exactly once per point -- for the true ``(*objectives, violation)``
    row at the optimum -- because that count is what the speed-up rests on: a
    constraint wired back to ``evaluate`` would silently double the cost."""
    from fit_pcsaft._pure.polish import polish_front

    calls = []

    def evaluate(rows):
        t = np.asarray(rows, dtype=float)[:, 0]
        calls.append(len(t))
        return np.column_stack([t, 1.0 / t, np.zeros_like(t)])

    def evaluate_grad(theta):
        t = float(theta[0])
        return np.array([t, 1.0 / t]), np.array([[1.0], [-1.0 / t**2]])

    X = np.array([[1.0], [2.0]])
    F = evaluate(X)[:, :2] + 0.3
    calls.clear()
    _, Fp = polish_front(X, F, evaluate, [(0.5, 4.0)], evaluate_grad=evaluate_grad)
    cap = F[:, 1]
    assert Fp[:, 0] == pytest.approx(1.0 / cap, rel=1e-6)
    assert Fp[:, 1] == pytest.approx(cap, rel=1e-6)
    assert calls == [1, 1], calls


def test_polish_converges_on_a_badly_scaled_parameter():
    """The solve runs in unit-box coordinates. PC-SAFT's parameters span kappa_ab
    ~1e-2 to epsilon_k_ab ~3e3, and SLSQP starts from an identity Hessian, so a
    parameter of scale 1e6 in raw units moves by ~1e-6 per early step and the
    50-iteration budget runs out before the closed form is reached -- in raw
    units the first point below comes back untouched at f1 = 1.0. Measured on
    thymol's 170-point front, normalising alone took the summed polished
    objective from 356 to 139 at unchanged cost."""
    from fit_pcsaft._pure.polish import polish_front

    scale = 1e6  # theta = scale * f1 on the front f2 = 1/f1, f1 in [0.5, 4]

    def evaluate(rows):
        t = np.asarray(rows, dtype=float)[:, 0] / scale
        return np.column_stack([t, 1.0 / t, np.zeros_like(t)])

    X = np.array([[1.0 * scale], [2.0 * scale]])
    F = evaluate(X)[:, :2] + 0.3
    _, Fp = polish_front(X, F, evaluate, [(0.5 * scale, 4.0 * scale)])
    cap = F[:, 1]
    assert Fp[:, 0] == pytest.approx(1.0 / cap, rel=1e-6)
    assert Fp[:, 1] == pytest.approx(cap, rel=1e-6)


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
    # Non-dominance alone passes for a no-op too (the pushed-off points are
    # still mutually non-dominated). Require each point back on the true
    # front f2 == 1/f1, which a no-op -- sitting at 1/f1 + 0.25 -- fails.
    assert Fp[:, 1] == pytest.approx(1.0 / Fp[:, 0], rel=1e-6)


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


def test_polish_caps_every_other_objective_in_three_dimensions():
    """Three objectives: minimise f1 with f2 AND f3 capped at the input point.
    Front: f = (a, b, 1/(a b)) on [0.5, 4]^2. Minimising a under b <= cap_b and
    1/(a b) <= cap_c lands at b = cap_b, a = 1/(cap_b cap_c) in closed form."""
    from fit_pcsaft._pure.pareto import non_dominated
    from fit_pcsaft._pure.polish import polish_front

    def evaluate(rows):
        t = np.atleast_2d(np.asarray(rows, dtype=float))
        a, b = t[:, 0], t[:, 1]
        return np.column_stack([a, b, 1.0 / (a * b), np.zeros_like(a)])

    X = np.array([[1.0, 1.0], [2.0, 1.0]])
    F = evaluate(X)[:, :3] + 0.25                        # pushed off the front
    _, Fp = polish_front(X, F, evaluate, [(0.5, 4.0), (0.5, 4.0)])
    assert Fp.shape[1] == 3
    assert np.all(Fp <= F + 1e-9)
    assert non_dominated(Fp).all()
    cap_b, cap_c = F[:, 1], F[:, 2]
    assert Fp[:, 0] == pytest.approx(1.0 / (cap_b * cap_c), rel=1e-5)
    assert Fp[:, 1] == pytest.approx(cap_b, rel=1e-5)
