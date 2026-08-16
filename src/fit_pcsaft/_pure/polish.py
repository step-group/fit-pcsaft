"""Deterministic epsilon-constraint polish for a MOEA/D pareto front.

MOEA/D (``pareto.py``) is kept for what it is good at: derivative-free global
coverage of a box where ~80% has no stable interface at all, and (via
``n_restarts``) finding both of water's disconnected parameter basins. What it
does not give is a *smooth* front -- raggedness that depends on the random
seed. Graham, Forte, Burger, Galindo, Jackson & Adjiman (2022), Comput. Chem.
Eng. 167:108015 -- the closest published analogue, by Forte's own co-authors,
tracing SAFT-VR Mie water fronts -- report two ingredients as load-bearing for
a deterministic local solve on top of a global search: every scalarized
subproblem solved multi-start (they use 2048 Sobol points, because the
objective is non-convex with multiple local minima), and each point
warm-started from an already-solved neighbour.

Here the MOEA/D front point supplies that warm start for free -- it is
already near the front, a better guess than a Sobol draw -- so the multi-start
budget can be small (``n_starts`` defaults to 1; raise it only when a polish
visibly stalls). Sweeping in order of increasing ``F[:, axis]`` and
warm-starting each solve from its predecessor's answer is the continuation
Graham et al. get from following the front.

For each point, in that order: minimize ``f_axis(theta)`` subject to
``f_{1-axis}(theta) <= F[i, 1-axis]`` (the epsilon-constraint), inside
``bounds``, via ``scipy.optimize.minimize(method="SLSQP")`` with a
finite-difference gradient. A solve that fails, diverges, or does not
dominate-or-tie the point it started from leaves that row untouched --
polishing 200 points must survive any single SLSQP failure, and must never
make the front worse. The polished set is independently optimized point by
point, which can leave one dominating another, so the result is re-filtered
through ``non_dominated`` before it is returned.

``evaluate`` is the only place a solve costs anything, so it decides the
budget: it is not feos-specific here (the tests below pass a three-line
analytic function and run in milliseconds), but when ``fit_pure_pareto``
wires this in behind ``polish=True``, ``evaluate`` closes over the same
worker pool MOEA/D used and its cost is exactly ``_evaluate_point``'s. Under
``objectives=("vle", "sft")`` that means a DFT solve per call, so a
5-parameter finite-difference gradient is ~6 evaluations (~0.6 s each,
~3.6 s/gradient) and a 200-point front polishes in tens of minutes.
``objectives=("psat", "rho")`` never builds a functional, so the same front
polishes in seconds.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

_TOL = 1e-6  # numerical slack for "dominates or ties" and constraint feasibility


def _solve_one(
    x0: np.ndarray,
    cap: float,
    bounds: list[tuple[float, float]],
    axis: int,
    other: int,
    evaluate: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
) -> np.ndarray:
    """One SLSQP solve; returns the true ``(f1, f2, violation)`` row at the optimum.

    Objective and constraint share one cache keyed by parameter vector, so a
    theta scipy asks about for both the objective's gradient and the
    constraint's gradient costs one ``evaluate`` call, not two -- see the
    module docstring's cost accounting.
    """
    cache: dict[tuple, np.ndarray] = {}

    def _row(theta: np.ndarray) -> np.ndarray:
        key = tuple(np.asarray(theta, dtype=float).tolist())
        row = cache.get(key)
        if row is None:
            row = np.asarray(
                evaluate(np.asarray(theta, dtype=float)[None, :])[0], dtype=float
            )
            cache[key] = row
        return row

    result = minimize(
        lambda theta: _row(theta)[axis],
        np.asarray(x0, dtype=float),
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": lambda theta: cap - _row(theta)[other]}],
        # ftol doubles as SLSQP's internal constraint-accuracy target (the
        # Fortran routine's single `acc` parameter serves both), so tightening
        # it is what keeps the capped objective from drifting past `cap` by
        # more than solver noise.
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    return np.asarray(result.x, dtype=float), _row(result.x)


def polish_front(
    X: np.ndarray,
    F: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    bounds: list[tuple[float, float]],
    *,
    axis: int = 0,
    n_starts: int = 1,
    max_iter: int = 50,
    on_start: Callable[[np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive each front point onto the true front by epsilon-constraint SLSQP.

    ``X`` is ``(n, n_params)`` physical parameter vectors, ``F`` is the matching
    ``(n, 2)`` objective values. ``evaluate`` has the same ``(rows) -> (m, 3)``
    contract as ``pareto._map_evaluate`` -- columns ``f1, f2, violation`` -- so
    ``fit_pure_pareto`` can pass a closure over its worker pool and a test can
    pass a three-line analytic function. ``bounds`` matches ``X``'s columns.

    ``axis`` picks which objective is minimized; the other is capped at its
    value on the input point. Points are processed in order of increasing
    ``F[:, axis]``; each is warm-started from the *previous* point's polished
    parameters when that solve improved the point, else from its own ``X[i]``
    -- the continuation Graham et al. 2022 get from sweeping the front.
    ``n_starts > 1`` adds that many perturbed restarts around the warm start
    and keeps the best feasible result, guarding against the non-convexity
    that made Graham et al. use 2048 Sobol starts; ``on_start`` (if given) is
    called once per point with the resolved warm start, before any restart
    perturbation, so a test can observe the warm-start order.

    A point is replaced only when its polish is feasible (``violation <= 0``,
    the capped objective actually respected) and dominates or ties the
    original row on both objectives; anything else -- an exception, a NaN, a
    solve that drifted worse -- leaves the input row untouched. Because each
    point is optimized against its own point's original cap independently,
    one polished point can end up dominating another; the returned set is
    re-filtered with ``non_dominated`` (``pareto.py``) before it is returned,
    in the input row order.
    """
    from fit_pcsaft._pure.pareto import non_dominated

    X = np.atleast_2d(np.asarray(X, dtype=float))
    F = np.atleast_2d(np.asarray(F, dtype=float))
    other = 1 - axis

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    rng = np.random.default_rng(0)

    X_out = X.copy()
    F_out = F.copy()

    order = np.argsort(F[:, axis])
    prev_x = None  # previous point's polished params, only set when it improved

    for i in order:
        cap = float(F[i, other])
        x0 = np.asarray(prev_x if prev_x is not None else X[i], dtype=float)
        if on_start is not None:
            on_start(x0)

        starts = [x0]
        for _ in range(n_starts - 1):
            jitter = rng.uniform(-0.1, 0.1, size=x0.shape) * (hi - lo)
            starts.append(np.clip(x0 + jitter, lo, hi))

        best_x, best_row = None, None
        for s in starts:
            try:
                x_sol, row = _solve_one(s, cap, bounds, axis, other, evaluate, max_iter)
            except Exception:
                continue
            if not np.all(np.isfinite(row)):
                continue
            if row[2] > 0.0 or row[other] > cap + _TOL:
                continue
            if best_row is None or row[axis] < best_row[axis]:
                best_x, best_row = x_sol, row

        improved = (
            best_row is not None
            and best_row[axis] <= F[i, axis] + _TOL
            and best_row[other] <= F[i, other] + _TOL
        )
        if improved:
            X_out[i] = best_x
            F_out[i] = best_row[:2]
            prev_x = best_x
        else:
            prev_x = None

    keep = non_dominated(F_out)
    return X_out[keep], F_out[keep]
