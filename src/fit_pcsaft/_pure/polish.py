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
``bounds``, via ``scipy.optimize.minimize(method="SLSQP")`` in unit-box
coordinates, on the exact gradient where the caller supplies one
(``evaluate_grad``) and a finite-difference one otherwise. A solve that
fails, diverges, or does not
dominate-or-tie the point it started from leaves that row untouched --
polishing 200 points must survive any single SLSQP failure, and must never
make the front worse. The polished set is independently optimized point by
point, which can leave one dominating another, so the result is re-filtered
through ``non_dominated`` before it is returned.

``evaluate`` is the only place a solve costs anything, so it decides the
budget: it is not feos-specific here (the tests below pass a three-line
analytic function and run in milliseconds), but when ``fit_pure_pareto``
wires this in behind ``polish=True``, ``evaluate`` goes through the same
``_map_evaluate`` MOEA/D used. Under ``objectives=("vle", "sft")`` each call
is a DFT solve and there is no ``evaluate_grad``, so a 5-parameter
finite-difference gradient is ~6 evaluations (~0.6 s each, ~3.6 s/gradient)
and a 200-point front polishes in tens of minutes. Under ``("psat", "rho")``
``evaluate_grad`` is one feos AD call per property (``_evaluate_with_grad``,
0.28 ms against 0.25 ms for the plain row) and the cost is SLSQP's line
search. Measured on thymol's production front (170 densified points,
``max_iter=50``, 2026-09-05): the finite-difference solve in raw parameter
units took 20.6 s at ~460 evaluations per point, 65 % of solves hitting
``maxiter``, for a summed polished objective of 356 and an eq-32 tangent of
0.3992; the unit box alone brought the sum to 139 (tangent 0.3956) at
unchanged cost, and exact gradients on top to 103 (0.3955) in 11-12 s at
~250 evaluations per point. Normalisation is the quality half and AD the
speed half -- do not revert one to "simplify" the other. What remains is
the line search (nfev ~5 per iteration even with ``jac``; AARD has a kink at
every data point), so ``max_iter`` is a fidelity knob: 150 iterations cost
what 50 finite-difference ones did and halved the sum again, to 59.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

_TOL = 1e-6  # numerical slack for "dominates or ties" and constraint feasibility


def _solve_one(
    x0: np.ndarray,
    caps: np.ndarray,
    bounds: list[tuple[float, float]],
    axis: int,
    others: list[int],
    evaluate: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
    evaluate_grad: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """One SLSQP solve; returns the true ``(*objectives, violation)`` row at the optimum.

    SLSQP works in unit-box coordinates ``u = (theta - lo) / (hi - lo)``: it
    starts from an identity Hessian, so in raw units a parameter of scale 1e3
    (``epsilon_k_ab``) next to one of scale 1e-2 (``kappa_ab``) crawls, and the
    iteration budget runs out first. ``evaluate`` and the cache only ever see
    the physical theta.

    Objective and constraint share one cache keyed by parameter vector, so a
    theta scipy asks about for both the objective's gradient and the
    constraint's gradient costs one ``evaluate`` call, not two -- see the
    module docstring's cost accounting. With ``evaluate_grad`` the objective,
    its gradient, the constraint and its Jacobian all come out of one cached
    ``evaluate_grad`` call per theta instead, and ``evaluate`` is called once,
    at the optimum, for the true row -- it is where ``violation`` comes from.
    """
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)  # a pinned bound must not divide by zero

    def _theta(u: np.ndarray) -> np.ndarray:
        return lo + np.asarray(u, dtype=float) * span

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

    grads: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _grad(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(np.asarray(theta, dtype=float).tolist())
        got = grads.get(key)
        if got is None:
            f, j = evaluate_grad(np.asarray(theta, dtype=float))
            got = (np.asarray(f, dtype=float), np.asarray(j, dtype=float))
            grads[key] = got
        return got

    def _values(theta: np.ndarray) -> np.ndarray:
        return _grad(theta)[0] if evaluate_grad is not None else _row(theta)[:-1]

    constraint = {"type": "ineq", "fun": lambda u: caps - _values(_theta(u))[others]}
    kw = {}
    if evaluate_grad is not None:  # chain rule for u = (theta - lo) / span
        kw["jac"] = lambda u: _grad(_theta(u))[1][axis] * span
        constraint["jac"] = lambda u: -_grad(_theta(u))[1][others] * span

    result = minimize(
        lambda u: _values(_theta(u))[axis],
        np.clip((np.asarray(x0, dtype=float) - lo) / span, 0.0, 1.0),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(bounds),
        constraints=[constraint],
        # ftol doubles as SLSQP's internal constraint-accuracy target (the
        # Fortran routine's single `acc` parameter serves both), so tightening
        # it is what keeps the capped objective from drifting past `cap` by
        # more than solver noise. Both are in objective units, so the unit
        # box does not retune them.
        options={"maxiter": max_iter, "ftol": 1e-12},
        **kw,
    )
    x_sol = _theta(result.x)
    return x_sol, _row(x_sol)


def polish_front(
    X: np.ndarray,
    F: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    bounds: list[tuple[float, float]],
    *,
    evaluate_grad: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    axis: int = 0,
    n_starts: int = 1,
    max_iter: int = 50,
    on_start: Callable[[np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive each front point onto the true front by epsilon-constraint SLSQP.

    ``X`` is ``(n, n_params)`` physical parameter vectors, ``F`` is the matching
    ``(n, n_obj)`` objective values. ``evaluate`` has the same ``(rows) -> (m, n_obj + 1)``
    contract as ``pareto._map_evaluate`` -- the objectives, then ``violation`` -- so
    ``fit_pure_pareto`` can pass a closure over its worker pool and a test can
    pass a three-line analytic function. ``bounds`` matches ``X``'s columns.
    ``evaluate_grad``, if given, maps ONE physical theta to ``(objectives
    (n_obj,), jacobian (n_obj, n_params))`` and replaces the finite-difference
    gradient of both the objective and the epsilon-constraint; the true
    ``(*objectives, violation)`` row is still read from ``evaluate`` at the
    optimum, so it never has to model the violation.

    ``axis`` picks which objective is minimized; every other objective is
    capped at its value on the input point. Points are processed in order of increasing
    ``F[:, axis]``; each is warm-started from the *previous* point's polished
    parameters when that solve improved the point, else from its own ``X[i]``
    -- the continuation Graham et al. 2022 get from sweeping the front.
    ``n_starts > 1`` adds that many perturbed restarts around the warm start
    and keeps the best feasible result, guarding against the non-convexity
    that made Graham et al. use 2048 Sobol starts; ``on_start`` (if given) is
    called once per point with the resolved warm start, before any restart
    perturbation, so a test can observe the warm-start order.

    A point is replaced only when its polish is feasible (``violation <= 0``,
    every capped objective actually respected) and dominates or ties the
    original row on every objective; anything else -- an exception, a NaN, a
    solve that drifted worse -- leaves the input row untouched. Because each
    point is optimized against its own point's original cap independently,
    one polished point can end up dominating another; the returned set is
    re-filtered with ``non_dominated`` (``pareto.py``) before it is returned,
    in the input row order.
    """
    from fit_pcsaft._pure.pareto import non_dominated

    X = np.atleast_2d(np.asarray(X, dtype=float))
    F = np.atleast_2d(np.asarray(F, dtype=float))
    others = [j for j in range(F.shape[1]) if j != axis]

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    rng = np.random.default_rng(0)

    X_out = X.copy()
    F_out = F.copy()

    order = np.argsort(F[:, axis])
    prev_x = None  # previous point's polished params, only set when it improved

    for i in order:
        caps = F[i, others]
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
                x_sol, row = _solve_one(
                    s, caps, bounds, axis, others, evaluate, max_iter, evaluate_grad
                )
            except Exception:
                continue
            if not np.all(np.isfinite(row)):
                continue
            if row[-1] > 0.0 or np.any(row[others] > caps + _TOL):
                continue
            if best_row is None or row[axis] < best_row[axis]:
                best_x, best_row = x_sol, row

        improved = best_row is not None and np.all(best_row[:-1] <= F[i] + _TOL)
        if improved:
            X_out[i] = best_x
            F_out[i] = best_row[:-1]
            prev_x = best_x
        else:
            prev_x = None

    keep = non_dominated(F_out)
    return X_out[keep], F_out[keep]
