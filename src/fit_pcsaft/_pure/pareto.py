"""Bi-objective (pareto) PC-SAFT parameter estimation.

Implements Rehner & Gross, J. Chem. Eng. Data 2020, 65, 5698-5707. Bulk-phase
error and interfacial error are treated as two independent objectives instead of
being collapsed into one weighted sum, so the whole trade-off curve is available
and the arbitrariness of a weight choice becomes visible.

    objective 1 (eq 30)  AAD_vle = mean(AARD(psat), AARD(rho))  [%]
    objective 2 (eq 31)  AAD_sft = mean|gamma_calc - gamma_exp| [mN/m]

Whether eq 30 averages the two AARDs or sums them is not legible in the
published equation. It averages: the paper's own water PC-SAFT 2B parameters
(Table 1) reproduce their AAD_vle = 2.14% (Table 2) only under the mean —
they give 2.4% as a mean and 4.7% as a sum over a comparable temperature
range.

Absolute rather than relative deviation for gamma: it goes to zero at the
critical point, where a relative error diverges and would dominate the fit.

The front is generated with pymoo's NSGA-II. A derivative-free population method
is required here: large regions of parameter space have no vapour-liquid
equilibrium or no stable interface at all, and those points are handled by
returning a large objective value rather than a gradient.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import si_units as si

from fit_pcsaft._fit_utils import _build_eos, _build_functional
from fit_pcsaft._metrics import compute_metrics_from_arrays
from fit_pcsaft._pure.surface_tension import predict_surface_tension
from fit_pcsaft._types import Compound, ModelSpec, PureData, Units

_BIG = 1.0e6  # objective value where nothing at all could be evaluated

# A dataset counts as evaluable when the model produced a finite value at more
# than this fraction of its experimental points.
_MIN_VALID_FRACTION = 0.5


def non_dominated(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of a minimization objective array.

    A row is dominated when another row is <= in every objective and < in at
    least one. Exact duplicates: the first occurrence is kept.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
    # Drop exact duplicates, keeping the first.
    seen: set = set()
    for i in range(n):
        if not keep[i]:
            continue
        key = tuple(F[i])
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    return keep


def _argmin_scalarized(F: np.ndarray, ref_vle: float, ref_sft: float) -> int:
    """Index of the point minimising eq 32: AAD_vle/ref_vle + AAD_sft/ref_sft.

    Geometrically this is the tangent point of the front with a line of slope
    -ref_sft/ref_vle, which is how the paper picks its published parameters.
    """
    F = np.asarray(F, dtype=float)
    return int(np.argmin(F[:, 0] / ref_vle + F[:, 1] / ref_sft))


def _penalize(R: np.ndarray, ref_vle: float, ref_sft: float) -> np.ndarray:
    """Scaled objectives with the feasibility violation folded in.

    pymoo's MOEA/D refuses a constrained problem outright (``moead.py``: its
    ``_setup`` asserts ``not problem.has_constraints()``), so the graded
    violation ``_evaluate_point`` reports cannot be handed over as ``out["G"]``.
    It is added to both objectives instead. Feasible rows (``violation <= 0``)
    are untouched, so the front itself is unaffected; infeasible ones stay
    *rankable among themselves*, which is the whole point of grading them —
    about 80% of a wide bounds box has no stable interface, and an optimizer
    that cannot tell "failed two of twenty-five gamma points" from "no VLE at
    all" has no direction to climb out.

    The division by ``(ref_vle, ref_sft)`` is not cosmetic. MOEA/D decomposes
    with Tchebicheff, which pymoo applies to raw objective values — it is handed
    an ideal point but never a nadir point, so nothing normalizes the two axes.
    On water AAD_vle spans some 9 units above the ideal against AAD_sft's 1.3,
    so weight vectors only start separating points near w1/w2 = 0.14 and most of
    a uniform fan collapses onto one corner of the front. Eq 32 already defines
    what commensurate means for this pair; this reuses it.

    A degenerate row can carry ``_BIG`` while still reporting ``violation <= 0``
    (an infinite AARD over points that all evaluated). Such a row can outrank a
    mildly infeasible one here. That is deliberate: both are dropped by the
    feasibility filter in ``_front_from``, and pushing the search away from a
    ``_BIG`` point is wanted either way.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    F = R[:, :2] / np.array([ref_vle, ref_sft], dtype=float)
    return F + (_BIG * np.maximum(R[:, 2], 0.0))[:, None]


def _evaluate_point(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
    data: PureData,
    units: Units,
    sft_options=None,
) -> tuple[float, float, float]:
    """Return ``(AAD_vle [%], AAD_sft [mN/m], violation)`` for a parameter vector.

    ``params_vec`` is in PHYSICAL space, not sqrt-transformed.

    ``violation <= 0`` means every dataset was evaluable at more than
    ``_MIN_VALID_FRACTION`` of its points. Above zero it measures how far short
    the *worst* dataset fell, and the objectives are still computed from
    whatever points did work.

    That grading matters: roughly 80% of a wide parameter box has no stable
    interface for an associating fluid, and a single flat penalty maps all of it
    onto one indistinguishable point. The optimizer then cannot tell "failed two
    of twenty-five gamma points" from "no vapour-liquid equilibrium at all", and
    has no direction to climb out. Reporting the degree of failure gives it one.
    """
    from fit_pcsaft.result import _predict_per_property

    worst = _BIG_VIOLATION = 1.0
    try:
        eos = _build_eos(params_vec, compound, spec)
    except Exception:
        return _BIG, _BIG, _BIG_VIOLATION

    def _fraction(metrics, n_total):
        return 1.0 if n_total == 0 else metrics.n / n_total

    preds = _predict_per_property(eos, data, units)
    m_psat = compute_metrics_from_arrays(*preds["psat"])
    m_rho = compute_metrics_from_arrays(*preds["rho"])
    worst = min(
        _fraction(m_psat, len(data.T_psat)), _fraction(m_rho, len(data.T_rho))
    )

    aad_vle = 0.5 * (m_psat.aard_pct + m_rho.aard_pct)
    if not np.isfinite(aad_vle):
        aad_vle = _BIG

    if len(data.T_sft) == 0:
        return float(aad_vle), 0.0, _MIN_VALID_FRACTION - worst

    try:
        functional = _build_functional(params_vec, compound, spec)
    except Exception:
        return float(aad_vle), _BIG, _BIG_VIOLATION

    gamma = predict_surface_tension(functional, data.T_sft, units, sft_options)
    m_sft = compute_metrics_from_arrays(gamma, data.sft)
    worst = min(worst, _fraction(m_sft, len(data.T_sft)))
    aad_sft = m_sft.mae if np.isfinite(m_sft.mae) else _BIG

    return float(aad_vle), float(aad_sft), _MIN_VALID_FRACTION - worst


def aad_objectives(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
    data: PureData,
    units: Units,
    sft_options=None,
) -> tuple[float, float]:
    """Return ``(AAD_vle [%], AAD_sft [mN/m])`` for one physical parameter vector.

    ``params_vec`` is in PHYSICAL space, not sqrt-transformed. Returns
    ``(_BIG, _BIG)`` when the parameter set is infeasible — see
    ``_evaluate_point``, which the optimizer uses directly because it also
    reports *how* infeasible.
    """
    f1, f2, g = _evaluate_point(params_vec, compound, spec, data, units, sft_options)
    return (_BIG, _BIG) if g > 0.0 else (f1, f2)


@contextmanager
def _silence_fd_stderr(active: bool = True):
    """Redirect file descriptor 2 to /dev/null.

    feos panics inside Rust worker threads when the DFT solver fails to
    converge ("IterationFailed(...)"). Those panics are already handled — the
    objective returns _BIG — but Rust writes the panic message straight to fd 2,
    where no Python-level redirect can reach it. Over thousands of NSGA-II
    evaluations that is thousands of lines of noise about failures we expect.

    This suppresses the whole fd, so genuine stderr output from the wrapped
    block is lost too. Pass active=False to keep it when debugging.
    """
    if not active:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


# --------------------------------------------------------------------------
# Parallel evaluation
#
# feos does not release the GIL — measured speedup with a 16-thread pool is
# 0.89x, i.e. slower than serial — so threads are useless and worker *processes*
# are required. feos.Identifier cannot be pickled either, so nothing feos-shaped
# may cross a process boundary: each worker rebuilds its own from plain strings
# once, in an initializer, and thereafter only numpy rows are sent.
# --------------------------------------------------------------------------

_IDENT_FIELDS = ("cas", "name", "iupac_name", "smiles", "inchi", "formula")
_WORKER: dict = {}


def _identifier_fields(identifier) -> dict:
    return {f: getattr(identifier, f, None) for f in _IDENT_FIELDS}


def _worker_init(ident_fields, mw, spec, data, units, sft_options, quiet):
    import feos

    # Each worker owns one core; letting every one of them spin up its own rayon
    # pool oversubscribes the machine badly (feos defaults to 52 threads here).
    try:
        feos.set_num_threads(1)
    except Exception:
        pass

    if quiet:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)  # Rust panics go to fd 2; see _silence_fd_stderr

    _WORKER.update(
        compound=Compound(identifier=feos.Identifier(**ident_fields), mw=mw),
        spec=spec, data=data, units=units, sft_options=sft_options,
    )


def _worker_evaluate(x):
    return _evaluate_point(
        np.asarray(x, dtype=float), _WORKER["compound"], _WORKER["spec"],
        _WORKER["data"], _WORKER["units"], _WORKER["sft_options"],
    )


def _resolve_n_jobs(n_jobs: int) -> int:
    """-1 means every core bar two, so the machine stays usable during a run."""
    if n_jobs is not None and n_jobs >= 1:
        return int(n_jobs)
    cpus = os.process_cpu_count() or os.cpu_count() or 1
    return max(1, cpus - 2)


@contextmanager
def _worker_pool(n_jobs, compound, spec, data, units, sft_options, quiet):
    """A spawn-based pool, or None when running serially.

    "spawn", not "fork": feos has already started its rayon threads in the
    parent by this point, and forking a process with live threads is a
    well-known way to deadlock in the child.
    """
    if n_jobs <= 1:
        yield None
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        n_jobs,
        initializer=_worker_init,
        initargs=(_identifier_fields(compound.identifier), compound.mw,
                  spec, data, units, sft_options, quiet),
    )
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()


def _make_problem(compound, spec, data, units, bounds, sft_options, pool=None):
    """Population-at-a-time problem.

    Deliberately a vectorized ``Problem`` rather than pymoo's
    ``ElementwiseProblem`` + ``StarmapParallelization``: that route pickles the
    problem itself to each worker, which fails here because the problem closes
    over a ``feos.Identifier``. Taking the whole population and mapping it
    ourselves keeps every feos object in the process that made it.
    """
    from pymoo.core.problem import Problem

    xl = np.array([b[0] for b in bounds], dtype=float)
    xu = np.array([b[1] for b in bounds], dtype=float)

    class PcSaftBiObjective(Problem):
        def __init__(self):
            super().__init__(n_var=len(bounds), n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            rows = [np.asarray(x, dtype=float) for x in np.atleast_2d(X)]
            if pool is None:
                results = [
                    _evaluate_point(x, compound, spec, data, units, sft_options)
                    for x in rows
                ]
            else:
                results = pool.map(_worker_evaluate, rows)
            R = np.asarray(results, dtype=float)
            out["F"] = R[:, :2]
            out["G"] = R[:, 2:3]

    return PcSaftBiObjective()


def _map_evaluate(rows, pool, compound, spec, data, units, sft_options):
    """Evaluate parameter vectors, through the worker pool when there is one."""
    if pool is None:
        return [
            _evaluate_point(x, compound, spec, data, units, sft_options)
            for x in rows
        ]
    return pool.map(_worker_evaluate, rows)


def _densify(X, F, n_between, pool, compound, spec, data, units, sft_options):
    """Fill the gaps NSGA-II leaves between adjacent front points.

    NSGA-II returns a population, not a curve. Crowding distance preserves
    diversity but nothing forces even spacing, so the front comes back as
    clusters of near-identical parameter sets separated by voids — on water,
    eight of eighty points sat within 0.2% of each other on the AAD_vle axis
    while a 2.8% stretch of the same axis held none.

    The voids are a sampling artefact, not structure: every point obtained by
    linearly interpolating the parameter vectors across them is itself
    non-dominated. So interpolating and re-evaluating buys resolution at the
    price of arithmetic — no new search, just more points on a curve already
    found. Costs ``(len(X) - 1) * n_between`` evaluations.

    Interpolating in parameter space rather than objective space is what makes
    this sound: the result is a real parameter set with real objective values,
    never a drawn-in line between two computed points.

    The ``n_between`` interpolants per segment are a budget, not a fixed count:
    they are handed out in proportion to each segment's length in normalized
    objective space. Spreading them evenly instead wastes most of them inside
    the clusters, which are already dense, and leaves the voids — the whole
    point of the exercise — barely touched.

    Costs about ``(len(X) - 1) * n_between`` evaluations.
    """
    if len(X) < 2 or n_between < 1:
        return X, F

    span = np.ptp(F, axis=0)
    span[span <= 0.0] = 1.0
    seg = np.linalg.norm(np.diff(F, axis=0) / span, axis=1)
    budget = n_between * (len(X) - 1)
    share = seg / seg.sum() if seg.sum() > 0 else np.full(len(seg), 1.0 / len(seg))
    # At least one interpolant per segment, so no gap is skipped entirely.
    counts = np.maximum(1, np.round(budget * share).astype(int))

    rows = [
        (1.0 - t) * X[k] + t * X[k + 1]
        for k, n_k in enumerate(counts)
        for t in np.linspace(0.0, 1.0, n_k + 2)[1:-1]
    ]
    if not rows:
        return X, F
    R = np.asarray(
        _map_evaluate(rows, pool, compound, spec, data, units, sft_options),
        dtype=float,
    )
    ok = (R[:, 2] <= 0.0) & (R[:, 0] < _BIG)
    if not ok.any():
        return X, F
    X_all = np.vstack([X, np.asarray(rows, dtype=float)[ok]])
    F_all = np.vstack([F, R[ok, :2]])
    keep = non_dominated(F_all)
    order = np.argsort(F_all[keep, 0])
    return X_all[keep][order], F_all[keep][order]


def _initial_sampling(use_lhs: bool):
    """Sampling operator for generation zero.

    Latin hypercube rather than uniform random: only about 20% of a wide
    parameter box is feasible for an associating fluid once the interface has to
    converge too, so how evenly generation zero covers the box decides how much
    of the budget is spent merely finding feasible ground.

    Deliberately *not* seeded from ``fit_pure``'s curated multi-start sets. Those
    were chosen for alcohols — sigma 3.1 to 3.9 — and for water, whose optimum is
    near sigma 2.33, they are feasible but wrong: they survive selection and breed
    the population away from the answer. Measured on water, seeding with them
    moved the best AAD_vle from 4.8% to 11.2% at equal budget.
    """
    if use_lhs:
        from pymoo.operators.sampling.lhs import LatinHypercubeSampling
        return LatinHypercubeSampling()
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    return FloatRandomSampling()


def _ref_dirs(pop_size: int) -> np.ndarray:
    """The weight-vector fan MOEA/D decomposes against, one per population slot.

    Das-Dennis on two objectives returns ``n_partitions + 1`` vectors, so
    ``pop_size - 1`` partitions keeps ``pop_size`` meaning what it meant under
    NSGA-II. pymoo needs these up front: ``MOEAD.__init__`` reads
    ``len(ref_dirs)`` to set the population size before its own ``None``
    fallback in ``_setup`` ever runs.
    """
    from pymoo.util.ref_dirs import get_reference_directions

    if pop_size < 2:
        raise ValueError(f"MOEA/D needs at least 2 weight vectors, got {pop_size}")
    return get_reference_directions("uniform", 2, n_partitions=pop_size - 1)


def _make_algorithm(pop_size: int, lhs: bool):
    """MOEA/D, in the variant that evaluates a whole population at a time.

    ``ParallelMOEAD``, not ``MOEAD``: the plain class is a ``LoopwiseAlgorithm``
    whose ``_next`` yields a single offspring per step, which would hand the
    worker pool batches of one and give back all of the 4.3x that pool buys.
    ``ParallelMOEAD`` overrides ``_infill``/``_advance`` to mate the whole
    population first, and inherits ``n_offsprings = pop_size``, so one
    generation is one ``pool.map`` — the same rhythm NSGA-II had.
    """
    from pymoo.algorithms.moo.moead import ParallelMOEAD

    return ParallelMOEAD(_ref_dirs(pop_size), sampling=_initial_sampling(lhs))


@dataclass(frozen=True)
class ParetoResult:
    """The pareto front of a bi-objective PC-SAFT fit.

    ``F[:, 0]`` is AAD_vle in %, ``F[:, 1]`` is AAD_sft in the surface tension
    input unit (mN/m by default). ``X[i]`` is the physical parameter vector that
    produced ``F[i]``, in the standard ordering
    ``[m, sigma, epsilon_k] (+ [mu]) (+ [kappa_ab, epsilon_k_ab])``.
    Rows are sorted by increasing AAD_vle.
    """

    X: np.ndarray
    F: np.ndarray
    data: PureData
    compound: Compound
    spec: ModelSpec
    units: Units
    fit_mu: bool
    is_associative: bool
    time_elapsed: float
    input_name: str = ""
    algorithm_result: object = None

    @property
    def param_names(self) -> list[str]:
        names = ["m", "sigma", "epsilon_k"]
        if self.fit_mu:
            names.append("mu")
        if self.is_associative:
            names += ["kappa_ab", "epsilon_k_ab"]
        return names

    def select(self, ref_vle: float = 2.0, ref_sft: float = 0.7):
        """Pick the point on the front that minimises eq 32, as a FitResult.

        Paper defaults: water ref_vle=2%, ref_sft=0.7 mN/m. Small alcohols use
        ref_sft=1.5; alcohols from 1-pentanol up use ref_sft=3.0.
        """
        from fit_pcsaft._pure.fit import _extract_params_dict
        from fit_pcsaft.result import FitResult, _compute_pure_metrics

        i = _argmin_scalarized(self.F, ref_vle, ref_sft)
        params_vec = self.X[i]
        eos = _build_eos(params_vec, self.compound, self.spec)
        functional = (
            _build_functional(params_vec, self.compound, self.spec)
            if len(self.data.T_sft) > 0
            else None
        )
        params = _extract_params_dict(
            params_vec, self.spec.mu, assoc=self.is_associative
        )
        metrics = _compute_pure_metrics(
            eos, self.data, self.units, functional=functional
        )
        cost = float(self.F[i, 0] / ref_vle + self.F[i, 1] / ref_sft)
        scipy_result = SimpleNamespace(
            # FitResult.__str__ derives RMS from 2*cost/len(fun)
            cost=0.5 * cost**2,
            fun=np.array([cost]),
            x=params_vec,
            success=True,
            message=f"NSGA-II front point {i} (ref_vle={ref_vle}, ref_sft={ref_sft})",
            nfev=len(self.F),
        )
        return FitResult(
            params=params,
            eos=eos,
            data=self.data,
            compound=self.compound,
            spec=self.spec,
            units=self.units,
            metrics=metrics,
            scipy_result=scipy_result,
            time_elapsed=self.time_elapsed,
            input_name=self.input_name,
            functional=functional,
        )

    def plot(self, path=None, ref_vle: float = 2.0, ref_sft: float = 0.7):
        from fit_pcsaft._plot import _plot_pareto
        return _plot_pareto(self, path=path, ref_vle=ref_vle, ref_sft=ref_sft)

    def to_csv(self, path: "Path | str") -> None:
        """Write the front: one row per point, objectives then parameters."""
        import polars as pl

        df = pl.DataFrame(
            {"aad_vle_pct": self.F[:, 0], "aad_sft": self.F[:, 1]}
            | {n: self.X[:, k] for k, n in enumerate(self.param_names)}
        )
        df.write_csv(str(path))

    def __str__(self) -> str:
        i_v = int(np.argmin(self.F[:, 0]))
        i_s = int(np.argmin(self.F[:, 1]))
        return (
            f"Pareto front — {self.input_name}\n"
            f"  points: {len(self.F)}   time: {self.time_elapsed:.1f} s\n"
            f"  best AAD_vle: {self.F[i_v, 0]:.2f}% "
            f"(AAD_sft there: {self.F[i_v, 1]:.2f})\n"
            f"  best AAD_sft: {self.F[i_s, 1]:.2f} "
            f"(AAD_vle there: {self.F[i_s, 0]:.2f}%)"
        )


def fit_pure_pareto(
    id: str,
    psat_path: "Path | str",
    density_path: "Path | str",
    sft_path: "Path | str",
    hvap_path: "Optional[Path | str]" = None,
    mu: "Optional[float]" = 0.0,
    q: float = 0.0,
    na: "Optional[int]" = None,
    nb: "Optional[int]" = None,
    bounds: "Optional[list]" = None,
    pop_size: int = 60,
    n_gen: int = 60,
    seed: int = 1,
    sft_options=None,
    pressure_unit: "si.SIObject" = si.KILO * si.PASCAL,
    temperature_unit: "si.SIObject" = si.KELVIN,
    density_unit: "si.SIObject" = si.KILOGRAM / (si.METER**3),
    enthalpy_unit: "si.SIObject" = si.KILO * si.JOULE / si.MOL,
    surface_tension_unit: "si.SIObject" = si.MILLI * si.NEWTON / si.METER,
    verbose: bool = True,
    quiet_solver: bool = True,
    lhs: bool = True,
    n_jobs: int = -1,
    refine: int = 4,
) -> ParetoResult:
    """Generate the AAD_vle / AAD_sft pareto front with NSGA-II.

    Cost is ``pop_size * n_gen`` objective evaluations, each roughly 120 ms at
    sensible parameters and up to 1 s where the VLE solve fails. Budget decides
    front quality far more than anything else; measured on water (2B, 14
    workers), best-of-front and the largest hole in it:

        evals   points   AAD_vle   AAD_sft   max gap   time
         1200       32      9.47      1.12      4.37     56s
         3600       47      2.27      0.70     19.22    127s
         9600       80      1.44      0.78      2.82    301s

    Below a few thousand evaluations the search is still finding feasible
    ground rather than resolving the trade-off, and the front is a cluster
    rather than a curve. The defaults here are a compromise; raise n_gen when
    the front looks patchy. Note the m-range differed between the 3600 and
    9600 runs, so even 9600 is not fully converged -- treat a single run as
    one sample, not the answer.

    The "max gap" column above is the raw NSGA-II output; ``refine`` (below)
    closes most of it afterwards for a few percent of the same budget.

    Returns a ``ParetoResult``; call ``.select(ref_vle, ref_sft)`` on it to get
    a single ``FitResult``.

    ``quiet_solver`` suppresses fd-level stderr for the duration of the search,
    hiding the Rust panic messages feos emits from worker threads when the DFT
    solver fails on an infeasible parameter set. Pass False when debugging.

    ``lhs`` uses Latin hypercube sampling for generation zero instead of uniform
    random draws, which covers the box more evenly. Pass False to compare.

    ``n_jobs`` evaluates the population across worker processes; -1 (the
    default) uses every core bar two. Processes rather than threads because feos
    does not release the GIL. On a spawn platform, call this from inside an
    ``if __name__ == "__main__":`` guard.

    ``refine`` interpolates roughly this many parameter sets between each
    adjacent pair of front points once the search is done, keeping the ones that
    turn out to be non-dominated. Measured on the 9600-evaluation water run, at
    3% of the search cost:

        80 -> 209 points, median objective-space spacing 3.4x finer, 316
        evaluations, 7 s on 14 workers.

    It does two things. The obvious one is resolution: NSGA-II's crowding
    distance keeps diversity but never forces even spacing, so its output is
    clusters separated by voids, and every point interpolated across a void is
    itself non-dominated -- the voids are a sampling artefact. The less obvious
    one is correctness: the raw front is not always a front. On water the fill
    dominated a whole stretch of it, moving the eq-32 tangent point from
    (1.44%, 1.62) to (1.90%, 1.29) -- eq 32 from 3.04 to 2.79. Trust a raw
    front's selected point less than a refined one's.

    What it does not fix is a genuine hole, where the straight line between two
    front points in parameter space does not track the front in objective space.
    Water keeps one, at the knee near AAD_vle = 2%: a second pass moves eq 32 by
    0.01 for 3.6x the evaluations. That corner needs search budget, not
    arithmetic. Set refine=0 to see the raw population. See ``_densify``.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    from fit_pcsaft._pure.fit import _default_de_bounds, _setup_pure_fit

    t0 = time.perf_counter()
    (data, compound, spec, units, _config,
     _cost_fn, _jac_fn, _errors, fit_mu, is_associative) = _setup_pure_fit(
        id=id,
        psat_path=psat_path,
        density_path=density_path,
        hvap_path=hvap_path,
        sft_path=sft_path,
        mu=mu, q=q, na=na, nb=nb,
        psat_weight=3.0, density_weight=2.0, hvap_weight=1.0, sft_weight=1.0,
        extrapolate_psat=False,
        pressure_unit=pressure_unit,
        temperature_unit=temperature_unit,
        density_unit=density_unit,
        enthalpy_unit=enthalpy_unit,
        surface_tension_unit=surface_tension_unit,
        sft_options=sft_options,
        verbose=False,
    )

    if bounds is None:
        bounds = _default_de_bounds(fit_mu, is_associative)

    algorithm = NSGA2(pop_size=pop_size, sampling=_initial_sampling(lhs))
    n_workers = _resolve_n_jobs(n_jobs)
    if verbose:
        print(f"NSGA-II: pop {pop_size} x {n_gen} gen on {n_workers} process(es)")

    with _worker_pool(
        n_workers, compound, spec, data, units, sft_options, quiet_solver
    ) as pool:
        problem = _make_problem(
            compound, spec, data, units, bounds, sft_options, pool=pool
        )
        with _silence_fd_stderr(quiet_solver):
            res = minimize(
                problem,
                algorithm,
                ("n_gen", n_gen),
                seed=seed,
                verbose=verbose,
                save_history=False,
            )

        if res.F is None:
            raise RuntimeError(
                "NSGA-II found no feasible parameter set: every candidate failed "
                "to produce a vapour-liquid equilibrium or a stable interface. "
                "Check the bounds and the association scheme."
            )
        F = np.atleast_2d(np.asarray(res.F, dtype=float))
        X = np.atleast_2d(np.asarray(res.X, dtype=float))
        keep = non_dominated(F) & (F[:, 0] < _BIG)
        if not keep.any():
            raise RuntimeError(
                "NSGA-II returned only degenerate solutions. Check the bounds and "
                "the association scheme."
            )
        order = np.argsort(F[keep, 0])
        X, F = X[keep][order], F[keep][order]

        if refine > 0 and len(X) > 1:
            n_before = len(F)
            with _silence_fd_stderr(quiet_solver):
                X, F = _densify(
                    X, F, refine, pool, compound, spec, data, units, sft_options
                )
            if verbose:
                print(f"refine: {n_before} -> {len(F)} front points")

    return ParetoResult(
        X=X,
        F=F,
        data=data,
        compound=compound,
        spec=spec,
        units=units,
        fit_mu=fit_mu,
        is_associative=is_associative,
        time_elapsed=time.perf_counter() - t0,
        input_name=id,
        algorithm_result=res,
    )
