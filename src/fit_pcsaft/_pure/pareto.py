"""Bi-objective (pareto) PC-SAFT parameter estimation.

Implements Rehner & Gross, J. Chem. Eng. Data 2020, 65, 5698-5707. Bulk-phase
error and interfacial error are treated as two independent objectives instead of
being collapsed into one weighted sum, so the whole trade-off curve is available
and the arbitrariness of a weight choice becomes visible.

    objective 1 (eq 30)  AAD_vle = AARD(psat) + AARD(rho)       [%]
    objective 2 (eq 31)  AAD_sft = mean|gamma_calc - gamma_exp| [mN/m]

Absolute rather than relative deviation for gamma: it goes to zero at the
critical point, where a relative error diverges and would dominate the fit.

The front is generated with pymoo's NSGA-II. A derivative-free population method
is required here: large regions of parameter space have no vapour-liquid
equilibrium or no stable interface at all, and those points are handled by
returning a large objective value rather than a gradient.
"""

from __future__ import annotations

import time
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

_BIG = 1.0e6  # objective value for parameter sets with no VLE / no interface


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


def aad_objectives(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
    data: PureData,
    units: Units,
    sft_options=None,
) -> tuple[float, float]:
    """Return ``(AAD_vle [%], AAD_sft [mN/m])`` for one physical parameter vector.

    ``params_vec`` is in PHYSICAL space, not sqrt-transformed.
    Returns ``(_BIG, _BIG)`` if the EOS, any bulk property, or the interface
    cannot be evaluated at more than half the experimental points.
    """
    from fit_pcsaft.result import _predict_per_property

    try:
        eos = _build_eos(params_vec, compound, spec)
    except Exception:
        return _BIG, _BIG

    preds = _predict_per_property(eos, data, units)
    m_psat = compute_metrics_from_arrays(*preds["psat"])
    m_rho = compute_metrics_from_arrays(*preds["rho"])
    if m_psat.n < 0.5 * len(data.T_psat) or m_rho.n < 0.5 * len(data.T_rho):
        return _BIG, _BIG
    aad_vle = m_psat.aard_pct + m_rho.aard_pct
    if not np.isfinite(aad_vle):
        return _BIG, _BIG

    if len(data.T_sft) == 0:
        return float(aad_vle), 0.0

    try:
        functional = _build_functional(params_vec, compound, spec)
    except Exception:
        return _BIG, _BIG

    gamma = predict_surface_tension(functional, data.T_sft, units, sft_options)
    m_sft = compute_metrics_from_arrays(gamma, data.sft)
    if m_sft.n < 0.5 * len(data.T_sft) or not np.isfinite(m_sft.mae):
        return _BIG, _BIG

    return float(aad_vle), float(m_sft.mae)


def _make_problem(compound, spec, data, units, bounds, sft_options):
    from pymoo.core.problem import ElementwiseProblem

    xl = np.array([b[0] for b in bounds], dtype=float)
    xu = np.array([b[1] for b in bounds], dtype=float)

    class PcSaftBiObjective(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(bounds), n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = list(
                aad_objectives(
                    np.asarray(x, dtype=float), compound, spec, data, units, sft_options
                )
            )

    return PcSaftBiObjective()


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
) -> ParetoResult:
    """Generate the AAD_vle / AAD_sft pareto front with NSGA-II.

    Cost: ``pop_size * n_gen`` objective evaluations, each roughly
    ``5 ms * len(sft data)`` plus the bulk properties. 60x60 with 20 gamma
    points is on the order of 10 minutes single-threaded. Call
    ``feos.set_num_threads(n)`` beforehand to use more cores.

    Returns a ``ParetoResult``; call ``.select(ref_vle, ref_sft)`` on it to get
    a single ``FitResult``.
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

    problem = _make_problem(compound, spec, data, units, bounds, sft_options)
    res = minimize(
        problem,
        NSGA2(pop_size=pop_size),
        ("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
        save_history=False,
    )

    F = np.atleast_2d(np.asarray(res.F, dtype=float))
    X = np.atleast_2d(np.asarray(res.X, dtype=float))
    keep = non_dominated(F) & (F[:, 0] < _BIG)
    if not keep.any():
        raise RuntimeError(
            "NSGA-II found no feasible parameter set: every candidate failed to "
            "produce a vapour-liquid equilibrium or a stable interface. Check the "
            "bounds and the association scheme."
        )
    order = np.argsort(F[keep, 0])

    return ParetoResult(
        X=X[keep][order],
        F=F[keep][order],
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
