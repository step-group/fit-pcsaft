import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Tuple

import feos
import numpy as np
import polars as pl
import pubchempy as pcp
import si_units as si

from fit_pcsaft._csv import load_density_csv, load_hvap_csv, load_psat_csv
from fit_pcsaft._pure.surface_tension import predict_surface_tension
from fit_pcsaft._types import Compound, FitConfig, ModelSpec, PureData, Units

# Sentinel residual returned when the EOS cannot be evaluated at all. Legitimate
# residuals are relative deviations scaled by sqrt(w/n)/f_scale and stay far below it.
_PENALTY = 1e10


def _first_error_hint(errors: "list | None") -> str:
    """Format the first swallowed exception for an aggregate-failure message.

    The per-point handlers must keep swallowing (points legitimately fall outside
    the EOS validity range), so the aggregate failure is the only place that can
    explain itself.
    """
    if not errors:
        return ""
    exc = errors[0]
    return f" First internal failure: {type(exc).__name__}: {exc}"


def _first_error(errors: "list | None"):
    """The exception to chain via ``raise ... from``, or None."""
    return errors[0] if errors else None


# Seconds to wait between the three PubChem lookup attempts. PubChem flaps:
# observed alternating between reachable and HTTP 000 for an hour while the
# rest of the network stayed up. With the lru_cache below there is one lookup
# per compound per process, so this retry is the whole difference between a
# long batch run dying at startup and it not.
_RETRY_BACKOFF = (5.0, 15.0)

_LOOKUP_NAMESPACES = ("name", "smiles", "inchi")


@lru_cache(maxsize=None)
def _fetch_compound(id_str: str) -> Tuple[feos.Identifier, float]:
    """Fetch compound information from PubChem.

    Tries multiple lookup methods:

    1. By common name
    2. By SMILES
    3. By InChI

    A pass over all three that fails with an *exception* is retried up to
    three times with ``_RETRY_BACKOFF``; a pass where every namespace simply
    returned nothing is a genuine "not found" and fails immediately.

    Cached on ``id_str``: every caller that fits/searches the same compound
    more than once (e.g. a MOEA/D sweep re-running ``fit_pure_pareto`` for
    many cells) previously repeated this network round-trip once per call --
    66 identical PubChem lookups for one compound in one observed run,
    which is 66 independent chances for a transient PubChem outage to kill
    the whole thing. ``feos.Identifier`` is immutable (attribute assignment
    raises ``AttributeError`` -- verified directly), so handing every caller
    the same cached instance cannot let one caller's mutation leak into
    another's.
    """
    compound = None
    last_exc: "Exception | None" = None

    for backoff in (*_RETRY_BACKOFF, None):
        for namespace in _LOOKUP_NAMESPACES:
            try:
                compounds = pcp.get_compounds(id_str, namespace)
            except Exception as exc:
                last_exc = exc
                continue
            if compounds:
                compound = compounds[0]
                break
        if compound is not None or last_exc is None or backoff is None:
            break
        time.sleep(backoff)

    if not compound:
        detail = f" ({type(last_exc).__name__}: {last_exc})" if last_exc else ""
        raise ValueError(f"Compound '{id_str}' not found in PubChem{detail}")

    # Extract CAS from synonyms
    cas = None
    syn_exc: "Exception | None" = None
    try:
        synonyms = pcp.get_synonyms(compound.cid, "cid")
        if synonyms:
            for syn_entry in synonyms:
                for synonym in syn_entry.get("Synonym", []):
                    match = re.search(r"(\d{2,7}-\d\d-\d)", synonym)
                    if match:
                        cas = match.group(1)
                        break
                if cas:
                    break
    except Exception as exc:
        syn_exc = exc

    if not cas:
        # Not retried: the lookup above just succeeded, so PubChem was up a
        # moment ago and this is far more likely to be a real gap in the
        # synonym list than a flap. Naming the exception is what keeps an
        # outage from reading as missing data, which is the whole bug.
        detail = f" ({type(syn_exc).__name__}: {syn_exc})" if syn_exc else ""
        raise ValueError(f"Could not extract CAS number for '{id_str}'{detail}")

    # Create feos Identifier
    identifier = feos.Identifier(
        cas=cas,
        name=compound.iupac_name or compound.preferred_iupac_name or id_str,
        iupac_name=compound.iupac_name or compound.preferred_iupac_name or "",
        smiles=compound.smiles or "",
        inchi=compound.inchi or "",
        formula=compound.molecular_formula or "",
    )

    return identifier, compound.molecular_weight



def _is_assoc(spec: ModelSpec) -> bool:
    return spec.na is not None and spec.nb is not None and spec.na > 0 and spec.nb > 0


def param_names(spec: ModelSpec) -> list[str]:
    """The optimiser vector's layout -- the one rule every consumer reads.

    ``[m, sigma, epsilon_k]`` (+ ``mu`` when it is fitted, i.e. ``spec.mu is None``)
    (+ ``kappa_ab, epsilon_k_ab`` when associating).
    """
    names = ["m", "sigma", "epsilon_k"]
    if spec.mu is None:
        names.append("mu")
    if _is_assoc(spec):
        names += ["kappa_ab", "epsilon_k_ab"]
    return names


def params_dict(params_vec, spec: ModelSpec) -> dict[str, float]:
    """``param_names`` zipped with the vector; a length mismatch is a bug, so it raises."""
    return dict(zip(param_names(spec), map(float, params_vec), strict=True))


def ad_model(spec: ModelSpec):
    """The feos AD model whose row layout ``ad_rows`` produces. Neither takes q."""
    if _is_assoc(spec):
        return feos.EquationOfStateAD.PcSaftFull
    return feos.EquationOfStateAD.PcSaftNonAssoc


def ad_rows(X, spec: ModelSpec) -> np.ndarray:
    """``(n, k)`` optimiser vectors -> ``(n, 4|8)`` rows in feos's AD order.

    ``[m, sigma, epsilon_k, mu]`` for ``PcSaftNonAssoc``, plus ``[kappa_ab,
    epsilon_k_ab, na, nb]`` for ``PcSaftFull``. Fixed values (``mu``, ``na``,
    ``nb``) are filled from ``spec``. A single ``(k,)`` vector gives ``(1, ...)``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = len(X)
    cols = [
        X[:, 0],
        X[:, 1],
        X[:, 2],
        X[:, 3] if spec.mu is None else np.full(n, float(spec.mu)),
    ]
    if _is_assoc(spec):
        k = 4 if spec.mu is None else 3
        cols += [X[:, k], X[:, k + 1], np.full(n, float(spec.na)), np.full(n, float(spec.nb))]
    return np.ascontiguousarray(np.column_stack(cols))


def _build_record(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
) -> feos.PureRecord:
    """Build a feos PureRecord from a vector laid out as ``param_names(spec)``."""
    p = params_dict(params_vec, spec)
    kwargs = dict(
        identifier=compound.identifier,
        molarweight=compound.mw,
        m=p["m"],
        sigma=p["sigma"],
        epsilon_k=p["epsilon_k"],
        mu=p["mu"] if spec.mu is None else float(spec.mu),
        q=spec.q,
    )
    if _is_assoc(spec):
        kwargs["association_sites"] = [
            {
                "na": spec.na,
                "nb": spec.nb,
                "epsilon_k_ab": p["epsilon_k_ab"],
                "kappa_ab": p["kappa_ab"],
            }
        ]
    return feos.PureRecord(**kwargs)


def _build_eos(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
) -> feos.EquationOfState:
    """Build the PC-SAFT equation of state. See _build_record for vector ordering."""
    parameters = feos.Parameters.new_pure(_build_record(params_vec, compound, spec))
    return feos.EquationOfState.pcsaft(parameters)


def _build_functional(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
):
    """Build the PC-SAFT Helmholtz energy functional used for DFT surface tension.

    feos 0.10 returns an ``EquationOfState`` here — the same type ``_build_eos``
    returns — but a *different* object: this one carries the non-local weight
    functions ``PlanarInterface`` needs. See _build_record for vector ordering.
    """
    parameters = feos.Parameters.new_pure(_build_record(params_vec, compound, spec))
    return feos.HelmholtzEnergyFunctional.pcsaft(parameters)


def ideal_gas_cp(compound: Compound, dippr107, T_K) -> np.ndarray:
    """Ideal-gas molar cp at ``T_K`` from feos's DIPPR-107 model, in J/(mol K).

    ``dippr107`` are the five Aly-Lee constants in DIPPR units, J/(kmol K), as
    feos takes them. Parameter-independent, so it is evaluated once per fit and
    carried on ``PureData.cp_ig``, never inside an objective.

    Built as its own ideal-gas-only EoS: a bare PC-SAFT EoS has no ideal-gas
    model and *panics* (a ``PanicException``, not an ``Exception``) when asked
    for ``Contributions.IdealGas`` -- hence the ``BaseException`` guard.
    """
    coeffs = [float(c) for c in dippr107] if dippr107 is not None else []
    if len(coeffs) != 5:
        raise ValueError(f"cp_ig takes the five DIPPR-107 coefficients, got {coeffs!r}")
    unit = si.JOULE / (si.MOL * si.KELVIN)
    try:
        record = feos.PureRecord(
            identifier=compound.identifier, molarweight=compound.mw, DIPPR107=coeffs
        )
        eos = feos.EquationOfState.ideal_gas().dippr(feos.Parameters.new_pure(record))
        return np.array([
            float(
                feos.State(eos, temperature=float(t) * si.KELVIN, pressure=si.BAR)
                .molar_isobaric_heat_capacity(feos.Contributions.IdealGas) / unit
            )
            for t in np.asarray(T_K, dtype=float)
        ])
    except BaseException as exc:  # feos panics are BaseException
        raise ValueError(f"cp_ig: feos rejected the DIPPR-107 model: {exc}") from None


def predict_bulk(eos, mw: float, data: PureData, units: Units) -> dict[str, np.ndarray]:
    """psat, rho, hvap and cp at every data temperature, in ``units``.

    One rayon-parallel feos call per property (``feos.Property.*``), so the cost
    is per call, not per point. NaN wherever the VLE solve did not converge
    (supercritical T, or parameters with no VLE); nothing raises per point.
    ``mw`` (g/mol) turns feos's kmol/m3 into kg/m3 -- the EoS does not expose it.

    ``cp`` is the *total* liquid cp: feos's residual at ``(T, P)`` plus the
    tabulated ``data.cp_ig``. A NaN in ``data.P_cp`` means saturation and is
    filled from psat; a row whose psat does not converge is NaN.
    """
    to_K = units.temperature / si.KELVIN
    to_p = 1.0 / (units.pressure / si.PASCAL)
    to_rho = mw / (units.density / (si.KILOGRAM / si.METER**3))
    to_h = 1.0 / (units.enthalpy / (si.JOULE / si.MOL))
    to_cp = 1.0 / (units.heat_capacity / (si.JOULE / (si.MOL * si.KELVIN)))

    def run(fn, T, factor):
        T = np.asarray(T, dtype=float)
        if T.size == 0:
            return np.empty(0)
        vals, ok = fn(eos, np.ascontiguousarray(T[:, None] * to_K))
        return np.where(ok, vals, np.nan) * factor

    def run_cp():
        T = np.asarray(data.T_cp, dtype=float)
        if T.size == 0:
            return np.empty(0)
        P = np.asarray(data.P_cp, dtype=float) / to_p               # user units -> Pa
        if not np.isfinite(P).all():
            psat, ok = feos.Property.vapor_pressure(eos, np.ascontiguousarray(T[:, None] * to_K))
            P = np.where(np.isfinite(P), P, np.where(ok, psat, np.nan))
        bad = ~np.isfinite(P)
        # feos is never handed a NaN: a 1 bar placeholder, masked out below.
        inp = np.ascontiguousarray(np.column_stack([T * to_K, np.where(bad, 1.0e5, P)]))
        vals, ok = feos.Property.residual_isobaric_heat_capacity(eos, inp)
        return np.where(ok & ~bad, vals, np.nan) * to_cp + np.asarray(data.cp_ig, dtype=float)

    return {
        "psat": run(feos.Property.vapor_pressure, data.T_psat, to_p),
        "rho": run(feos.Property.equilibrium_liquid_density, data.T_rho, to_rho),
        "hvap": run(feos.Property.enthalpy_of_vaporization, data.T_hvap, to_h),
        "cp": run_cp(),
    }


_EPS_2PT = np.sqrt(np.finfo(float).eps)  # ~1.49e-8

_VALID_PURE_PROPS = frozenset({"psat", "rho", "hvap", "sft"})


def _normalize_f_scale(
    f_scale,
    loss: str,
    active_properties: set,
) -> dict:
    """Normalize f_scale to a fully-populated dict[str, float] for active properties.

    Rules:
    - None + linear loss   → 1.0 for every active property (identity, no effect)
    - None + robust loss   → raise (user must choose a meaningful margin)
    - float                → broadcast to every active property
    - dict                 → validate keys; fill missing with 1.0 (linear) or raise (robust)
    """
    if f_scale is None:
        if loss != "linear":
            raise ValueError(
                f"loss='{loss}' requires f_scale (e.g. f_scale=0.05 for a 5% "
                "relative-deviation soft margin, or a dict per property)."
            )
        return {p: 1.0 for p in active_properties}

    if isinstance(f_scale, (int, float)):
        val = float(f_scale)
        if val <= 0:
            raise ValueError(f"f_scale must be positive, got {val}")
        return {p: val for p in active_properties}

    if isinstance(f_scale, dict):
        unknown = set(f_scale) - _VALID_PURE_PROPS
        if unknown:
            raise ValueError(
                f"Unknown f_scale key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(_VALID_PURE_PROPS)}."
            )
        for k, v in f_scale.items():
            if v <= 0:
                raise ValueError(f"f_scale['{k}'] must be positive, got {v}")
        missing = active_properties - set(f_scale)
        if missing and loss != "linear":
            raise ValueError(
                f"loss='{loss}' requires f_scale for every active property. "
                f"Missing: {sorted(missing)}."
            )
        return {p: f_scale.get(p, 1.0) for p in active_properties}

    raise TypeError(f"f_scale must be float, dict, or None; got {type(f_scale).__name__}")


def _make_cost_fn(
    data: PureData,
    compound: Compound,
    spec: ModelSpec,
    units: Units,
    config: FitConfig,
    errors: "list | None" = None,
) -> Callable:
    """Create cost function closure for non-associating optimization.

    ``errors`` collects swallowed exceptions so a fit that never leaves its
    initial guess can report *why* instead of claiming convergence. The bulk
    properties come from ``predict_bulk``, which reports non-convergence as
    NaN rather than raising, so ``errors`` only sees a failure to build the
    EoS or to call feos -- never a point that merely did not converge.
    """
    T_psat = data.T_psat
    d_psat = data.p_psat
    T_rho = data.T_rho
    d_rho = data.rho
    T_hvap = data.T_hvap
    d_hvap = data.hvap
    T_sft = data.T_sft
    d_sft = data.sft
    n_psat = len(T_psat)
    n_rho = len(T_rho)
    n_hvap = len(T_hvap)
    n_sft = len(T_sft)
    n_total = n_psat + n_rho + n_hvap + n_sft
    psat_cost_scale = np.sqrt(config.w_psat / n_psat) / config.f_scale["psat"]
    rho_cost_scale = np.sqrt(config.w_rho / n_rho) / config.f_scale["rho"] if n_rho > 0 else 0.0
    hvap_cost_scale = np.sqrt(config.w_hvap / n_hvap) / config.f_scale["hvap"] if n_hvap > 0 else 0.0
    # Surface tension uses an ABSOLUTE deviation normalized by the mean
    # experimental gamma, not a relative one: gamma -> 0 at the critical point,
    # where a relative residual diverges (Rehner & Gross 2020, eq 31).
    sft_cost_scale = (
        np.sqrt(config.w_sft / n_sft) / float(np.mean(d_sft)) / config.f_scale["sft"]
        if n_sft > 0
        else 0.0
    )
    inv_d_psat = 1.0 / d_psat
    inv_d_rho = 1.0 / d_rho if n_rho > 0 else None
    inv_d_hvap = 1.0 / d_hvap if n_hvap > 0 else None
    inv_T_psat = 1.0 / T_psat

    def cost_function(params_vec):
        """Compute weighted relative residuals."""
        try:
            eos = _build_eos(params_vec**2, compound, spec)
            pred = predict_bulk(eos, compound.mw, data, units)
        except Exception as exc:
            if errors is not None:
                errors.append(exc)
            return np.full(n_total, _PENALTY)

        residuals = []

        # Vapor pressure: the only block allowed to extrapolate
        p_pred = pred["psat"]
        success = np.isfinite(p_pred)
        if not success.all():
            if config.extrapolate_psat and success.sum() >= 2:
                # August eqn.: ln(P) = a + b/T — linear regression in log space
                X = np.column_stack([np.ones(success.sum()), inv_T_psat[success]])
                coeffs = np.linalg.lstsq(X, np.log(p_pred[success]), rcond=None)[0]
                p_pred = p_pred.copy()
                p_pred[~success] = np.exp(coeffs[0] + coeffs[1] * inv_T_psat[~success])
            else:
                return np.full(n_total, _PENALTY)

        residuals.append(psat_cost_scale * (p_pred * inv_d_psat - 1.0))

        # Density and enthalpy of vaporization: one failed point voids the property
        if n_rho > 0:
            if not np.isfinite(pred["rho"]).all():
                return np.full(n_total, _PENALTY)
            residuals.append(rho_cost_scale * (pred["rho"] * inv_d_rho - 1.0))

        if n_hvap > 0:
            if not np.isfinite(pred["hvap"]).all():
                return np.full(n_total, _PENALTY)
            residuals.append(hvap_cost_scale * (pred["hvap"] * inv_d_hvap - 1.0))

        # Surface tension residuals (absolute, normalized by mean gamma — a
        # relative residual diverges as gamma -> 0 at the critical point)
        if n_sft > 0:
            try:
                functional = _build_functional(params_vec**2, compound, spec)
            except Exception as exc:
                if errors is not None:
                    errors.append(exc)
                return np.full(n_total, _PENALTY)

            sft_pred = predict_surface_tension(
                functional, T_sft, units, config.sft_options
            )
            if not np.isfinite(sft_pred).all():
                return np.full(n_total, _PENALTY)

            residuals.append(sft_cost_scale * (sft_pred - d_sft))

        return np.concatenate(residuals)

    return cost_function


def _make_f_and_df_numerical(
    data: PureData,
    compound: Compound,
    spec: ModelSpec,
    units: Units,
    config: FitConfig,
) -> Tuple[Callable, Callable, list]:
    """Create cost function + 2-point numerical Jacobian with shared base-eval cache.

    Scipy calls f(x) then jac(x) at the same x each iteration. Caching the last
    (x, f(x)) means the Jacobian reuses the base evaluation instead of rerunning feos.

    Returns ``(f, df, errors)``; ``errors`` accumulates swallowed exceptions.
    """
    errors: list = []
    _cost = _make_cost_fn(data, compound, spec, units, config, errors)

    # Use standard variables instead of a list hack
    x_cached = None
    f_cached = None

    def f(x: np.ndarray) -> np.ndarray:
        nonlocal x_cached, f_cached
        fx = _cost(x)
        x_cached = x.copy()
        f_cached = fx
        return fx

    def df(x: np.ndarray) -> np.ndarray:
        nonlocal x_cached, f_cached

        if x_cached is not None and np.array_equal(x, x_cached):
            f0 = f_cached
        else:
            f0 = _cost(x)
            x_cached = x.copy()
            f_cached = f0

        n = len(x)
        J = np.empty((len(f0), n))

        for i in range(n):
            # Calculate a relative step size, falling back to absolute for values near 0
            h = _EPS_2PT * max(abs(x[i]), 1.0)

            x_pert = x.copy()
            x_pert[i] += h
            J[:, i] = (_cost(x_pert) - f0) / h

        return J

    return f, df, errors
