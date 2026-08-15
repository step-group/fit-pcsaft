import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Tuple

import feos
import numpy as np
import polars as pl
import pubchempy as pcp

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



def _build_record(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
) -> feos.PureRecord:
    """
    Build a feos PureRecord from a parameter vector.

    Non-associating (na/nb are None):
        mu fixed: params_vec = [m, sigma, epsilon_k]
        mu fitted: params_vec = [m, sigma, epsilon_k, mu]

    Associating (na/nb are integers):
        mu fixed: params_vec = [m, sigma, epsilon_k, kappa_ab, epsilon_k_ab]
        mu fitted: params_vec = [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab]
    """
    identifier = compound.identifier
    mw = compound.mw
    mu = spec.mu
    q = spec.q
    na = spec.na
    nb = spec.nb

    idx = 0
    m = float(params_vec[idx])
    idx += 1
    sigma = float(params_vec[idx])
    idx += 1
    epsilon_k = float(params_vec[idx])
    idx += 1

    if mu is None:
        mu_val = float(params_vec[idx])
        idx += 1
    else:
        mu_val = float(mu)

    assoc = na is not None and nb is not None and na > 0 and nb > 0
    if assoc:
        kappa_ab = float(params_vec[idx])
        idx += 1
        epsilon_k_ab = float(params_vec[idx])
        idx += 1
        return feos.PureRecord(
            identifier=identifier,
            molarweight=mw,
            m=m,
            sigma=sigma,
            epsilon_k=epsilon_k,
            mu=mu_val,
            q=q,
            association_sites=[
                {"na": na, "nb": nb, "epsilon_k_ab": epsilon_k_ab, "kappa_ab": kappa_ab}
            ],
        )
    return feos.PureRecord(
        identifier=identifier,
        molarweight=mw,
        m=m,
        sigma=sigma,
        epsilon_k=epsilon_k,
        mu=mu_val,
        q=q,
    )


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
    initial guess can report *why* instead of claiming convergence.
    """
    T_psat = data.T_psat
    d_psat = data.p_psat
    T_rho = data.T_rho
    d_rho = data.rho
    T_hvap = data.T_hvap
    d_hvap = data.hvap
    T_sft = data.T_sft
    d_sft = data.sft
    temperature_unit = units.temperature
    psat_unit = units.pressure
    rho_unit = units.density
    enthalpy_unit = units.enthalpy

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
        except Exception as exc:
            if errors is not None:
                errors.append(exc)
            return np.full(n_total, _PENALTY)

        residuals = []

        # Vapor pressure residuals
        p_pred = np.empty(n_psat)
        success = np.ones(n_psat, dtype=bool)
        for i, T in enumerate(T_psat):
            try:
                p_pred[i] = (
                    feos.PhaseEquilibrium.vapor_pressure(eos, T * temperature_unit)[0]
                    / psat_unit
                )
            except Exception as exc:
                if errors is not None:
                    errors.append(exc)
                success[i] = False

        if not success.all():
            if config.extrapolate_psat and success.sum() >= 2:
                # August eqn.: ln(P) = a + b/T — linear regression in log space
                X = np.column_stack([np.ones(success.sum()), inv_T_psat[success]])
                coeffs = np.linalg.lstsq(X, np.log(p_pred[success]), rcond=None)[0]
                p_pred[~success] = np.exp(coeffs[0] + coeffs[1] * inv_T_psat[~success])
            else:
                return np.full(n_total, _PENALTY)

        residuals.append(psat_cost_scale * (p_pred * inv_d_psat - 1.0))

        # Density residuals
        if n_rho > 0:
            try:
                rho_pred_vals = [
                    feos.PhaseEquilibrium.pure(
                        eos, T * temperature_unit
                    ).liquid.mass_density()
                    / rho_unit
                    for T in T_rho
                ]
                rho_pred = np.array(rho_pred_vals)
            except Exception as exc:
                if errors is not None:
                    errors.append(exc)
                return np.full(n_total, _PENALTY)

            residuals.append(rho_cost_scale * (rho_pred * inv_d_rho - 1.0))

        # Enthalpy of vaporization residuals
        if n_hvap > 0:
            try:
                hvap_pred_vals = []
                for T in T_hvap:
                    vle = feos.PhaseEquilibrium.pure(eos, T * temperature_unit)
                    hvap_pred_vals.append(
                        (
                            vle.vapor.molar_enthalpy(feos.Contributions.Residual)
                            - vle.liquid.molar_enthalpy(feos.Contributions.Residual)
                        )
                        / enthalpy_unit
                    )
                hvap_pred = np.array(hvap_pred_vals)
            except Exception as exc:
                if errors is not None:
                    errors.append(exc)
                return np.full(n_total, _PENALTY)

            residuals.append(hvap_cost_scale * (hvap_pred * inv_d_hvap - 1.0))

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
