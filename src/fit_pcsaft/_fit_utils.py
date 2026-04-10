import re
from pathlib import Path
from typing import Callable, Tuple

import feos
import numpy as np
import polars as pl
import pubchempy as pcp

from fit_pcsaft._csv import load_density_csv, load_hvap_csv, load_psat_csv
from fit_pcsaft._types import Compound, FitConfig, ModelSpec, PureData, Units


def _fetch_compound(id_str: str) -> Tuple[feos.Identifier, float]:
    """Fetch compound information from PubChem.

    Tries multiple lookup methods:

    1. By common name
    2. By SMILES
    3. By InChI
    """
    compound = None

    # Try name lookup first
    if not compound:
        try:
            compounds = pcp.get_compounds(id_str, "name")
            if compounds:
                compound = compounds[0]
        except Exception:
            pass

    # Try SMILES lookup
    if not compound:
        try:
            compounds = pcp.get_compounds(id_str, "smiles")
            if compounds:
                compound = compounds[0]
        except Exception:
            pass

    # Try InChI lookup
    if not compound:
        try:
            compounds = pcp.get_compounds(id_str, "inchi")
            if compounds:
                compound = compounds[0]
        except Exception:
            pass

    if not compound:
        raise ValueError(f"Compound '{id_str}' not found in PubChem")

    # Extract CAS from synonyms
    cas = None
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
    except Exception:
        pass

    if not cas:
        raise ValueError(f"Could not extract CAS number for '{id_str}'")

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



def _build_eos(
    params_vec: np.ndarray,
    compound: Compound,
    spec: ModelSpec,
) -> feos.EquationOfState:
    """
    Build PC-SAFT equation of state from parameters.

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
        record = feos.PureRecord(
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
    else:
        record = feos.PureRecord(
            identifier=identifier,
            molarweight=mw,
            m=m,
            sigma=sigma,
            epsilon_k=epsilon_k,
            mu=mu_val,
            q=q,
        )

    parameters = feos.Parameters.new_pure(record)
    return feos.EquationOfState.pcsaft(parameters)


_EPS_2PT = np.sqrt(np.finfo(float).eps)  # ~1.49e-8


def _make_cost_fn(
    data: PureData,
    compound: Compound,
    spec: ModelSpec,
    units: Units,
    config: FitConfig,
) -> Callable:
    """Create cost function closure for non-associating optimization."""
    T_psat = data.T_psat
    d_psat = data.p_psat
    T_rho = data.T_rho
    d_rho = data.rho
    T_hvap = data.T_hvap
    d_hvap = data.hvap
    temperature_unit = units.temperature
    psat_unit = units.pressure
    rho_unit = units.density
    enthalpy_unit = units.enthalpy

    n_psat = len(T_psat)
    n_rho = len(T_rho)
    n_hvap = len(T_hvap)
    n_total = n_psat + n_rho + n_hvap
    psat_cost_scale = np.sqrt(config.w_psat / n_psat)
    rho_cost_scale = np.sqrt(config.w_rho / n_rho) if n_rho > 0 else 0.0
    hvap_cost_scale = np.sqrt(config.w_hvap / n_hvap) if n_hvap > 0 else 0.0
    inv_d_psat = 1.0 / d_psat
    inv_d_rho = 1.0 / d_rho if n_rho > 0 else None
    inv_d_hvap = 1.0 / d_hvap if n_hvap > 0 else None
    inv_T_psat = 1.0 / T_psat

    def cost_function(params_vec):
        """Compute weighted relative residuals."""
        try:
            eos = _build_eos(params_vec**2, compound, spec)
        except Exception:
            return np.full(n_total, 1e10)

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
            except Exception:
                success[i] = False

        if not success.all():
            if config.extrapolate_psat and success.sum() >= 2:
                # August eqn.: ln(P) = a + b/T — linear regression in log space
                X = np.column_stack([np.ones(success.sum()), inv_T_psat[success]])
                coeffs = np.linalg.lstsq(X, np.log(p_pred[success]), rcond=None)[0]
                p_pred[~success] = np.exp(coeffs[0] + coeffs[1] * inv_T_psat[~success])
            else:
                return np.full(n_total, 1e10)

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
            except Exception:
                return np.full(n_total, 1e10)

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
            except Exception:
                return np.full(n_total, 1e10)

            residuals.append(hvap_cost_scale * (hvap_pred * inv_d_hvap - 1.0))

        return np.concatenate(residuals)

    return cost_function


def _make_f_and_df_numerical(
    data: PureData,
    compound: Compound,
    spec: ModelSpec,
    units: Units,
    config: FitConfig,
) -> Tuple[Callable, Callable]:
    """Create cost function + 2-point numerical Jacobian with shared base-eval cache.

    Scipy calls f(x) then jac(x) at the same x each iteration. Caching the last
    (x, f(x)) means the Jacobian reuses the base evaluation instead of rerunning feos.
    """
    _cost = _make_cost_fn(data, compound, spec, units, config)

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

    return f, df
