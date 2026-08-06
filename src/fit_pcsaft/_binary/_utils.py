"""Shared utilities for binary k_ij fitting."""

from pathlib import Path

import feos
import numpy as np
import si_units as si


def _load_pure_records(
    params_path: "Path | str | list[Path | str]", id1: str, id2: str
) -> "tuple[feos.PureRecord, feos.PureRecord]":
    """Load two pure-component records from one or more feos JSON parameter files.

    When params_path is a list, the JSON arrays are merged into a temporary
    file so feos can search across all of them.
    """
    import json
    import tempfile

    if isinstance(params_path, (list, tuple)):
        combined = []
        for p in params_path:
            combined.extend(json.loads(Path(p).read_text(encoding="utf-8")))
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(combined, tmp)
        tmp.close()
        pure_path = tmp.name
    else:
        pure_path = str(params_path)

    params = feos.Parameters.from_json([id1, id2], pure_path=pure_path)

    if isinstance(params_path, (list, tuple)):
        Path(pure_path).unlink(missing_ok=True)

    records = params.pure_records
    return records[0], records[1]


def _build_binary_eos(
    record1: "feos.PureRecord", record2: "feos.PureRecord", kij: float
) -> "feos.EquationOfState":
    """Build a binary PC-SAFT EOS with the given k_ij."""
    params = feos.Parameters.new_binary([record1, record2], k_ij=kij)
    return feos.EquationOfState.pcsaft(params, max_iter_cross_assoc=100)


def _is_self_associating(record: "feos.PureRecord") -> bool:
    """Return True if the record has at least one full association site with kappa_ab and epsilon_k_ab > 0."""
    for site in record.association_sites:
        if "kappa_ab" in site and "epsilon_k_ab" in site and float(site["epsilon_k_ab"]) > 0.0:
            return True
    return False


def _apply_induced_association(
    record1: "feos.PureRecord", record2: "feos.PureRecord"
) -> "tuple[feos.PureRecord, feos.PureRecord]":
    """Apply the induced-association mixing rule to a self-associating / non-associating pair.

    The non-associating component receives:
      - epsilon_k_ab = 0.0
      - kappa_ab     = kappa_ab of the self-associating component (first full site)
      - na = 1.0, nb = 1.0  (2B scheme)

    If both components are already self-associating, induced association is not applicable
    and the records are returned unchanged (a note is printed).
    Raises ValueError if neither component is self-associating.
    """
    import json
    import warnings

    assoc1 = _is_self_associating(record1)
    assoc2 = _is_self_associating(record2)

    if assoc1 and assoc2:
        warnings.warn(
            "induced_association=True has no effect: both components are already "
            "self-associating. Records returned unchanged.",
            stacklevel=3,
        )
        return record1, record2
    if not assoc1 and not assoc2:
        raise ValueError(
            "induced_assoc=True requires exactly one self-associating component, "
            "but neither component has association sites with epsilon_k_ab > 0."
        )

    assoc_record, solvating_record = (record1, record2) if assoc1 else (record2, record1)

    # Pick kappa_ab from the first full site of the self-associating component
    kappa_ab = float(assoc_record.association_sites[0]["kappa_ab"])

    # Rebuild the solvating record with induced-association site
    d = solvating_record.to_dict()
    d["association_sites"] = [{"na": 1.0, "nb": 1.0, "kappa_ab": kappa_ab, "epsilon_k_ab": 0.0}]
    solvating_mod = feos.PureRecord.from_json_str(json.dumps(d))

    return (record1, solvating_mod) if assoc1 else (solvating_mod, record2)


def _kij_at_T(coeffs: np.ndarray, T: float, t_ref: float) -> float:
    """Evaluate the k_ij polynomial at temperature T."""
    dT = T - t_ref
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * dT**i
    return result


LOG_PENALTY = 10.0
"""Residual returned for a failed prediction when log_residuals=True.

In linear-x the failure penalty is 1.0, safely larger than any real
|x_pred - x_exp| (both are mole fractions <= 1). In ln-space that no longer
holds: |ln(x_pred/x_exp)| = 1.0 is only a factor of e, so a penalty of 1.0
would make a *failed* flash score better than a converged-but-poor prediction,
and least_squares would walk k_ij toward the region where the flash breaks.
10.0 = a factor of 22000, above ln(1000) = 6.9, the worst residual a real
(if bad) PC-SAFT water model produces on aqueous terpene solubilities.
"""


def _comp_resid(pred: float, exp: float, log_residuals: bool,
                relative: bool = False) -> float:
    """One composition residual, in ln-space or linear/relative space.

    log_residuals=True returns ln(pred) - ln(exp) and *overrides* `relative`:
    a log difference is already a relative measure. Non-finite or non-positive
    predictions -- and non-positive experimental values, which cannot be
    logged -- return the failure penalty for the active mode.

    Sign convention matches _metrics.py: positive = model overshoots.
    """
    if log_residuals:
        if not np.isfinite(pred) or pred <= 0.0 or not np.isfinite(exp) or exp <= 0.0:
            return LOG_PENALTY
        return float(np.log(pred) - np.log(exp))
    if not np.isfinite(pred):
        return 1.0
    return (pred - exp) / max(exp, 1e-6) if relative else pred - exp


def _fit_kij_polynomial(
    T_arr: np.ndarray,
    kij_arr: np.ndarray,
    ard_arr: np.ndarray,
    kij_order: int,
    kij_t_ref: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Fit k_ij(T) polynomial with ARD-based weighting and Cauchy robust loss.

    Points with high per-point ARD (unreliable k_ij) are down-weighted using:
        w_i = 1 / (1 + (ard_i / ard_median)^2)

    The Cauchy loss additionally down-weights outliers in k_ij space.

    Returns (kij_coeffs, unweighted_poly_residuals).
    """
    from scipy.optimize import least_squares as _lsq

    n = len(T_arr)
    effective_order = min(kij_order, n - 1)
    dT = T_arr - kij_t_ref

    ard_med = float(np.median(ard_arr))
    if ard_med > 0.0:
        w = 1.0 / (1.0 + (ard_arr / ard_med) ** 2)
    else:
        w = np.ones(n)
    w_sqrt = np.sqrt(w / w.max())

    ols_rev = np.polyfit(dT, kij_arr, effective_order, w=w_sqrt)
    x0_poly = ols_rev[::-1]  # lowest-order first

    if effective_order == 0 or n == 1:
        kij_coeffs = x0_poly
    else:
        def _poly_resid(coeffs):
            pred = sum(c * dT**j for j, c in enumerate(coeffs))
            return w_sqrt * (pred - kij_arr)

        rob = _lsq(
            _poly_resid, x0_poly,
            loss="cauchy", f_scale=0.01,
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        kij_coeffs = rob.x

    poly_resid = kij_arr - np.array([_kij_at_T(kij_coeffs, float(T), kij_t_ref) for T in T_arr])
    return kij_coeffs, poly_resid


def _make_binary_jac_fn(fun, n_params: int, h: float = 1e-6):
    """Build a central-difference (3-point) Jacobian for a binary cost function.

    ``h`` is absolute, not relative. The parameters are k_ij polynomial
    coefficients: c0 is O(0.01-0.5) and higher coefficients are bounded to
    +/-0.01, so a single absolute step suits every column. Higher-order
    coefficients multiply dT (tens of kelvin), which amplifies their effect
    on k_ij rather than shrinking it.

    Do not lower h. The residuals come from an iterative flash/bubble-point
    solve whose own convergence tolerance sets the noise floor: at h=1e-12
    this returned 7.50 for an ethanol/water bubble-point derivative whose
    true value is 4.4259 (69% error). Every h from 1e-8 to 1e-4 agrees to
    5+ significant digits.
    """

    def jac(x: np.ndarray) -> np.ndarray:
        cols = []
        for i in range(n_params):
            dx = np.zeros(n_params)
            dx[i] = h
            cols.append((fun(x + dx) - fun(x - dx)) / (2 * h))
        return np.column_stack(cols) if len(cols) > 1 else np.array(cols).T

    return jac
