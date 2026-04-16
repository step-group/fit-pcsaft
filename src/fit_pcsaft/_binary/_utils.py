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

    Raises ValueError if both or neither component is self-associating.
    """
    import json

    assoc1 = _is_self_associating(record1)
    assoc2 = _is_self_associating(record2)

    if assoc1 and assoc2:
        raise ValueError(
            "induced_assoc=True requires exactly one self-associating component, "
            "but both components have association sites with epsilon_k_ab > 0."
        )
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


def _make_binary_jac_fn(fun, n_params: int, h: float = 1e-012):
    """Build a central-difference (3-point) Jacobian for a binary cost function."""

    def jac(x: np.ndarray) -> np.ndarray:
        cols = []
        for i in range(n_params):
            dx = np.zeros(n_params)
            dx[i] = h
            cols.append((fun(x + dx) - fun(x - dx)) / (2 * h))
        return np.column_stack(cols) if len(cols) > 1 else np.array(cols).T

    return jac
