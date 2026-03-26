"""Shared utilities for binary k_ij fitting."""

from pathlib import Path

import feos
import numpy as np
import polars as pl
import si_units as si


def _load_pure_records(
    params_path: "Path | str", id1: str, id2: str
) -> "tuple[feos.PureRecord, feos.PureRecord]":
    """Load two pure-component records from a feos JSON parameter file."""
    params_path = str(params_path)
    params = feos.Parameters.from_json(
        [id1, id2],
        pure_path=params_path,
    )
    records = params.pure_records
    return records[0], records[1]


def _build_binary_eos(
    record1: "feos.PureRecord", record2: "feos.PureRecord", kij: float
) -> "feos.EquationOfState":
    """Build a binary PC-SAFT EOS with the given k_ij."""
    params = feos.Parameters.new_binary([record1, record2], k_ij=kij)
    return feos.EquationOfState.pcsaft(params)


def _kij_at_T(coeffs: np.ndarray, T: float, t_ref: float) -> float:
    """Evaluate the k_ij polynomial at temperature T."""
    dT = T - t_ref
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * dT**i
    return result


_COL_ALIASES: dict[str, str] = {
    # Temperature
    "temperature_K": "T",
    "temperature": "T",
    "T_K": "T",
    # Pressure
    "pressure_kPa": "P",
    "pressure": "P",
    "P_kPa": "P",
    # Liquid mole fraction (VLE / SLE solubility)
    "x": "x1",
    "x_1": "x1",
    # Vapor mole fraction (VLE)
    "y": "y1",
    "y_1": "y1",
    # LLE phase mole fractions (already standard; listed for completeness)
    "x1_I": "x1_I",
    "x1_II": "x1_II",
}


def _load_binary_csv(path: "Path | str") -> "dict[str, np.ndarray]":
    """Load a multi-column CSV and return {normalized_column_name: array}.

    Column names are normalized via _COL_ALIASES so callers always see
    standard keys regardless of the source CSV naming style.
    """
    df = pl.read_csv(Path(path), infer_schema_length=9999, truncate_ragged_lines=True)
    result: dict[str, np.ndarray] = {}
    for col in df.columns:
        key = _COL_ALIASES.get(col, col)
        result[key] = df[col].to_numpy()
    return result


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
