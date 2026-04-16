"""Unified CSV loader for experimental data.

All experimental data CSVs are loaded through ``load_csv``, which normalises
column headers via ``_COL_ALIASES`` and validates against a ``CsvSchema``.
Convenience wrappers (``load_psat_csv``, ``load_density_csv``, ``load_hvap_csv``)
preserve the ``(T, property)`` tuple-unpacking pattern used by pure fitting code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Canonical column names and their accepted aliases
# ---------------------------------------------------------------------------

# Any column whose name appears as a key here is mapped to the canonical value.
# Column names that are already canonical (appear as values) are kept as-is
# without needing an explicit entry.
_COL_ALIASES: dict[str, str] = {
    # --- Temperature (canonical: "T") ---
    "temperature_K":         "T",
    "temperature":           "T",
    "Temperature ( K )":     "T",
    "Temperature (K)":       "T",
    "T_K":                   "T",
    "t":                     "T",
    "t_K":                   "T",
    "temp":                  "T",
    # --- Pressure (canonical: "P") ---
    "pressure_kPa":          "P",
    "pressure":              "P",
    "P_kPa":                 "P",
    "p":                     "P",
    # --- Vapor pressure (canonical: "psat") ---
    "vapor pressure ( kPa )":"psat",
    "vapor_pressure":        "psat",
    "vapor_pressure_kPa":    "psat",
    "psat_kPa":              "psat",
    "psat (kPa)":            "psat",
    "Psat":                  "psat",
    "p_sat":                 "psat",
    # --- Liquid density (canonical: "rho") ---
    "density ( kg/m 3 )":    "rho",
    "density":               "rho",
    "density_kg_m3":         "rho",
    "density (kg/m3)":       "rho",
    "rho_kg_m3":             "rho",
    # --- Enthalpy of vaporisation (canonical: "hvap") ---
    "\u0394 vap H (kJ/mol)": "hvap",    # Δ vap H (kJ/mol)  – NIST NIST
    "Delta_vapH (kJ/mol)":   "hvap",
    "delta_vapH":            "hvap",
    "hvap_kJ_mol":           "hvap",
    "hvap":                  "hvap",    # already canonical
    # --- VLE / SLE mole fractions (canonical: "x1", "y1") ---
    "x":                     "x1",
    "x_1":                   "x1",
    "y":                     "y1",
    "y_1":                   "y1",
    # --- LLE mole fractions (canonical: "x1_I", "x1_II") ---
    "xI":                    "x1_I",
    "x_I":                   "x1_I",
    "xII":                   "x1_II",
    "x_II":                  "x1_II",
    # --- LLE mass fractions (canonical: "w1_I", "w1_II") ---
    "wI":                    "w1_I",
    "w_I":                   "w1_I",
    "wII":                   "w1_II",
    "w_II":                  "w1_II",
    # --- Henry's law constant (canonical: "H") ---
    "henry":                 "H",
    "henry_constant":        "H",
    "H_MPa":                 "H",
    "H_kPa":                 "H",
    "H_bar":                 "H",
    "H_Pa":                  "H",
    "h":                     "H",
    "h_atm":                 "H",
    # --- Viscosity (canonical: "eta") ---
    "Viscosity (Pa*s)":      "eta",
    "Viscosity (Pa s)":      "eta",
    "Viscosity (Pa·s)":      "eta",
    "Viscosity (mPa*s)":     "eta",
    "Viscosity":             "eta",
    "viscosity":             "eta",
    "viscosity_Pa_s":        "eta",
    "eta_Pa_s":              "eta",
    # --- Pressure in MPa (canonical: "P") ---
    "Pressure (MPa)":        "P",
    "pressure_MPa":          "P",
    "P_MPa":                 "P",
}

# Set of all canonical names (values in the alias map plus bare canonicals).
_CANONICAL_NAMES: frozenset[str] = frozenset(_COL_ALIASES.values()) | frozenset(
    ["T", "P", "psat", "rho", "hvap", "x1", "y1",
     "x1_I", "x1_II", "w1_I", "w1_II", "H", "eta"]
)


# ---------------------------------------------------------------------------
# Schema declarations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CsvSchema:
    """Declares which canonical columns a data type requires and may have."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = field(default_factory=tuple)
    name: str = ""  # used in error messages


SCHEMA_PSAT       = CsvSchema(required=("T", "psat"),           name="vapor pressure")
SCHEMA_VISCOSITY  = CsvSchema(required=("T", "P", "eta"),       name="viscosity")
SCHEMA_DENSITY    = CsvSchema(required=("T", "rho"),            name="density")
SCHEMA_HVAP       = CsvSchema(required=("T", "hvap"),           name="enthalpy of vaporization")
SCHEMA_VLE        = CsvSchema(required=("T", "P", "x1"),        optional=("y1",),                   name="VLE")
SCHEMA_SLE        = CsvSchema(required=("T", "x1"),             name="SLE")
SCHEMA_LLE        = CsvSchema(required=("T",),                  optional=("x1_I", "x1_II", "w1_I", "w1_II"), name="LLE")
SCHEMA_VLLE       = CsvSchema(required=("T", "P"),              optional=("x1_I", "x1_II", "y1"),   name="VLLE")
SCHEMA_HENRY      = CsvSchema(required=("T", "H"),              name="Henry")


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_csv(path: "Path | str", schema: CsvSchema) -> dict[str, np.ndarray]:
    """Load a CSV file, normalise column names, and validate against *schema*.

    Parameters
    ----------
    path:
        Path to the CSV file.
    schema:
        Declares required and optional canonical column names.

    Returns
    -------
    dict mapping canonical column names to ``float64`` numpy arrays.
    Columns that do not match any known alias or canonical name are silently
    dropped (e.g. ``source``, ``Set Label``, ``Uncertainty``).

    Raises
    ------
    ValueError
        If required columns are missing, a column cannot be converted to
        float, the file is empty, or two CSV columns resolve to the same
        canonical name.
    """
    df = pl.read_csv(
        Path(path),
        infer_schema_length=9999,
        truncate_ragged_lines=True,
    )

    if df.is_empty():
        label = f"{schema.name} " if schema.name else ""
        raise ValueError(
            f"{label}CSV file is empty: {path}"
        )

    # Normalise column names (strip whitespace, then apply alias map).
    result: dict[str, np.ndarray] = {}
    original_headers = df.columns
    for col in original_headers:
        stripped = col.strip()
        canonical = _COL_ALIASES.get(stripped, stripped if stripped in _CANONICAL_NAMES else None)
        if canonical is None:
            continue  # unrecognised column – silently drop
        if canonical in result:
            raise ValueError(
                f"Two CSV columns both map to '{canonical}' in {path}. "
                f"Original headers: {original_headers}"
            )
        try:
            arr = df[col].cast(pl.Float64).to_numpy()
        except Exception:
            raise ValueError(
                f"Column '{col}' in {path} could not be converted to float. "
                f"Check for non-numeric values."
            )
        result[canonical] = arr

    # Check required columns.
    missing = [c for c in schema.required if c not in result]
    if missing:
        label = f"{schema.name} " if schema.name else ""
        raise ValueError(
            f"Missing required {label}columns {missing} in {path}.\n"
            f"CSV headers found: {original_headers}\n"
            f"Expected canonical names: {list(schema.required)}"
        )

    return result


# ---------------------------------------------------------------------------
# Convenience wrappers for pure-component data
# ---------------------------------------------------------------------------

def load_psat_csv(path: "Path | str") -> tuple[np.ndarray, np.ndarray]:
    """Load a vapor-pressure CSV. Returns ``(T, psat)`` arrays."""
    data = load_csv(path, SCHEMA_PSAT)
    return data["T"], data["psat"]


def load_density_csv(path: "Path | str") -> tuple[np.ndarray, np.ndarray]:
    """Load a saturated-liquid-density CSV. Returns ``(T, rho)`` arrays."""
    data = load_csv(path, SCHEMA_DENSITY)
    return data["T"], data["rho"]


def load_hvap_csv(path: "Path | str") -> tuple[np.ndarray, np.ndarray]:
    """Load an enthalpy-of-vaporization CSV. Returns ``(T, hvap)`` arrays."""
    data = load_csv(path, SCHEMA_HVAP)
    return data["T"], data["hvap"]
