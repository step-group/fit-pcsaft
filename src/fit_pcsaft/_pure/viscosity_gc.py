"""Group-contribution helper for Lötgering-Lin entropy scaling viscosity.

Computes A_gc = Σ_α (n_α · m_α · σ_α³ · A_α) from LL & Gross 2015 (eq. 11).
Segment parameters (m_α, σ_α) are from Sauer et al. 2014; viscosity group
parameters (A_α) are from Lötgering-Lin & Gross 2015, Table 1.  Both are
bundled in viscosity_gc_data/loetgeringlin2015_homo.json.
"""

import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "viscosity_gc_data" / "loetgeringlin2015_homo.json"
_GROUP_TABLE: dict = {}


def _load() -> None:
    global _GROUP_TABLE
    if not _GROUP_TABLE:
        data = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        _GROUP_TABLE = {e["identifier"]: e for e in data}


def compute_a_gc(groups: "dict[str, int]") -> float:
    """LL & Gross 2015 eq. 11: Σ_α (n_α · m_α · σ_α³ · A_α).

    Parameters
    ----------
    groups : dict[str, int]
        Segment name → count, e.g. ``{"CH3": 2, "CH2": 4}`` for hexane.
        Names must match identifiers in the bundled loetgeringlin2015_homo.json.

    Returns
    -------
    float
        A_gc value (unnormalised; caller must add 0.5·ln(1/m_i) shift).

    Raises
    ------
    KeyError
        If a group name is unknown or the group lacks viscosity parameters.
    """
    _load()
    _supported = sorted(k for k, v in _GROUP_TABLE.items() if "viscosity" in v)

    total = 0.0
    for name, count in groups.items():
        if name not in _GROUP_TABLE:
            raise KeyError(
                f"Unknown group '{name}'. "
                f"Available groups: {sorted(_GROUP_TABLE)}"
            )
        entry = _GROUP_TABLE[name]
        if "viscosity" not in entry:
            raise KeyError(
                f"Group '{name}' has no viscosity parameters. "
                f"Groups with viscosity params: {_supported}"
            )
        m_a = entry["m"]
        sigma_a = entry["sigma"]
        A_a = entry["viscosity"][0]
        total += count * m_a * sigma_a**3 * A_a

    return total


def available_groups() -> list[str]:
    """Return segment names that have viscosity parameters."""
    _load()
    return sorted(k for k, v in _GROUP_TABLE.items() if "viscosity" in v)
