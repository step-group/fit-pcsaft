"""Domain dataclasses for fit-pcsaft."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import si_units as si


@dataclass
class PureData:
    T_psat: np.ndarray  # vapor pressure temperatures (user units)
    p_psat: np.ndarray  # vapor pressures (user units)
    T_rho: np.ndarray  # density temperatures (user units)
    rho: np.ndarray  # liquid densities (user units)
    T_hvap: np.ndarray = field(default_factory=lambda: np.array([]))  # hvap temperatures (user units)
    hvap: np.ndarray = field(default_factory=lambda: np.array([]))  # enthalpies of vaporization (user units)


@dataclass
class Compound:
    identifier: object  # feos.Identifier
    mw: float


@dataclass
class ModelSpec:
    mu: Optional[float] = 0.0  # None = fit it
    na: Optional[int] = None
    nb: Optional[int] = None
    q: float = 0.0


@dataclass
class Units:
    temperature: object = field(default_factory=lambda: si.KELVIN)
    pressure: object = field(default_factory=lambda: si.KILO * si.PASCAL)
    density: object = field(default_factory=lambda: si.KILOGRAM / si.METER**3)
    enthalpy: object = field(default_factory=lambda: si.KILO * si.JOULE / si.MOL)


@dataclass
class FitConfig:
    w_psat: float = 3.0
    w_rho: float = 2.0
    w_hvap: float = 1.0
    extrapolate_psat: bool = False
