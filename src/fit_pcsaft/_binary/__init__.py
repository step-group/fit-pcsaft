"""Binary interaction parameter fitting module.

For binary interaction parameter k_ij from:
    - VLE data -- bubble-point pressure and vapor composition
    - LLE data -- liquid-liquid equilibrium compositions
    - SLE data -- requires enthalpy of fusion and melting point
"""
from fit_pcsaft._binary.lle import fit_kij_lle
from fit_pcsaft._binary.result import BinaryFitResult
from fit_pcsaft._binary.sle import fit_kij_sle
from fit_pcsaft._binary.vle import fit_kij_vle

__all__ = ["fit_kij_vle", "fit_kij_lle", "fit_kij_sle", "BinaryFitResult"]
