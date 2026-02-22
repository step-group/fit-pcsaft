"""Pure component fitting module.

For pure component PC-SAFT parameter fitting from vapor pressure and
liquid density data using feos automatic differentiation.
"""

from fit_pcsaft._pure.jacobian import _make_f_and_df

__all__ = ["make_f_and_df"]
