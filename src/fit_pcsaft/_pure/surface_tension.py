"""Surface tension of pure components from feos density functional theory.

The interface profile is initialized with predictive density gradient theory
(pDGT) and then relaxed by full DFT. feos' Python binding discards the pDGT
surface tension itself — ``PlanarInterface.from_pdgt`` returns only the density
profile — so the value reported here is the converged DFT result.

Note on psi: feos hardcodes ``PSI_DFT = 1.3862`` (Sauer 2017) for homosegmented
PC-SAFT. Rehner & Gross (2020) use ``psi_pDGT = 1.3286``. There is no way to
override it from Python, so our numbers track the paper's AAD_DFT column, not
its AAD_pDGT column.
"""

from __future__ import annotations

from dataclasses import dataclass

import feos
import numpy as np


@dataclass(frozen=True)
class SurfaceTensionOptions:
    """DFT grid settings.

    ``n_grid=256`` reproduces ``n_grid=2048`` to six significant figures when
    the profile is pDGT-initialized, at roughly 5 ms per temperature instead of
    39 ms. The ``l_grid``/``critical_temperature`` values are only used by the
    tanh fallback initializer.
    """

    n_grid: int = 256
    l_grid_angstrom: float = 100.0
    critical_temperature_k: float = 500.0


_DEFAULT_OPTIONS = SurfaceTensionOptions()


def predict_surface_tension(functional, T_vals, units, options=None) -> np.ndarray:
    """Surface tension at each temperature, in ``units.surface_tension``.

    Returns NaN for any temperature where the VLE or the DFT solve fails
    (supercritical, non-converging, or a parameter set with no stable
    interface). Never raises.

    ``BaseException`` is caught deliberately: ``PlanarInterface.from_pdgt``
    signals unsupported input with ``pyo3_runtime.PanicException``, which
    derives from ``BaseException`` and slips past ``except Exception``.
    """
    opts = options or _DEFAULT_OPTIONS
    T_vals = np.asarray(T_vals, dtype=float)
    out = np.full(T_vals.size, np.nan)
    if T_vals.size == 0:
        return out

    tu = units.temperature
    su = units.surface_tension

    for i, T in enumerate(T_vals):
        try:
            vle = feos.PhaseEquilibrium.pure(functional, float(T) * tu)
            iface = feos.PlanarInterface.from_pdgt(vle, opts.n_grid)
            iface.solve()
            gamma = iface.surface_tension
            if gamma is None:
                continue
            out[i] = float(gamma / su)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            continue

    # gamma -> 0 at the critical point and feos can return a tiny negative
    # value there; treat non-positive results as unavailable.
    out[out <= 0.0] = np.nan
    return out
