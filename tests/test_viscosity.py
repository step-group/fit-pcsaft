"""Fast offline smoke test for entropy-scaling viscosity fitting.

Mirrors examples/pure/09_viscosity_hexane.py: build feos.Parameters inline (no
PubChem), fit [A, B, C, D] via the fast lstsq path. Regression-style, loose bounds.
"""
import math
from pathlib import Path

import feos

from fit_pcsaft import ViscosityFitResult, fit_viscosity_entropy_scaling

DATA = Path(__file__).parent.parent / "examples" / "data"


def _hexane_params():
    # PC-SAFT parameters for hexane (Gross & Sadowski 2001)
    ident = feos.Identifier(
        cas="110-54-3",
        name="hexane",
        iupac_name="hexane",
        smiles="CCCCCC",
        inchi="InChI=1S/C6H14/c1-3-5-6-4-2/h3-6H2,1-2H3",
        formula="C6H14",
    )
    record = feos.PureRecord(
        identifier=ident, molarweight=86.177, m=3.0576, sigma=3.7983, epsilon_k=236.77
    )
    return feos.Parameters.new_pure(record)


def test_viscosity_hexane_fit():
    result = fit_viscosity_entropy_scaling(
        _hexane_params(),
        DATA / "viscosity" / "hexane_viscosity.csv",
        name="hexane",
        groups={"CH3": 2, "CH2": 4},
    )
    assert isinstance(result, ViscosityFitResult)
    assert len(result.viscosity_params) == 4
    assert all(math.isfinite(p) for p in result.viscosity_params)
    assert result.a_fixed and result.d_fixed  # groups -> A fixed; D always fixed
    assert math.isfinite(result.ard) and result.ard < 10.0


def test_viscosity_robust_loss_matches_linear_on_clean_data():
    """With an exact Jacobian, a robust fit on clean data lands on the OLS fit.

    Exercises the real fit path, not a synthetic scipy call. A wrong or noisy
    Jacobian moves the robust optimum away from the OLS solution.
    """
    kwargs = dict(
        viscosity_path=DATA / "viscosity" / "hexane_viscosity.csv",
        name="hexane",
        groups={"CH3": 2, "CH2": 4},
    )
    ols = fit_viscosity_entropy_scaling(_hexane_params(), loss="linear", **kwargs)
    rob = fit_viscosity_entropy_scaling(
        _hexane_params(), loss="huber", f_scale=10.0, **kwargs
    )

    for a, b in zip(ols.viscosity_params, rob.viscosity_params):
        assert math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-6)
