"""Aggregate failures must name the exception that actually caused them.

Every feos call in the fitting loops sits inside a bare ``except Exception``.
That is deliberate — individual points legitimately fall outside the EOS
validity range — but it meant the feos 0.10 API breaks surfaced as
"check that T/P conditions are within the EOS validity range" or, worse, as
``Converged: True`` on a fit that never left its initial guess.

Each test below monkeypatches a feos entry point to raise the *exact*
exception the 0.10 upgrade produced, and asserts the user-facing error names
it.
"""
import math
from pathlib import Path

import feos
import pytest

from fit_pcsaft import fit_kij_lle, fit_pure, fit_viscosity_entropy_scaling

DATA = Path(__file__).parent.parent / "examples" / "data"

# The three real exceptions from the feos 0.9 -> 0.10 upgrade.
_STATE_KWARG_ERROR = "State.__new__() got an unexpected keyword argument 'total_moles'"
_MOLES_KWARG_ERROR = "State.__new__() got an unexpected keyword argument 'moles'"
_AD_MOVED_ERROR = "module 'feos' has no attribute 'vapor_pressure_derivatives'"


def _hexane_params():
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


def _raiser(message, exc_type=TypeError):
    def boom(*args, **kwargs):
        raise exc_type(message)

    return boom


def test_viscosity_reports_underlying_state_error(monkeypatch):
    """0 valid points must name the TypeError, not blame the T/P range."""
    monkeypatch.setattr(feos, "State", _raiser(_STATE_KWARG_ERROR))

    with pytest.raises(RuntimeError, match="total_moles"):
        fit_viscosity_entropy_scaling(
            _hexane_params(),
            DATA / "viscosity" / "hexane_viscosity.csv",
            name="hexane",
            groups={"CH3": 2, "CH2": 4},
        )


def test_lle_reports_underlying_state_error(monkeypatch):
    """"No temperatures converged" must name the TypeError, not kij_bounds."""
    import json

    water_models_path = DATA / "parameters" / "water_models.json"
    water_name = json.loads(water_models_path.read_text())[0]["identifier"]["name"]

    monkeypatch.setattr(feos, "State", _raiser(_MOLES_KWARG_ERROR))

    with pytest.raises(RuntimeError, match="moles"):
        fit_kij_lle(
            id1="1-octanol",
            id2=water_name,
            lle_path=DATA / "lle" / "1-octanol_water.csv",
            params_path=[DATA / "parameters" / "alkanols_lle.json", water_models_path],
            kij_order=0,
            require_both_phases=False,
        )


def test_pure_fit_raises_instead_of_reporting_a_stalled_fit(monkeypatch):
    """A dead AD Jacobian must raise, never return the initial guess as converged.

    This is the regression that silently wrote m=4.0, sigma=3.7, epsilon_k=300
    into examples_pure.json while printing "Converged: True".
    """

    class _DeadProperty:
        vapor_pressure_derivatives = staticmethod(
            _raiser(_AD_MOVED_ERROR, AttributeError)
        )
        equilibrium_liquid_density_derivatives = staticmethod(
            _raiser(_AD_MOVED_ERROR, AttributeError)
        )

    monkeypatch.setattr(feos, "Property", _DeadProperty)

    with pytest.raises(RuntimeError, match="vapor_pressure_derivatives"):
        fit_pure(
            id="propane",
            psat_path=DATA / "psat" / "propane.csv",
            density_path=DATA / "density" / "propane.csv",
        )


def test_healthy_fit_still_succeeds():
    """Guard against the stall check firing on a legitimate fit."""
    result = fit_pure(
        id="propane",
        psat_path=DATA / "psat" / "propane.csv",
        density_path=DATA / "density" / "propane.csv",
    )
    assert math.isclose(result.params["m"], 2.008839, rel_tol=1e-4)
    assert result.metrics["psat"].aard_pct < 30.0
