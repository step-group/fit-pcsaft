"""The batched/parallel fit paths must reproduce their serial counterparts.

fit_pure_de's population objective against the per-candidate cost_fn, and
fit_pure's process-pool multistart against the serial loop.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fit_pcsaft._fit_utils import _PENALTY
from fit_pcsaft._pure.fit import (
    _loss_sum,
    _make_de_population_objective,
    _setup_pure_fit,
)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("NO_NETWORK")), reason="_fetch_compound hits PubChem"
)


def _setup(**kw):
    import si_units as si

    defaults = {
        "id": "propane",
        "psat_path": DATA / "psat" / "propane.csv",
        "density_path": DATA / "density" / "propane.csv",
        "hvap_path": DATA / "hvap" / "propane.csv",
        "mu": 0.0, "q": 0.0, "na": None, "nb": None,
        "psat_weight": 3.0, "density_weight": 2.0, "hvap_weight": 1.0,
        "extrapolate_psat": False,
        "pressure_unit": si.KILO * si.PASCAL,
        "temperature_unit": si.KELVIN,
        "density_unit": si.KILOGRAM / si.METER**3,
        "enthalpy_unit": si.KILO * si.JOULE / si.MOL,
        "verbose": False,
    }
    defaults.update(kw)
    return _setup_pure_fit(**defaults)


def _population():
    rng = np.random.default_rng(0)
    # Wide enough that some candidates have no VLE at all and some lose only
    # the near-critical psat points (exercising extrapolation).
    lo, hi = np.array([1.0, 2.0, 50.0]), np.array([20.0, 6.0, 700.0])
    X = lo + (hi - lo) * rng.random((60, 3))
    near = np.array([[2.0, 3.7, 208.0], [2.0, 3.7, 205.0], [2.0, 3.73, 210.7]])
    return np.vstack([X, near])


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"extrapolate_psat": True},
        {"loss": "huber", "f_scale": 0.05},
        {"extrapolate_psat": True, "loss": "cauchy", "f_scale": 0.1},
        {"extrapolate_psat": True, "short_rho": True},
    ],
)
def test_population_objective_matches_cost_fn_row_for_row(kw, tmp_path):
    kw = dict(kw)
    short_rho = kw.pop("short_rho", False)
    if short_rho:
        # Density only to 300 K and no hvap: a candidate whose Tc falls inside
        # the psat range then survives on extrapolated psat instead of a penalty.
        from fit_pcsaft._csv import load_density_csv

        T, rho = load_density_csv(DATA / "density" / "propane.csv")
        kw["density_path"] = tmp_path / "rho.csv"
        kw["density_path"].write_text(
            "T,rho\n" + "".join(f"{a},{b}\n" for a, b in zip(T, rho) if a <= 300.0)
        )
        kw["hvap_path"] = None
    data, compound, spec, units, config, cost_fn, _, errors, _, _ = _setup(**kw)
    f = _make_de_population_objective(data, compound, spec, units, config, errors)
    X = _population()

    batched = f(X.T)
    serial = np.array([_loss_sum(cost_fn(np.sqrt(x)), config.loss) for x in X])

    assert batched.shape == (len(X),)
    penalty = _loss_sum(np.full(len(cost_fn(np.sqrt(X[0]))), _PENALTY), config.loss)
    assert (serial == penalty).any() and (serial < penalty).any()  # both regimes covered
    np.testing.assert_allclose(batched, serial, rtol=1e-9)
    # A single (n_params,) vector, as DE's polish sends it.
    np.testing.assert_allclose(f(X[-1]), serial[-1:], rtol=1e-9)
    if short_rho:  # the extrapolation must actually rescue a candidate here
        s = _setup(**{**kw, "extrapolate_psat": False})
        plain = _make_de_population_objective(*s[:5], s[7])(X.T)
        assert ((plain == penalty) & (batched < penalty)).any()


def test_population_covers_partial_psat_failure():
    """At least one candidate must fail only some psat points, else extrapolation is untested."""
    import feos

    from fit_pcsaft._fit_utils import ad_model, ad_rows

    data, _, spec, *_ = _setup()
    X = _population()
    nT = len(data.T_psat)
    _, _, ok = feos.Property.vapor_pressure_derivatives(
        ad_model(spec), [], np.repeat(ad_rows(X, spec), nT, axis=0),
        np.ascontiguousarray(np.tile(data.T_psat, len(X))[:, None]),
    )
    n_ok = ok.reshape(len(X), nT).sum(axis=1)
    assert ((n_ok >= 2) & (n_ok < nT)).any()


def test_fit_pure_sft_pool_matches_serial(tmp_path):
    """Same starts, same tie-break: the pooled multistart returns the serial fit exactly."""
    from fit_pcsaft import fit_pure

    sft = tmp_path / "sft.csv"
    sft.write_text("T,sft\n280.00,74.6768\n355.00,62.3330\n445.00,43.9985\n")
    kw = {
        "id": "water", "na": 1, "nb": 1,
        "psat_path": DATA / "psat" / "water.csv",
        "density_path": DATA / "density" / "water.csv",
        "sft_path": sft,
        "scipy_kwargs": {"max_nfev": 3},  # cheap: equality, not convergence, is the point
    }
    serial = fit_pure(n_jobs=1, **kw)
    pooled = fit_pure(n_jobs=3, **kw)
    assert pooled.params == serial.params
    assert pooled.scipy_result.cost == serial.scipy_result.cost
