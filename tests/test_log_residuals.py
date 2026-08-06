"""ln-ln residuals for dilute-solubility fitting.

The linear objective (x_pred - x) weights an aqueous branch at x~1e-4 to
essentially nothing, so a fit that is a factor of 3 wrong there costs less
than a fit 1% wrong on the organic branch at x~0.8. The log residual is
scale-free and symmetric in over/under-prediction.
"""
import inspect

import numpy as np
import pytest

from fit_pcsaft._binary._utils import LOG_PENALTY, _comp_resid


def test_linear_mode_is_unchanged():
    assert _comp_resid(0.3, 0.2, log_residuals=False) == pytest.approx(0.1)
    assert _comp_resid(0.3, 0.2, log_residuals=False, relative=True) == pytest.approx(0.5)
    assert _comp_resid(float("nan"), 0.2, log_residuals=False) == 1.0


def test_log_mode_is_the_log_ratio():
    assert _comp_resid(2e-4, 1e-4, log_residuals=True) == pytest.approx(np.log(2.0))
    assert _comp_resid(1e-4, 1e-4, log_residuals=True) == pytest.approx(0.0)


def test_log_overrides_relative():
    """A log difference is already a relative measure."""
    assert _comp_resid(2e-4, 1e-4, log_residuals=True, relative=True) == pytest.approx(
        np.log(2.0)
    )


def test_log_mode_is_symmetric_and_scale_free():
    over = _comp_resid(2e-4, 1e-4, log_residuals=True)
    under = _comp_resid(1e-4, 2e-4, log_residuals=True)
    assert over == pytest.approx(-under)
    # same ratio error, four orders of magnitude apart, same cost
    assert _comp_resid(1.1e-4, 1.0e-4, log_residuals=True) == pytest.approx(
        _comp_resid(0.88, 0.80, log_residuals=True)
    )


@pytest.mark.parametrize("pred", [float("nan"), float("inf"), 0.0, -1e-6])
def test_unloggable_prediction_takes_the_penalty(pred):
    assert _comp_resid(pred, 1e-4, log_residuals=True) == LOG_PENALTY


def test_unloggable_experiment_takes_the_penalty():
    assert _comp_resid(1e-4, 0.0, log_residuals=True) == LOG_PENALTY
    assert _comp_resid(1e-4, float("nan"), log_residuals=True) == LOG_PENALTY


def test_penalty_dominates_any_real_residual():
    """The whole reason the penalty is not 1.0.

    In ln-space a factor-e error is a residual of 1.0, so a penalty of 1.0
    would make a failed flash *preferable* to a converged-but-poor prediction
    and least_squares would chase failures. Eugenol's aqueous branch is
    already at |ln| ~ 2.1 with the incumbent water model.
    """
    worst_plausible = abs(np.log(1e-3 / 1.0))  # a 1000x miss -> 6.9
    assert LOG_PENALTY > worst_plausible


def test_sle_exposes_log_residuals_off_by_default():
    from fit_pcsaft import fit_kij_sle

    assert inspect.signature(fit_kij_sle).parameters["log_residuals"].default is False
