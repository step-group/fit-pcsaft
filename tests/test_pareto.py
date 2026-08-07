import numpy as np
import pytest

from fit_pcsaft._pure.pareto import _BIG, _argmin_scalarized, aad_objectives, non_dominated
from fit_pcsaft._types import Units
from tests.test_surface_tension import (
    HEXANE,
    HEXANE_P,
    HEXANE_SPEC,
    _hexane_data_with_sft,
)


def test_non_dominated_keeps_only_the_front():
    F = np.array([
        [1.0, 5.0],   # on front
        [2.0, 2.0],   # on front
        [5.0, 1.0],   # on front
        [3.0, 3.0],   # dominated by (2,2)
        [2.0, 2.0],   # duplicate of a front point
    ])
    mask = non_dominated(F)
    assert mask[0] and mask[1] and mask[2]
    assert not mask[3]
    assert not mask[4]


def test_non_dominated_single_point():
    assert non_dominated(np.array([[1.0, 1.0]])).tolist() == [True]


def test_objectives_zero_on_self_consistent_sft():
    data = _hexane_data_with_sft()
    aad_vle, aad_sft = aad_objectives(HEXANE_P, HEXANE, HEXANE_SPEC, data, Units())
    assert aad_sft == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(aad_vle)


def test_objectives_penalize_impossible_parameters():
    data = _hexane_data_with_sft()
    bad = np.array([1.0e-6, 1.0e-6, 1.0e-6])
    assert aad_objectives(bad, HEXANE, HEXANE_SPEC, data, Units()) == (_BIG, _BIG)


def test_select_picks_the_tangent_point():
    F = np.array([[1.0, 6.0], [2.0, 2.0], [6.0, 1.0]])
    # ref_vle=2%, ref_sft=1 -> costs: 6.5, 3.0, 4.0 -> index 1
    assert _argmin_scalarized(F, ref_vle=2.0, ref_sft=1.0) == 1
    # heavy weight on sft (small ref_sft) -> the low-sft corner wins
    assert _argmin_scalarized(F, ref_vle=100.0, ref_sft=0.1) == 2
