from pathlib import Path

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


# Rehner & Gross 2020, water, PC-SAFT: parameters from Table 1, AADs from Table 2.
# (scheme, params, na, nb, paper AAD_vle %, paper AAD_DFT mN/m)
WATER_SCHEMES = [
    ("2B", [1.0000, 2.9375, 272.03, 0.044480, 3125.3], 1, 1, 2.14, 1.59),
    ("3B", [1.6330, 2.4570, 238.32, 0.037807, 2749.0], 2, 1, 3.06, 1.14),
    ("4C", [1.8668, 2.3950, 169.78, 0.133738, 1772.0], 2, 2, 1.84, 1.81),
]


def _water_data():
    import feos

    from fit_pcsaft._csv import load_density_csv, load_psat_csv, load_sft_csv
    from fit_pcsaft._types import Compound, PureData

    d = Path(__file__).parent.parent / "examples" / "data"
    T1, p1 = load_psat_csv(d / "psat" / "water.csv")
    T2, r2 = load_density_csv(d / "density" / "water.csv")
    T3, g3 = load_sft_csv(d / "surface_tension" / "water.csv")
    data = PureData(T_psat=T1, p_psat=p1, T_rho=T2, rho=r2, T_sft=T3, sft=g3)
    water = Compound(
        identifier=feos.Identifier(name="water", cas="7732-18-5"), mw=18.015
    )
    return data, water


@pytest.mark.parametrize(
    "scheme,params,na,nb,paper_vle,paper_sft", WATER_SCHEMES,
    ids=[c[0] for c in WATER_SCHEMES],
)
def test_water_forward_check_against_paper(scheme, params, na, nb, paper_vle, paper_sft):
    """Rehner & Gross Tables 1 and 2, forward-evaluated through our objectives.

    Surface tension reproduces the paper's AAD_DFT column closely (within
    0.05 mN/m for all three schemes). AAD_vle runs systematically higher: for
    water the paper also fitted liquid and supercritical densities over
    273-1073 K, whereas this data set is saturation-only over 280-620 K. The
    ranking across schemes (4C < 2B < 3B) is reproduced either way.
    """
    from fit_pcsaft._types import ModelSpec

    data, water = _water_data()
    spec = ModelSpec(mu=0.0, na=na, nb=nb, q=0.0)
    aad_vle, aad_sft = aad_objectives(np.array(params), water, spec, data, Units())
    print(
        f"\n{scheme}: AAD_vle={aad_vle:.2f}% (paper {paper_vle})  "
        f"AAD_sft={aad_sft:.2f} mN/m (paper {paper_sft})"
    )
    assert aad_sft == pytest.approx(paper_sft, abs=0.05)
    assert aad_vle < 2.5 * paper_vle


def test_water_scheme_ranking_matches_paper():
    """4C gives the best bulk AAD and 3B the best surface tension, as in Table 2."""
    from fit_pcsaft._types import ModelSpec

    data, water = _water_data()
    res = {}
    for scheme, params, na, nb, _, _ in WATER_SCHEMES:
        res[scheme] = aad_objectives(
            np.array(params), water, ModelSpec(mu=0.0, na=na, nb=nb, q=0.0),
            data, Units(),
        )
    assert min(res, key=lambda k: res[k][0]) == "4C"   # best bulk
    assert min(res, key=lambda k: res[k][1]) == "3B"   # best surface tension


def test_violation_is_graded_not_flat():
    """Infeasible sets must be rankable, not collapsed onto one penalty value."""
    from fit_pcsaft._pure.pareto import _evaluate_point

    data = _hexane_data_with_sft()
    good = _evaluate_point(HEXANE_P, HEXANE, HEXANE_SPEC, data, Units())
    hopeless = _evaluate_point(
        np.array([1e-6, 1e-6, 1e-6]), HEXANE, HEXANE_SPEC, data, Units()
    )
    assert good[2] <= 0.0, "a good parameter set must be feasible"
    assert hopeless[2] > 0.0, "a hopeless set must violate"
    # bounded above by the worst possible violation, so it stays comparable
    assert hopeless[2] <= 1.0


def test_lhs_sampling_selected():
    """Generation zero must use Latin hypercube coverage by default."""
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    from fit_pcsaft._pure.pareto import _initial_sampling

    assert isinstance(_initial_sampling(True), LatinHypercubeSampling)
    assert isinstance(_initial_sampling(False), FloatRandomSampling)


def test_select_picks_the_tangent_point():
    F = np.array([[1.0, 6.0], [2.0, 2.0], [6.0, 1.0]])
    # ref_vle=2%, ref_sft=1 -> costs: 6.5, 3.0, 4.0 -> index 1
    assert _argmin_scalarized(F, ref_vle=2.0, ref_sft=1.0) == 1
    # heavy weight on sft (small ref_sft) -> the low-sft corner wins
    assert _argmin_scalarized(F, ref_vle=100.0, ref_sft=0.1) == 2
