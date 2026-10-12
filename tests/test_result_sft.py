"""Surface tension in per-point result outputs: to_csv and sft_options."""

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fit_pcsaft import eval_pure
from fit_pcsaft._pure.surface_tension import (
    SurfaceTensionOptions,
    predict_surface_tension,
)

DATA = Path(__file__).parent.parent / "examples" / "data"
# Rehner & Gross 2020 Table 1, water PC-SAFT 2B
WATER_RG = {"m": 1.0, "sigma": 2.9375, "epsilon_k": 272.03,
            "kappa_ab": 0.044480, "epsilon_k_ab": 3125.3}


@pytest.fixture(scope="module")
def water():
    return eval_pure(
        id="water", params=WATER_RG, na=1, nb=1,
        psat_path=DATA / "psat" / "water.csv",
        density_path=DATA / "density" / "water.csv",
        sft_path=DATA / "surface_tension" / "water.csv",
    )


def test_to_csv_fills_sft_model_column(water, tmp_path):
    water.to_csv(tmp_path)
    sft = pl.read_csv(tmp_path / "water_exp.csv").filter(pl.col("property") == "sft")
    assert len(sft) == len(water.data.T_sft) > 0
    assert np.isfinite(sft["model"].to_numpy()).all()


def test_residuals_predict_with_the_results_sft_options(water):
    # n_grid=32 differs from the default grid (and fails at 10 of 25 points here).
    opts = SurfaceTensionOptions(n_grid=32)
    custom = dataclasses.replace(water, sft_options=opts)

    got = custom.residuals().filter(pl.col("property") == "sft")["model"].to_numpy()
    want = predict_surface_tension(water.functional, water.data.T_sft, water.units, opts)
    default = water.residuals().filter(pl.col("property") == "sft")["model"].to_numpy()

    np.testing.assert_array_equal(got, want)
    assert not np.allclose(got, default, equal_nan=True)


def test_eval_pure_carries_sft_options_into_metrics_and_residuals():
    opts = SurfaceTensionOptions(n_grid=32)
    r = eval_pure(
        id="water", params=WATER_RG, na=1, nb=1,
        psat_path=DATA / "psat" / "water.csv",
        density_path=DATA / "density" / "water.csv",
        sft_path=DATA / "surface_tension" / "water.csv",
        sft_options=opts,
    )
    want = predict_surface_tension(r.functional, r.data.T_sft, r.units, opts)
    got = r.residuals().filter(pl.col("property") == "sft")["model"].to_numpy()
    assert r.sft_options is opts
    np.testing.assert_array_equal(got, want)
    assert r.metrics["sft"].n == np.isfinite(want).sum()
