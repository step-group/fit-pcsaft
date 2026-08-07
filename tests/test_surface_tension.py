from pathlib import Path

import numpy as np
import pytest

from fit_pcsaft._csv import load_sft_csv


def test_load_sft_csv_canonical(tmp_path: Path):
    p = tmp_path / "g.csv"
    p.write_text("T,sft\n300.0,71.7\n350.0,63.2\n")
    T, g = load_sft_csv(p)
    assert np.allclose(T, [300.0, 350.0])
    assert np.allclose(g, [71.7, 63.2])


@pytest.mark.parametrize("header", ["gamma", "surface_tension", "sigma_st", "st"])
def test_load_sft_csv_aliases(tmp_path: Path, header: str):
    p = tmp_path / "g.csv"
    p.write_text(f"T,{header}\n300.0,71.7\n")
    T, g = load_sft_csv(p)
    assert np.allclose(T, [300.0]) and np.allclose(g, [71.7])


def test_load_sft_csv_missing_column(tmp_path: Path):
    p = tmp_path / "g.csv"
    p.write_text("T,psat\n300.0,3.5\n")
    with pytest.raises(ValueError):
        load_sft_csv(p)
