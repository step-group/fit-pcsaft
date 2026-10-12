"""k_ij(T) is a polynomial in kelvin, whatever unit the CSV's T column is in.

The global fits once evaluated it at the raw CSV number, so a kij_order >= 1
fit moved with temperature_unit. Same data in K and in kK must give the same
coefficients.
"""
from pathlib import Path

import numpy as np
import si_units as si

from fit_pcsaft import fit_kij_henry, fit_kij_sle, fit_kij_vle

DATA = Path(__file__).parent.parent / "examples" / "data"
BIN_PARAMS = DATA / "parameters" / "binary_params.json"
KK = si.KILO * si.KELVIN

VLE_ROWS = [  # T [K], P [kPa], x1, y1 -- ethanol/water, three isotherms
    (423.2, 808.2, 0.1803, 0.449),
    (423.2, 826.7, 0.229, 0.4867),
    (363.3, 125.7, 0.1601, 0.4717),
    (363.3, 137.5, 0.2791, 0.5328),
    (381.4, 156.1, 0.0172, 0.1495),
    (381.4, 183.0, 0.043, 0.2713),
]
SLE_ROWS = [(285.10, 0.050), (283.47, 0.115), (280.16, 0.262), (275.00, 0.433), (270.76, 0.559)]
HENRY_ROWS = [(303.0, 0.853409), (298.15, 1.16209)]


def _write(path: Path, header: str, rows, t_div: float) -> Path:
    lines = [header] + [",".join([repr(r[0] / t_div)] + [repr(v) for v in r[1:]]) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _both(tmp_path, name, header, rows, fit):
    res_k = fit(_write(tmp_path / f"{name}_K.csv", header, rows, 1.0), si.KELVIN)
    res_kk = fit(_write(tmp_path / f"{name}_kK.csv", header, rows, 1000.0), KK)
    return res_k, res_kk


def _assert_same(res_k, res_kk):
    assert abs(res_k.kij_coeffs[1]) > 1e-7  # a real slope, or the test proves nothing
    np.testing.assert_allclose(res_kk.kij_coeffs, res_k.kij_coeffs, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(res_kk.ard, res_k.ard, rtol=1e-6)


def test_vle_global_kij_order1_independent_of_temperature_unit(tmp_path):
    def fit(path, tu):
        return fit_kij_vle("ethanol", "water", path, BIN_PARAMS, kij_order=1,
                           kij_t_ref=298.15, kij_bounds=(-0.2, 0.2), temperature_unit=tu)

    _assert_same(*_both(tmp_path, "vle", "T,P,x1,y1", VLE_ROWS, fit))


def test_sle_global_kij_order1_independent_of_temperature_unit(tmp_path):
    def fit(path, tu):
        return fit_kij_sle("tetrachloromethane", "2-undecanone", path, BIN_PARAMS,
                           tm=250.77 * si.KELVIN, delta_hfus=3.273 * si.KILO * si.JOULE / si.MOL,
                           solid_index=0, tm2=285.84 * si.KELVIN,
                           delta_hfus2=34.544 * si.KILO * si.JOULE / si.MOL,
                           kij_order=1, kij_t_ref=298.0, kij_bounds=(-0.2, 0.2),
                           temperature_unit=tu)

    _assert_same(*_both(tmp_path, "sle", "T,x1", SLE_ROWS, fit))


def test_henry_kij_order1_independent_of_temperature_unit(tmp_path):
    def fit(path, tu):
        return fit_kij_henry("2-methyl-3-buten-2-ol", "water", path, BIN_PARAMS,
                             kij_order=1, kij_t_ref=298.15, kij_bounds=(-0.5, 0.5),
                             temperature_unit=tu, henry_unit=1.01325 * si.BAR)

    _assert_same(*_both(tmp_path, "henry", "T,H", HENRY_ROWS, fit))


# --- BinaryKijFitter: ard_by_type at the combined polynomial ------------------

EX_PURE = DATA / "parameters" / "examples_pure.json"
MIBK_VLE = [(387.19, 101.325, 0.9929, 0.9356), (381.17, 101.325, 0.9671, 0.767),
            (373.7, 101.325, 0.9207, 0.5845), (367.63, 101.325, 0.8678, 0.4608),
            (362.25, 101.325, 0.8165, 0.3777), (365.78, 101.325, 0.0017, 0.2352)]
MIBK_LLE = [(273.2, 0.00538, 0.912), (298.2, 0.0035, 0.893), (323.0, 0.0025, 0.8628),
            (353.3, 0.00214, 0.841)]


def test_fitter_ard_by_type_is_scored_at_the_combined_polynomial(tmp_path):
    from fit_pcsaft import BinaryKijFitter, fit_kij_lle
    from fit_pcsaft._binary.fitter import _ard_at_coeffs

    vle = _write(tmp_path / "vle.csv", "T,P,x1,y1", MIBK_VLE, 1.0)
    lle = _write(tmp_path / "lle.csv", "T,x1_I,x1_II", MIBK_LLE, 1.0)
    params = [EX_PURE, BIN_PARAMS]
    common = {"kij_order": 0, "kij_t_ref": 298.15, "kij_bounds": (-0.3, 0.3),
              "kij_per_point": True, "induced_assoc": True}

    res = (BinaryKijFitter("mibk", "water", params, kij_order=1, kij_t_ref=298.15,
                           induced_assoc=True)
           .add_vle(vle).add_lle(lle).fit())
    sub_vle = fit_kij_vle("mibk", "water", vle, params, **common)
    sub_lle = fit_kij_lle("mibk", "water", lle, params, **common)

    # The helper reproduces each module's own post-polynomial ARD ...
    assert np.isclose(_ard_at_coeffs(sub_vle, sub_vle.kij_coeffs, 298.15), sub_vle.ard, rtol=1e-9)
    assert np.isclose(_ard_at_coeffs(sub_lle, sub_lle.kij_coeffs, 298.15), sub_lle.ard, rtol=1e-6)
    # ... and ard_by_type is that ARD at the returned coefficients, not the sub-fit's.
    for t, sub in (("vle", sub_vle), ("lle", sub_lle)):
        reported = float(res.data[f"ard_{t}"][0])
        assert np.isclose(reported, _ard_at_coeffs(sub, res.kij_coeffs, 298.15), rtol=1e-9)
        assert not np.isclose(reported, sub.ard, rtol=1e-3)
