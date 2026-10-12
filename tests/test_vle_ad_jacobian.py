"""The P-only global VLE fit takes its Jacobian from feos AD; it must match
the central-difference Jacobian it replaced."""
from pathlib import Path

import numpy as np
import pytest

from fit_pcsaft._binary import vle
from fit_pcsaft._binary._utils import _make_binary_jac_fn

DATA = Path(__file__).parent.parent / "examples" / "data"


def test_ad_jacobian_matches_finite_differences(tmp_path, monkeypatch):
    lines = (DATA / "vle" / "ethanol_water_vle.csv").read_text().splitlines()
    csv = tmp_path / "ew_p_only.csv"  # every 4th row, y column (last) dropped
    csv.write_text("\n".join(line.rsplit(",", 1)[0] for line in lines[:1] + lines[1::4]))

    captured = {}

    def capture(fun, x0, jac=None, **_):
        captured.update(fun=fun, jac=jac)
        raise StopIteration

    monkeypatch.setattr(vle, "least_squares", capture)
    with pytest.raises(StopIteration):
        vle.fit_kij_vle("ethanol", "water", csv, DATA / "parameters" / "binary_params.json",
                        kij_order=1, kij_t_ref=298.15)

    fun, jac = captured["fun"], captured["jac"]
    for x in (np.array([0.0, 0.0]), np.array([-0.03, 1e-4])):
        fun(x)  # least_squares always evaluates fun at x before jac
        J_ad, J_fd = jac(x), _make_binary_jac_fn(fun, 2)(x)
        np.testing.assert_allclose(J_ad, J_fd, rtol=1e-5)
        # rows that fell back to finite differences are bit-identical; most must not have
        assert np.sum(np.any(J_ad != J_fd, axis=1)) > len(J_ad) // 2


def test_ad_failure_falls_back_to_finite_differences(tmp_path, monkeypatch):
    lines = (DATA / "vle" / "ethanol_water_vle.csv").read_text().splitlines()
    csv = tmp_path / "ew_p_only.csv"
    csv.write_text("\n".join(line.rsplit(",", 1)[0] for line in lines[:1] + lines[1::8]))

    class _Panic(BaseException):  # like pyo3's PanicException, not an Exception
        pass

    class _Panicking:
        def __getattr__(self, name):
            raise _Panic("simulated feos panic")

    captured = {}

    def capture(fun, x0, jac=None, **_):
        captured.update(fun=fun, jac=jac)
        raise StopIteration

    monkeypatch.setattr(vle, "least_squares", capture)
    with pytest.raises(StopIteration):
        vle.fit_kij_vle("ethanol", "water", csv, DATA / "parameters" / "binary_params.json",
                        kij_order=1, kij_t_ref=298.15)
    import feos
    monkeypatch.setattr(feos, "Property", _Panicking())

    fun, jac = captured["fun"], captured["jac"]
    x = np.array([-0.03, 1e-4])
    fun(x)
    np.testing.assert_array_equal(jac(x), _make_binary_jac_fn(fun, 2)(x))
