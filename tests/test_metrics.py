import numpy as np
import pytest

from fit_pcsaft._metrics import (
    Metrics,
    aggregate_metrics_from_rd,
    compute_metrics_from_arrays,
    count_weighted_aard,
)


def test_perfect_fit():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics_from_arrays(x, x)
    assert m.n == 5 and m.n_total == 5
    assert m.aard_pct == 0.0 and m.bias_pct == 0.0
    assert m.rmsd_pct == 0.0 and m.mard_pct == 0.0
    assert m.mae == 0.0 and m.rmsd == 0.0
    assert m.r2 == 1.0


def test_empty():
    m = compute_metrics_from_arrays(np.array([]), np.array([]))
    assert m.n == 0 and m.n_total == 0
    assert np.isnan(m.aard_pct) and np.isnan(m.r2)


def test_none_inputs():
    m = compute_metrics_from_arrays(None, None)
    assert m.n == 0 and np.isnan(m.aard_pct)


def test_mixed_nan():
    model = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    exp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics_from_arrays(model, exp, n_total=5)
    assert m.n == 3 and m.n_total == 5
    assert m.aard_pct == pytest.approx(0.0)
    assert m.bias_pct == pytest.approx(0.0)


def test_zero_exp_dropped():
    model = np.array([0.1, 1.0, 2.0])
    exp = np.array([0.0, 1.0, 2.0])
    m = compute_metrics_from_arrays(model, exp)
    assert m.n == 2 and m.n_total == 3


def test_zero_variance_r2_nan():
    exp = np.array([2.0, 2.0, 2.0, 2.0])
    model = np.array([2.0, 2.1, 1.9, 2.05])
    m = compute_metrics_from_arrays(model, exp)
    assert np.isnan(m.r2)
    assert np.isfinite(m.aard_pct)


def test_signed_bias():
    exp = np.array([1.0, 2.0, 3.0])
    high = exp * 1.05
    low = exp * 0.95
    assert compute_metrics_from_arrays(high, exp).bias_pct > 0
    assert compute_metrics_from_arrays(low, exp).bias_pct < 0


def test_shape_mismatch():
    with pytest.raises(ValueError):
        compute_metrics_from_arrays(np.array([1.0]), np.array([1.0, 2.0]))


def test_known_values():
    # model overshoots by 10%: RD = +10% for each point
    exp = np.array([100.0, 200.0, 300.0])
    model = exp * 1.10
    m = compute_metrics_from_arrays(model, exp)
    assert m.bias_pct == pytest.approx(10.0)
    assert m.aard_pct == pytest.approx(10.0)
    assert m.mard_pct == pytest.approx(10.0)
    assert m.rmsd_pct == pytest.approx(10.0)
    # mae = mean(|model - exp|) = mean([10, 20, 30]) = 20
    assert m.mae == pytest.approx(20.0)
    # r2: exp varies, residuals are proportional -> r2 = 1 - SS_res/SS_tot
    ss_res = (10.0**2 + 20.0**2 + 30.0**2)
    e_mean = 200.0
    ss_tot = (100.0 - 200.0)**2 + (200.0 - 200.0)**2 + (300.0 - 200.0)**2
    assert m.r2 == pytest.approx(1.0 - ss_res / ss_tot)


def test_count_weighted_aard():
    m1 = Metrics(n=10, n_total=10, bias_pct=0.0, aard_pct=2.0,
                 mard_pct=0.0, rmsd_pct=0.0, mae=0.0, rmsd=0.0, r2=1.0)
    m2 = Metrics(n=20, n_total=20, bias_pct=0.0, aard_pct=1.0,
                 mard_pct=0.0, rmsd_pct=0.0, mae=0.0, rmsd=0.0, r2=1.0)
    assert count_weighted_aard({"a": m1, "b": m2}) == pytest.approx(40.0 / 30.0)


def test_count_weighted_empty():
    assert np.isnan(count_weighted_aard({}))


def test_count_weighted_one_zero_n():
    m_zero = Metrics(n=0, n_total=5, bias_pct=float("nan"), aard_pct=float("nan"),
                     mard_pct=float("nan"), rmsd_pct=float("nan"),
                     mae=float("nan"), rmsd=float("nan"), r2=float("nan"))
    m_ok = Metrics(n=10, n_total=10, bias_pct=0.0, aard_pct=5.0,
                   mard_pct=0.0, rmsd_pct=0.0, mae=0.0, rmsd=0.0, r2=1.0)
    assert count_weighted_aard({"bad": m_zero, "good": m_ok}) == pytest.approx(5.0)


def test_aggregate_from_rd():
    rd = np.array([1.0, -2.0, 3.0, np.nan])
    m = aggregate_metrics_from_rd(rd)
    assert m.n == 3 and m.n_total == 4
    assert m.aard_pct == pytest.approx(2.0)
    assert m.bias_pct == pytest.approx(2.0 / 3.0)
    assert np.isnan(m.mae) and np.isnan(m.rmsd) and np.isnan(m.r2)


def test_aggregate_from_rd_all_nan():
    m = aggregate_metrics_from_rd(np.array([np.nan, np.nan]))
    assert m.n == 0 and m.n_total == 2
    assert np.isnan(m.aard_pct)


def test_metrics_str_zero_n():
    m = Metrics.empty(7)
    s = str(m)
    assert "n=0/7" in s


def test_metrics_str_finite():
    m = compute_metrics_from_arrays(np.array([1.05, 0.95]), np.array([1.0, 1.0]))
    s = str(m)
    assert "AARD=" in s and "RMSD=" in s and "R²=" in s
