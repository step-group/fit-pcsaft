"""Canonical fit-quality metrics. Pure-numpy; no project imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Metrics:
    """Fit-quality summary for one property (e.g. psat, rho, vle_P).

    All percent fields are in %. ``mae`` and ``rmsd`` are in the property's
    own physical units (kPa, kg/m³, mol/mol, Pa·s, …). ``r2`` is dimensionless.

    All fields are NaN when the metric cannot be defined (e.g. n=0, or
    zero-variance reference data for r2).

    RD% sign convention: positive = model overshoots experiment.
    """

    n: int
    n_total: int
    bias_pct: float    # mean signed RD%
    aard_pct: float    # mean |RD%|   (AARD — Average Absolute Relative Deviation)
    mard_pct: float    # max  |RD%|   (MARD — Maximum ARD)
    rmsd_pct: float    # sqrt(mean(RD%²))
    mae: float         # mean |model − exp|  [original units]
    rmsd: float        # sqrt(mean((model − exp)²))  [original units]
    r2: float          # coefficient of determination; NaN when SS_tot == 0

    @classmethod
    def empty(cls, n_total: int = 0) -> "Metrics":
        nan = float("nan")
        return cls(
            n=0, n_total=int(n_total),
            bias_pct=nan, aard_pct=nan, mard_pct=nan, rmsd_pct=nan,
            mae=nan, rmsd=nan, r2=nan,
        )

    def __str__(self) -> str:
        if self.n == 0:
            return f"n=0/{self.n_total} (no valid points)"
        r2_str = f"{self.r2:.4f}" if np.isfinite(self.r2) else "N/A"
        return (
            f"AARD={self.aard_pct:.2f}%  RMSD={self.rmsd_pct:.2f}%  "
            f"bias={self.bias_pct:+.2f}%  MaxARD={self.mard_pct:.2f}%  "
            f"R²={r2_str}  n={self.n}/{self.n_total}"
        )


def compute_metrics_from_arrays(
    model,
    exp,
    *,
    n_total: Optional[int] = None,
) -> Metrics:
    """Compute all metrics from aligned model and experimental arrays.

    Drops rows where any of (model, exp) is non-finite or where exp == 0
    (relative deviation undefined). ``n_total`` defaults to ``len(exp)`` before
    filtering and is reported as-is for transparency.

    r2 is NaN when SS_tot == 0 (constant reference, including the n=1 case).
    """
    if model is None or exp is None:
        return Metrics.empty(n_total or 0)

    model = np.asarray(model, dtype=float)
    exp = np.asarray(exp, dtype=float)

    if model.shape != exp.shape:
        raise ValueError(f"shape mismatch: model {model.shape} vs exp {exp.shape}")

    n_total = int(n_total if n_total is not None else len(exp))
    if n_total == 0:
        return Metrics.empty(0)

    finite = np.isfinite(model) & np.isfinite(exp) & (exp != 0.0)
    m, e = model[finite], exp[finite]
    n = int(m.size)
    if n == 0:
        return Metrics.empty(n_total)

    abs_resid = m - e
    rd = abs_resid / e * 100.0

    e_mean = float(np.mean(e))
    ss_tot = float(np.sum((e - e_mean) ** 2))
    r2 = (
        float("nan")
        if ss_tot == 0.0
        else 1.0 - float(np.sum(abs_resid ** 2)) / ss_tot
    )

    return Metrics(
        n=n,
        n_total=n_total,
        bias_pct=float(np.mean(rd)),
        aard_pct=float(np.mean(np.abs(rd))),
        mard_pct=float(np.max(np.abs(rd))),
        rmsd_pct=float(np.sqrt(np.mean(rd * rd))),
        mae=float(np.mean(np.abs(abs_resid))),
        rmsd=float(np.sqrt(np.mean(abs_resid * abs_resid))),
        r2=r2,
    )


def aggregate_metrics_from_rd(
    rd_pct,
    *,
    n_total: Optional[int] = None,
) -> Metrics:
    """Compute percent-based metrics from a precomputed RD% array (NaN-aware).

    Use when raw model/exp arrays are unavailable (e.g. log-space viscosity
    fallback). ``mae``, ``rmsd``, and ``r2`` are always NaN in this path because
    physical-unit residuals are not available.
    """
    rd = np.asarray(rd_pct, dtype=float)
    n_total = int(n_total if n_total is not None else len(rd))
    valid = rd[np.isfinite(rd)]
    n = int(valid.size)
    if n == 0:
        return Metrics.empty(n_total)

    nan = float("nan")
    return Metrics(
        n=n,
        n_total=n_total,
        bias_pct=float(np.mean(valid)),
        aard_pct=float(np.mean(np.abs(valid))),
        mard_pct=float(np.max(np.abs(valid))),
        rmsd_pct=float(np.sqrt(np.mean(valid * valid))),
        mae=nan,
        rmsd=nan,
        r2=nan,
    )


def count_weighted_aard(metrics: dict) -> float:
    """Pool AARD% across per-property panels by valid-point count.

    Returns NaN when every panel has n == 0 or the dict is empty.
    """
    num, den = 0.0, 0
    for m in metrics.values():
        if m.n > 0 and np.isfinite(m.aard_pct):
            num += m.aard_pct * m.n
            den += m.n
    return float("nan") if den == 0 else num / den
