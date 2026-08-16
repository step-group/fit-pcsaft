"""Task 5 Step 6: does polish actually improve a real Pareto front?

Fits the same water Forte front (``objectives=("psat", "rho")``) twice at one
seed -- once with ``polish=False``, once with ``polish=True`` -- so the two
runs share the identical unpolished MOEA/D search (``polish_front`` sits
strictly after ``refine``/``_densify`` in ``pareto.py`` and uses a
hard-coded internal RNG only when ``n_starts>1``, which ``fit_pure_pareto``
never sets, so nothing before the polish step depends on ``polish``). Scores
both with ``front_metrics`` against one shared reference front and one
shared ``ref_point`` (per-call values would make the two rows
incomparable), and reports how many points actually moved in parameter
space. Draws no conclusion -- see task-5-brief.md Step 8: a human decides
whether polish earns a default.

Same conventions as knob_sweep.py: every fitting call stays inside main()
because n_jobs=-1 spawns a worker pool and a worker re-imports this module.

    uv run python benchmarks/polish_check.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fit_pcsaft import fit_pure_pareto
from fit_pcsaft._pure.front_quality import front_metrics, reference_front

REPO = Path(__file__).parent.parent
DATA = REPO / "examples" / "data"
OUT = Path(__file__).parent / "knob-sweep"

BASE = dict(
    id="water",
    psat_path=DATA / "psat" / "water.csv",
    density_path=DATA / "density" / "water.csv",
    na=1, nb=1, objectives=("psat", "rho"),
    bounds=[(0.8, 3.0), (2.0, 3.5), (150.0, 400.0), (1e-3, 0.35), (1500.0, 4000.0)],
    pop_size=80, n_gen=80, n_restarts=8, refine=4, verbose=False, seed=101,
)
REGION = [(0.0, 3.0), None]  # same region knob_sweep.py used: AARD_psat <= 3%

# "Small relative tolerance" in parameter space: each of the 5 parameters has
# a very different scale (kappa_ab's bounds alone span 1e-3 to 0.35), so the
# change is normalized by that parameter's own bounds span -- the same
# per-axis normalization pareto.py's _densify and front_quality._max_gap use
# for exactly this multi-scale problem, reused here rather than reinvented.
# 1e-3 of a parameter's range is far above SLSQP's ftol=1e-12 solve noise, so
# it separates a real move from float-identical copies (untouched points are
# copied verbatim by polish_front, distance exactly 0.0) without flagging
# numerical dust as "moved".
_MOVE_TOL = 1e-3


def _moved_stats(X_before: np.ndarray, X_after: np.ndarray, bounds: list[tuple[float, float]]):
    """For each polished point, match it to its nearest `before` point (span-
    normalized Euclidean, since an untouched point is an exact copy and a
    polished one stays close to its own warm start) and report how many
    moved by more than `_MOVE_TOL` in any single parameter, plus the median
    of each moved point's largest per-parameter relative change.
    """
    span = np.array([hi - lo for lo, hi in bounds])
    Xb = np.asarray(X_before, dtype=float)
    Xa = np.asarray(X_after, dtype=float)
    d = np.linalg.norm((Xa[:, None, :] - Xb[None, :, :]) / span, axis=2)
    nearest = Xb[np.argmin(d, axis=1)]

    rel = np.abs(Xa - nearest) / span
    per_point = rel.max(axis=1)  # each point's single largest-moving parameter
    moved = per_point > _MOVE_TOL
    n_moved = int(moved.sum())
    med_rel = float(np.median(per_point[moved])) if n_moved else float("nan")
    return n_moved, med_rel


def _overlay(F_before: np.ndarray, F_after: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for F, label, color in (
        (F_before, "before (polish=False)", "tab:blue"),
        (F_after, "after (polish=True)", "tab:orange"),
    ):
        F = np.asarray(F, dtype=float)
        F = F[F[:, 0] <= REGION[0][1]]
        ax.scatter(F[:, 0], F[:, 1], s=22, alpha=0.8, color=color, label=label)
    ax.set_xlabel("AARD_psat [%]")
    ax.set_ylabel("AARD_rho [%]")
    ax.set_title("water Forte front, polish before/after (region: AARD_psat <= 3%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    try:
        t0 = time.perf_counter()
        front_before = fit_pure_pareto(**BASE, polish=False)
        t_before = time.perf_counter() - t0
    except Exception as exc:
        print(
            f"baseline fit failed at startup ({type(exc).__name__}: {exc}). "
            "If this is a PubChem lookup failure, PubChem is flapping -- "
            "stop and report rather than retrying."
        )
        raise SystemExit(1)
    print(f"before (polish=False): {len(front_before.F)} pts, {t_before:.1f}s")

    t0 = time.perf_counter()
    front_after = fit_pure_pareto(**BASE, polish=True)
    t_after = time.perf_counter() - t0
    print(f"after  (polish=True):  {len(front_after.F)} pts, {t_after:.1f}s")

    # ONE shared reference front and ONE shared ref_point for both rows --
    # per-call values would make them incomparable (front_quality.py, HV note).
    ref = reference_front(front_before.F, front_after.F, region=REGION)
    ref_point = np.max(np.vstack([front_before.F, front_after.F]), axis=0) * 1.1

    m_before = front_metrics(front_before.F, reference=ref, ref_point=ref_point, region=REGION)
    m_after = front_metrics(front_after.F, reference=ref, ref_point=ref_point, region=REGION)

    n_moved, med_rel = _moved_stats(front_before.X, front_after.X, BASE["bounds"])

    print(f"before: {m_before}")
    print(f"after:  {m_after}")
    print(
        f"moved: {n_moved}/{len(front_after.X)} polished points moved by more than "
        f"{_MOVE_TOL:g} (bounds-span-normalized) in some parameter; median relative "
        f"change among those (largest-moving parameter per point): {med_rel:.4g}"
    )

    rows = [
        {"run": "before", "polish": False, "seconds": t_before, **m_before},
        {"run": "after", "polish": True, "seconds": t_after, **m_after},
    ]
    csv_path = OUT / "polish_check.csv"
    with open(csv_path, "w") as f:
        f.write(
            "# water Forte front (objectives=('psat','rho')), seed=101, "
            "polish=False vs polish=True, same baseline as knob_sweep.py's "
            "80x80* cell -- see task-5-step6-report.md\n"
        )
        f.write(
            "# hv/igd_plus/spacing/max_gap are region-clipped to AARD_psat in [0, 3]% "
            "against one reference front (reference_front(F_before, F_after, "
            "region=REGION)) and one ref_point shared by both rows\n"
        )
        f.write(
            f"# moved-point check (not a column, both rows describe the same "
            f"comparison): {n_moved}/{len(front_after.X)} polished points moved by "
            f"more than {_MOVE_TOL:g} bounds-span-normalized in some parameter; "
            f"median relative change among them = {med_rel:.6g}\n"
        )
        f.write("# read with pl.read_csv(path, comment_prefix='#')\n")
        pl.DataFrame(rows).write_csv(f)
    print(f"wrote {csv_path}")

    _overlay(front_before.F, front_after.F, OUT / "polish.png")
    print(f"wrote {OUT / 'polish.png'}")


if __name__ == "__main__":
    main()
