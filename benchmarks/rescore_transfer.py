"""Re-score a finished ``transfer_check.py`` ladder inside the region of interest.

``transfer_check.py`` scores every cell unclipped (``REGION=None``), because
nothing is known in advance about where a new compound's front will land. That
is the honest choice at run time and the wrong one at read time: on camphor the
unclipped metrics were dominated by tail points reaching ``AARD_rho`` 235%, and
a reading taken off them ("SBX disjointly better at 2.7k") did not survive the
clip. Re-scoring is free -- ``fronts.json`` keeps every cell's raw front.

Everything here is computed against ONE shared reference front and ONE shared
reference point built from all 18 cells, so the columns are comparable across
operators and rungs. Both are rebuilt *after* the clip, not before.

**Read ``coverage`` and ``hv`` together, and distrust them separately.** They
disagreed on camphor: ``hv`` said parity, pooled ``coverage`` said DE led
0.851 to 0.260. ``hv`` is the weak one when the shared reference point sits far
outside the region -- it is then a huge rectangle in which the elbow detail is a
rounding error. ``coverage`` is sensitive to exactly that detail, but it is also
the metric a single stray point inverted once before. Neither is the answer on
its own; the per-seed spread below is there so a one-seed claim is visibly
unsafe.

    uv run python benchmarks/rescore_transfer.py carvone
    uv run python benchmarks/rescore_transfer.py camphor
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Run as a script, sys.path[0] is benchmarks/ and the repo root is not on it, so
# `benchmarks._problems` does not resolve. pytest adds the root itself, which is
# why the tests pass either way -- put it on the path here so both work.
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks._problems import PROBLEMS  # noqa: E402
from fit_pcsaft._pure.front_quality import front_metrics, reference_front  # noqa: E402
from fit_pcsaft._pure.pareto import coverage  # noqa: E402

COMPOUND = sys.argv[1] if len(sys.argv) > 1 else "camphor"
if COMPOUND not in PROBLEMS:
    raise SystemExit(f"unknown problem {COMPOUND!r}; use one of {sorted(PROBLEMS)}")
PROBLEM = PROBLEMS[COMPOUND]
OUT = Path(__file__).parent / f"{COMPOUND}-transfer"
# Per-axis, from the registry. A single scalar cap was fine while both axes were
# AARD in %; thymol's axis 1 is AAD_sft in mN/m, where a 10.0 cap would silently
# delete most of the front rather than clip its tail.
CAP = PROBLEM.rescore_cap
COL = {"sbx": "#1f77b4", "de-cr1.0": "#ff7f0e"}
RUNGS = [2700, 10800, 28800]


def _clip(F) -> np.ndarray:
    """Keep rows inside `CAP`. `None` on an axis leaves that axis unclipped."""
    F = np.asarray(F, dtype=float)
    keep = np.ones(len(F), dtype=bool)
    for axis, hi in enumerate(CAP):
        if hi is not None:
            keep &= F[:, axis] <= hi
    return F[keep]


def _eq32(F, refs) -> float:
    """Rehner & Gross eq 32 at the front's own tangent, on the UNCLIPPED front.

    Production's select() sees the whole front, so clipping first would report a
    number no TESIS run can reproduce. This is _argmin_scalarized's objective:
    F[:, 0]/refs[0] + F[:, 1]/refs[1].
    """
    F = np.asarray(F, dtype=float)
    return float(np.min(F[:, 0] / refs[0] + F[:, 1] / refs[1]))


def _cap_label() -> str:
    parts = [
        f"{lab} <= {hi:g}"
        for lab, hi in zip(PROBLEM.axis_labels, CAP)
        if hi is not None
    ]
    return ", ".join(parts) if parts else "no clip"


def main() -> None:
    rows = json.load(open(OUT / "fronts.json"))
    ops = sorted({r["op"] for r in rows}, key=lambda o: o != "sbx")

    clipped = {(r["op"], r["evals"], r["seed"]): _clip(r["F"]) for r in rows}
    non_empty = [F for F in clipped.values() if len(F)]
    if not non_empty:
        raise SystemExit(f"every cell is empty at {_cap_label()} -- raise rescore_cap")

    # Shared yardstick, built after the clip. ref_point from the clipped max so
    # hv is not swamped by the region the clip just excluded.
    ref = reference_front(*non_empty)
    ref_point = np.max(np.vstack(non_empty), axis=0) * 1.1

    print(f"=== {COMPOUND}: clipped to {_cap_label()} ===")
    for ev in RUNGS:
        print(f"\n--- {ev:,} evals/cell ---")
        for op in ops:
            for (o, e, seed), F in sorted(clipped.items()):
                if o != op or e != ev:
                    continue
                if len(F) == 0:
                    print(f"  {op:9s} seed={seed}: empty after clip")
                    continue
                m = front_metrics(F, reference=ref, ref_point=ref_point)
                print(
                    f"  {op:9s} seed={seed}: hv {m['hv']:10.2f}  "
                    f"igd+ {m['igd_plus']:7.4f}  spacing {m['spacing']:7.4f}  "
                    f"n={int(m['n_points']):4d}"
                )
        # Pooled over seeds: the per-seed spread is large for both operators, so
        # the pairwise question is only meaningful on the pooled sets.
        pooled = {}
        for op in ops:
            parts = [F for (o, e, _), F in clipped.items() if o == op and e == ev and len(F)]
            pooled[op] = np.vstack(parts) if parts else np.empty((0, 2))
        a, b = ops[0], ops[-1]
        if len(pooled[a]) and len(pooled[b]):
            print(
                f"  pooled: coverage({a}->{b}) {coverage(pooled[a], pooled[b]):.3f}   "
                f"coverage({b}->{a}) {coverage(pooled[b], pooled[a]):.3f}"
            )
            for axis, name in enumerate(PROBLEM.axis_labels):
                print(
                    f"  best {name}: "
                    + "   ".join(f"{op} {pooled[op][:, axis].min():.3f}" for op in ops)
                )

        # eq 32 on the UNCLIPPED fronts -- production's select() sees the whole
        # front, so a clipped number would be one no TESIS run can reproduce.
        if PROBLEM.select_refs is not None:
            for op in ops:
                vals = [
                    _eq32(r["F"], PROBLEM.select_refs)
                    for r in rows if r["op"] == op and r["evals"] == ev
                ]
                if vals:
                    print(
                        f"  eq32 ({op}): best {min(vals):.4f}  "
                        + "[" + " ".join(f"{v:.4f}" for v in sorted(vals)) + "]"
                    )
            if PROBLEM.baseline_eq32 is not None:
                print(
                    f"  committed fit eq32: {PROBLEM.baseline_eq32:.4f} "
                    "(96 000 evals, sbx -- a different budget, see the plan's Risk 3)"
                )

    fig, axes = plt.subplots(
        1, len(RUNGS), figsize=(5 * len(RUNGS), 4.6), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, ev in zip(axes, RUNGS):
        for op in ops:
            first = True
            for (o, e, _), F in sorted(clipped.items()):
                if o != op or e != ev or len(F) == 0:
                    continue
                F = F[np.argsort(F[:, 0])]
                ax.plot(
                    F[:, 0], F[:, 1], "o-", ms=4, lw=0.9, alpha=0.8,
                    color=COL.get(op), label=op if first else None,
                )
                first = False
        ax.set_title(f"{ev:,} evals/cell")
        ax.set_xlabel(PROBLEM.axis_labels[0])
        # An unclipped axis has no fixed upper limit to set -- let matplotlib
        # scale it to the clipped data instead of forcing a wrong one.
        if CAP[0] is not None:
            ax.set_xlim(0, CAP[0])
        if CAP[1] is not None:
            ax.set_ylim(0, CAP[1])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(PROBLEM.axis_labels[1])
    axes[0].legend(loc="upper right")
    fig.suptitle(
        f"{COMPOUND} fronts, zoomed to {_cap_label()} "
        f"({len({s for _, _, s in clipped})} seeds per operator)"
    )
    fig.tight_layout()
    path = OUT / "zoom_low_aard.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
