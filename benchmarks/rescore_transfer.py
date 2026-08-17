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

from fit_pcsaft._pure.front_quality import front_metrics, reference_front  # noqa: E402
from fit_pcsaft._pure.pareto import coverage  # noqa: E402

COMPOUND = sys.argv[1] if len(sys.argv) > 1 else "camphor"
OUT = Path(__file__).parent / f"{COMPOUND}-transfer"
CAP = 10.0  # both axes, in AARD %; the region anything downstream would use
COL = {"sbx": "#1f77b4", "de-cr1.0": "#ff7f0e"}
RUNGS = [2700, 10800, 28800]


def _clip(F) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return F[(F[:, 0] <= CAP) & (F[:, 1] <= CAP)]


def main() -> None:
    rows = json.load(open(OUT / "fronts.json"))
    ops = sorted({r["op"] for r in rows}, key=lambda o: o != "sbx")

    clipped = {(r["op"], r["evals"], r["seed"]): _clip(r["F"]) for r in rows}
    non_empty = [F for F in clipped.values() if len(F)]
    if not non_empty:
        raise SystemExit(f"every cell is empty at AARD <= {CAP}% -- raise CAP")

    # Shared yardstick, built after the clip. ref_point from the clipped max so
    # hv is not swamped by the region the clip just excluded.
    ref = reference_front(*non_empty)
    ref_point = np.max(np.vstack(non_empty), axis=0) * 1.1

    print(f"=== {COMPOUND}: clipped to AARD <= {CAP:g}% on both axes ===")
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
            for axis, name in ((0, "AARD_psat"), (1, "AARD_rho")):
                print(
                    f"  best {name}: "
                    + "   ".join(f"{op} {pooled[op][:, axis].min():.3f}%" for op in ops)
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
        ax.set_xlabel("AARD_psat [%]")
        ax.set_xlim(0, CAP)
        ax.set_ylim(0, CAP)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AARD_rho [%]")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"{COMPOUND} fronts, zoomed to AARD <= {CAP:g} % (3 seeds per operator)")
    fig.tight_layout()
    path = OUT / "zoom_low_aard.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
