"""Does the ``variant="de"`` win from the Forte sweep transfer to the real
objective pair, ``objectives=("vle", "sft")`` (Rehner & Gross)?

``benchmarks/knob_sweep.py`` found ``variant="de"`` clearly ahead of the
then-default ``variant="sbx"`` -- but only under ``objectives=("psat", "rho")``
(Forte et al. 2018), which never touches the DFT interface solve and is
~28x cheaper per evaluation than ``("vle", "sft")`` (see ``pareto.py``'s
``objectives`` paragraph). Nobody has checked whether that result holds once
every evaluation pays for a DFT solve. This script is that check: one knob
(``variant``), three levels, three seeds, in ``("vle", "sft")`` mode, at a
budget cut roughly 19x below the Forte sweep's baseline (pop_size=30 x
n_gen=30 x n_restarts=3 = 2700 evals/cell, vs. the Forte sweep's 51200) --
affordable only because there are 9 cells here, not 66.

**Axis meanings differ from the Forte sweep -- do not read its labels onto
this one.** Axis 0 is ``AAD_vle`` in **%** (eq 30: the mean of AARD_psat and
AARD_rho); axis 1 is ``AAD_sft`` in **mN/m** (eq 31: mean absolute surface-
tension error), not the AARD_rho percentage the Forte sweep plots there.
``REGION`` and any hand-set ``refs`` are in those units, so a Forte-sweep
region like ``[(0.0, 3.0), None]`` would mean something different here.
``refs`` is deliberately never passed to ``fit_pure_pareto`` -- its default
for this mode is ``(2.0, 0.7)``, not the ``(2.0, 2.0)`` Forte mode uses, and
letting the library apply its own per-mode default is the point.

    uv run python benchmarks/variant_sft_check.py

Resumable exactly like knob_sweep.py: every finished (level, seed) cell is
appended to variant-sft/_checkpoint.csv, and a re-run skips what is already
there. Delete that file to start over.

n_jobs=-1 (fit_pure_pareto's default) spawns a worker pool; a worker
re-imports this module, so every fitting call must stay inside main() --
module-level fitting code makes every worker start its own pool.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fit_pcsaft import fit_pure_pareto
from fit_pcsaft._pure.front_quality import front_metrics, reference_front

REPO = Path(__file__).parent.parent
DATA = REPO / "examples" / "data"
OUT = Path(__file__).parent / "variant-sft"
# Raw per-cell checkpoint -- crash recovery only, not a committed artefact
# (those are variant_sft.csv, fronts.json and the PNGs). Appended to as each
# cell finishes, so an interruption (a DFT solve gone rogue, a PubChem blip
# on a cache miss) costs only the cell in flight. See knob_sweep.py, which
# this pattern is copied from verbatim.
CHECKPOINT = OUT / "_checkpoint.csv"
# Every cell's raw front, kept for good -- so a later rescore against a
# different reference costs nothing, same reasoning as knob_sweep.py's
# fronts.json. Same row shape (knob/level/seed/seconds/F), "knob" is always
# "variant" here since there is only one.
FRONTS = OUT / "fronts.json"

BASE = dict(
    id="water",
    psat_path=DATA / "psat" / "water.csv",
    density_path=DATA / "density" / "water.csv",
    sft_path=DATA / "surface_tension" / "water.csv",  # required: "sft" is an objective
    na=1, nb=1,
    objectives=("vle", "sft"),
    bounds=[(0.8, 3.0), (2.0, 3.5), (150.0, 400.0), (1e-3, 0.35), (1500.0, 4000.0)],
    pop_size=30, n_gen=30, n_restarts=3, refine=4,
    verbose=False,
)
SEEDS = (101, 202, 303)
LEVELS = {  # level label -> fit_pure_pareto kwarg overrides; baseline marked with *
    # Explicit, not inherited: the library default moved to "de" after this ran.
    "sbx*":     {"variant": "sbx"},
    "de-cr1.0": {"variant": "de", "de_cr": 1.0},
    "de-cr0.5": {"variant": "de", "de_cr": 0.5},
}
REGION = [(0.0, 5.0), None]  # AAD_vle [%] axis; the water front spans ~1.4-8.6

_SEED_MARKERS = {SEEDS[0]: "o", SEEDS[1]: "s", SEEDS[2]: "^"}


def cells():
    for label, over in LEVELS.items():
        yield "variant", label, over


# Smallest possible check that the design hasn't silently drifted -- cheap
# (no feos/pymoo touched), fine to run at import time in a spawned worker too.
assert len(list(cells())) == 3, "cell count changed -- update this script's docstring"


def _append_checkpoint(path: Path, row: dict) -> None:
    """Append one completed cell's raw result. Writes the header on first write."""
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["knob", "level", "seed", "seconds", "F_json"])
        w.writerow([row["knob"], row["level"], row["seed"], row["seconds"], json.dumps(row["F"])])
        f.flush()


def _load_checkpoint(path: Path) -> dict[tuple[str, str, int], dict]:
    """(knob, level, seed) -> row, for whatever cells a prior run already finished."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {
            (row["knob"], row["level"], int(row["seed"])): {
                "seconds": float(row["seconds"]),
                "F": json.loads(row["F_json"]),
            }
            for row in csv.DictReader(f)
        }


def _checkpoint_roundtrip_selfcheck() -> None:
    """The smallest thing that fails if the checkpoint read/write breaks."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cp.csv"
        _append_checkpoint(path, dict(knob="k", level="l", seed=1, seconds=1.5, F=[[1.0, 2.0]]))
        _append_checkpoint(path, dict(knob="k2", level="l2", seed=2, seconds=2.5, F=[[3.0, 4.0]]))
        loaded = _load_checkpoint(path)
        assert len(loaded) == 2
        assert loaded[("k", "l", 1)] == {"seconds": 1.5, "F": [[1.0, 2.0]]}
        assert loaded[("k2", "l2", 2)] == {"seconds": 2.5, "F": [[3.0, 4.0]]}


_checkpoint_roundtrip_selfcheck()


def _clip_vle(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return F[F[:, 0] <= REGION[0][1]]


def _overlay_figure(runs: list[dict], path: Path) -> None:
    """Every level's front on shared axes; a level's 3 seeds share one colour."""
    levels = list(LEVELS)
    colors = dict(zip(levels, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in runs:
        F = _clip_vle(r["F"])
        if len(F) == 0:
            continue
        ax.scatter(
            F[:, 0], F[:, 1], s=18, alpha=0.75,
            color=colors[r["level"]], marker=_SEED_MARKERS[r["seed"]],
            label=r["level"],
        )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[level], label=level)
        for level in levels
    ]
    ax.legend(handles=handles, title="variant", fontsize=8)
    ax.set_xlabel("AAD_vle [%]")
    ax.set_ylabel("AAD_sft [mN/m]")
    ax.set_title(f"variant=de vs sbx, objectives=('vle','sft') (region: AAD_vle <= {REGION[0][1]}%; marker = seed)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_panel(df: pl.DataFrame, path: Path) -> None:
    """hv, igd_plus, spacing vs level, seed range as error bar."""
    metrics = ["hv", "igd_plus", "spacing"]
    levels = list(LEVELS)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for j, metric in enumerate(metrics):
        ax = axes[j]
        med, lo, hi = [], [], []
        for level in levels:
            vals = df.filter(pl.col("level") == level)[metric].drop_nulls().to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                med.append(np.nan); lo.append(0.0); hi.append(0.0)
                continue
            m = np.median(vals)
            med.append(m); lo.append(m - vals.min()); hi.append(vals.max() - m)
        x = np.arange(len(levels))
        ax.errorbar(x, med, yerr=[lo, hi], fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=9, rotation=30, ha="right")
        ax.set_title(metric)
    fig.suptitle("variant knob, objectives=('vle','sft'); median over 3 seeds, error bar = seed min-max")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    all_cells = list(cells())

    done = _load_checkpoint(CHECKPOINT)
    if done:
        print(f"resuming from {CHECKPOINT}: {len(done)} cell(s) already completed")

    runs = []
    for knob, label, over in all_cells:
        for seed in SEEDS:
            key = (knob, label, seed)
            if key in done:
                cached = done[key]
                F = np.asarray(cached["F"], dtype=float)
                runs.append(dict(knob=knob, level=label, seed=seed, F=F, seconds=cached["seconds"]))
                print(f"{label} seed={seed}: {len(F)} pts, {cached['seconds']:.1f}s [checkpoint]")
                continue
            t0 = time.perf_counter()
            front = fit_pure_pareto(**{**BASE, **over}, seed=seed)
            dt = time.perf_counter() - t0
            run = dict(knob=knob, level=label, seed=seed, F=front.F, seconds=dt)
            runs.append(run)
            _append_checkpoint(CHECKPOINT, {**run, "F": front.F.tolist()})
            print(f"{label} seed={seed}: {len(front.F)} pts, {dt:.1f}s")

    with open(FRONTS, "w") as f:
        json.dump(
            [
                {
                    "knob": r["knob"], "level": r["level"], "seed": r["seed"],
                    "seconds": r["seconds"], "F": np.asarray(r["F"]).tolist(),
                }
                for r in runs
            ],
            f,
        )
    print(f"wrote {FRONTS} ({len(runs)} cells)")

    # ONE shared reference front and ONE shared ref_point across all nine
    # cells -- exactly as knob_sweep.py does, per-cell values would make the
    # columns incomparable.
    ref = reference_front(*[r["F"] for r in runs], region=REGION)
    ref_point = np.max(np.vstack([r["F"] for r in runs]), axis=0) * 1.1

    rows = [
        {k: r[k] for k in ("knob", "level", "seed")}
        | {"seconds": r["seconds"]}
        | front_metrics(r["F"], reference=ref, ref_point=ref_point, region=REGION)
        for r in runs
    ]
    df = pl.DataFrame(rows)
    csv_path = OUT / "variant_sft.csv"
    with open(csv_path, "w") as f:
        f.write(
            "# water Rehner & Gross front (objectives=('vle','sft')) variant check -- "
            "does knob_sweep.py's variant='de' win transfer out of Forte mode?\n"
        )
        f.write(
            f"# pop_size={BASE['pop_size']} n_gen={BASE['n_gen']} n_restarts={BASE['n_restarts']} "
            "refine=4, a deliberately reduced budget (~2700 evals/cell) -- "
            "see variant-sft-report.md for the runtime this bought\n"
        )
        f.write(
            "# axis 0 is AAD_vle [%] (eq 30), axis 1 is AAD_sft [mN/m] (eq 31) -- "
            "NOT the Forte sweep's AARD_psat/AARD_rho axes\n"
        )
        f.write(
            "# every metric column is region-clipped to AAD_vle in [0, 5]% (AAD_sft "
            "unclipped) against one reference front and one ref_point shared across "
            "all nine cells\n"
        )
        f.write(
            "# refs was left at fit_pure_pareto's default for this mode, (2.0, 0.7) -- "
            "not the (2.0, 2.0) Forte mode uses -- and is not a column here\n"
        )
        f.write("# read with pl.read_csv(path, comment_prefix='#')\n")
        df.write_csv(f)
    print(f"wrote {csv_path} ({len(df)} rows)")

    _overlay_figure(runs, OUT / "overlay_variant_sft.png")
    _metric_panel(df, OUT / "metric_panel_sft.png")
    print(f"wrote figures to {OUT}")

    CHECKPOINT.unlink(missing_ok=True)  # clean finish: no need to keep the raw checkpoint


if __name__ == "__main__":
    main()
