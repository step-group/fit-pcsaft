"""Knob sweep for MOEA/D on the water Forte front. Writes CSV + figures +
fronts.json (every cell's raw front, for rescore.py); draws no conclusions --
a human reads the figures and gives the verdict (Task 4 is blocked on that
reading, see .superpowers/sdd/.../task-3-brief.md).

One-factor-at-a-time around a fixed baseline (pop_size=80, n_gen=80,
n_restarts=8, refine=4, objectives=("psat", "rho")) -- not a full factorial:
6 knobs at 3-4 levels each, 22 cells total, x3 seeds. Only ("psat", "rho") is
affordable for a sweep at all -- see pareto.py's ``objectives`` paragraph for
why the default ("vle", "sft") pair is ~28x more expensive per evaluation.

    uv run python benchmarks/knob_sweep.py --pilot     # time 2 cells x1 seed, then exit
    uv run python benchmarks/knob_sweep.py              # full sweep: CSV + figures

A full sweep is resumable: every finished cell is appended to
knob-sweep/_checkpoint.csv, and re-running skips the (knob, level, seed)
triples already in it. Delete that file to start over.

n_jobs=-1 (fit_pure_pareto's default) spawns a worker pool; a worker
re-imports this module, so every fitting call must stay inside main() --
module-level fitting code makes every worker start its own pool.
"""

from __future__ import annotations

import argparse
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
OUT = Path(__file__).parent / "knob-sweep"
# Raw per-cell checkpoint -- crash recovery only, NOT one of the committed
# artefacts (those are water_forte_knobs.csv, fronts.json and the PNGs).
# Appended to as each cell's fit finishes, so a crash mid-sweep (e.g. a
# PubChem blip on a cache miss) costs only the cell in flight: the next
# invocation reloads it and skips whatever is already there.
# water_forte_knobs.csv itself still needs every front gathered first
# (front_metrics' hv/igd_plus need the one shared reference front/ref_point
# built after all runs complete -- per-cell values would make the columns
# incomparable), so it stays a single batch write at the end; this
# checkpoint is what actually protects the expensive part, the ~15s/cell
# fit_pure_pareto call. Deleted automatically once the sweep finishes clean.
CHECKPOINT = OUT / "_checkpoint.csv"
# Every cell's raw front, kept for good -- unlike the checkpoint, this is
# committed. The fitting is the expensive part (~16 min total); hv/igd_plus/
# spacing are cheap derived values that depend on one particular
# reference-front choice, and re-deriving them against a different reference
# (e.g. excluding one variant from the yardstick, see rescore.py) should
# never require a refit. One JSON array, written once at the end next to
# water_forte_knobs.csv -- not JSONL, because nothing writes it
# incrementally the way the checkpoint is appended to; a plain array is the
# simplest thing that round-trips with one json.load().
FRONTS = OUT / "fronts.json"

BASE = dict(
    id="water",
    psat_path=DATA / "psat" / "water.csv",
    density_path=DATA / "density" / "water.csv",
    na=1, nb=1, objectives=("psat", "rho"),
    bounds=[(0.8, 3.0), (2.0, 3.5), (150.0, 400.0), (1e-3, 0.35), (1500.0, 4000.0)],
    pop_size=80, n_gen=80, n_restarts=8, refine=4, verbose=False,
)
SEEDS = (101, 202, 303)
REGION = [(0.0, 3.0), None]  # AARD_psat <= 3%, the region of interest

LEVELS = {  # knob -> {level label: override value}; baseline marked with *
    "n_neighbors": {"5": 5, "10": 10, "20*": 20, "40": 40},
    "prob_neighbor_mating": {"0.5": 0.5, "0.7": 0.7, "0.9*": 0.9, "1.0": 1.0},
    "n_replace": {"1": 1, "2*": 2, "5": 5, "20": 20},
}
# decomposition, variant and the budget shape don't map to a single kwarg,
# so they get their own explicit override dicts instead of living in LEVELS.
EXTRA = {
    "decomposition": {
        "tchebi*": {}, "pbi-1": {"decomposition": "pbi", "pbi_theta": 1.0},
        "pbi-5": {"decomposition": "pbi", "pbi_theta": 5.0},
        "pbi-10": {"decomposition": "pbi", "pbi_theta": 10.0},
    },
    "variant": {
        "sbx*": {}, "de-cr1.0": {"variant": "de", "de_cr": 1.0},
        "de-cr0.5": {"variant": "de", "de_cr": 0.5},
    },
    "budget_shape": {
        "40x160": {"pop_size": 40, "n_gen": 160},
        "80x80*": {},
        "160x40": {"pop_size": 160, "n_gen": 40},
    },
}


def cells():
    for knob, levels in LEVELS.items():
        for label, value in levels.items():
            yield knob, label, {knob: value}
    for knob, levels in EXTRA.items():
        for label, over in levels.items():
            yield knob, label, over


# Smallest possible check that the sweep design hasn't silently drifted --
# cheap (no feos/pymoo touched), so it is fine to run at import time in a
# spawned worker too.
assert len(list(cells())) == 22, "cell count changed -- update this script's docstring"


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


_KNOB_ORDER = list(LEVELS) + list(EXTRA)
_LEVEL_ORDER = {knob: list(levels) for knob, levels in {**LEVELS, **EXTRA}.items()}
_SEED_MARKERS = {SEEDS[0]: "o", SEEDS[1]: "s", SEEDS[2]: "^"}


def _clip_psat(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return F[F[:, 0] <= REGION[0][1]]


def _overlay_figure(knob: str, runs: list[dict], path: Path) -> None:
    """Every level's front on shared axes; a level's 3 seeds share one colour."""
    levels = _LEVEL_ORDER[knob]
    colors = dict(zip(levels, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in runs:
        if r["knob"] != knob:
            continue
        F = _clip_psat(r["F"])
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
    ax.legend(handles=handles, title=knob, fontsize=8)
    ax.set_xlabel("AARD_psat [%]")
    ax.set_ylabel("AARD_rho [%]")
    ax.set_title(f"knob={knob}  (region: AARD_psat <= {REGION[0][1]}%; marker = seed)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_panel(df: pl.DataFrame, path: Path) -> None:
    """hv, igd_plus, spacing vs level, one row per knob, seed range as error bar."""
    metrics = ["hv", "igd_plus", "spacing"]
    fig, axes = plt.subplots(
        len(_KNOB_ORDER), len(metrics), figsize=(4 * len(metrics), 3 * len(_KNOB_ORDER)),
        squeeze=False,
    )
    for i, knob in enumerate(_KNOB_ORDER):
        sub = df.filter(pl.col("knob") == knob)
        levels = _LEVEL_ORDER[knob]
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            med, lo, hi = [], [], []
            for level in levels:
                vals = sub.filter(pl.col("level") == level)[metric].drop_nulls().to_numpy()
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    med.append(np.nan); lo.append(0.0); hi.append(0.0)
                    continue
                m = np.median(vals)
                med.append(m); lo.append(m - vals.min()); hi.append(vals.max() - m)
            x = np.arange(len(levels))
            ax.errorbar(x, med, yerr=[lo, hi], fmt="o", capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, fontsize=8, rotation=30, ha="right")
            if i == 0:
                ax.set_title(metric)
            if j == 0:
                ax.set_ylabel(knob, fontsize=9)
    fig.suptitle("median over 3 seeds; error bar = seed min-max range")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _budget_shape_figure(runs: list[dict], df: pl.DataFrame, path: Path) -> None:
    """40x160 vs 80x80 vs 160x40 at fixed evaluations: front overlay + wall time."""
    fig, (ax_front, ax_time) = plt.subplots(1, 2, figsize=(11, 5))
    levels = _LEVEL_ORDER["budget_shape"]
    colors = dict(zip(levels, plt.get_cmap("tab10").colors))
    for r in runs:
        if r["knob"] != "budget_shape":
            continue
        F = _clip_psat(r["F"])
        if len(F) == 0:
            continue
        ax_front.scatter(
            F[:, 0], F[:, 1], s=18, alpha=0.75,
            color=colors[r["level"]], marker=_SEED_MARKERS[r["seed"]], label=r["level"],
        )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[level], label=level)
        for level in levels
    ]
    ax_front.legend(handles=handles, fontsize=8)
    ax_front.set_xlabel("AARD_psat [%]")
    ax_front.set_ylabel("AARD_rho [%]")
    ax_front.set_title("front (region: AARD_psat <= 3%; marker = seed)")

    sub = df.filter(pl.col("knob") == "budget_shape")
    med, lo, hi = [], [], []
    for level in levels:
        vals = sub.filter(pl.col("level") == level)["seconds"].to_numpy()
        m = np.median(vals)
        med.append(m); lo.append(m - vals.min()); hi.append(vals.max() - m)
    x = np.arange(len(levels))
    ax_time.errorbar(x, med, yerr=[lo, hi], fmt="o", capsize=4)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(levels)
    ax_time.set_ylabel("wall time [s]")
    ax_time.set_title("same pop_size * n_gen product; seed min-max range")

    fig.suptitle("budget shape: pop_size x n_gen at fixed evaluations")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_figures(runs: list[dict], df: pl.DataFrame) -> None:
    for knob in _KNOB_ORDER:
        _overlay_figure(knob, runs, OUT / f"overlay_{knob}.png")
    _metric_panel(df, OUT / "metric_panel.png")
    _budget_shape_figure(runs, df, OUT / "budget_shape.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot", action="store_true",
        help="time one baseline cell and one non-baseline cell at a single seed, "
             "print a projection for the full sweep, and exit -- no CSV, no figures",
    )
    parser.add_argument(
        "--n-restarts", type=int, default=BASE["n_restarts"],
        help="override the sweep-wide n_restarts, held constant across every cell "
             "(default: the brief's baseline, 8)",
    )
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    base = {**BASE, "n_restarts": args.n_restarts}
    evals_per_cell = base["pop_size"] * base["n_gen"] * base["n_restarts"]
    all_cells = list(cells())

    if args.pilot:
        baseline_cell = next(c for c in all_cells if c[1].endswith("*"))
        other_cell = next(c for c in all_cells if not c[1].endswith("*"))
        pilot_seconds = []
        for knob, label, over in (baseline_cell, other_cell):
            t0 = time.perf_counter()
            front = fit_pure_pareto(**{**base, **over}, seed=SEEDS[0])
            dt = time.perf_counter() - t0
            pilot_seconds.append(dt)
            print(
                f"[pilot] {knob}={label}: {len(front.F)} pts, {dt:.1f}s, "
                f"{dt / evals_per_cell * 1000:.4f} ms/eval "
                f"({evals_per_cell} evals/cell)"
            )
        mean_dt = sum(pilot_seconds) / len(pilot_seconds)
        n_runs = len(all_cells) * len(SEEDS)
        print(
            f"[pilot] mean {mean_dt:.1f}s/cell over {len(pilot_seconds)} timed cells "
            f"-> projected {n_runs} runs ({len(all_cells)} cells x {len(SEEDS)} seeds) "
            f"= {mean_dt * n_runs:.0f}s ({mean_dt * n_runs / 60:.1f} min)"
        )
        return

    # Step 3: objective scale. Reachable only through fit_pure_pareto's own
    # verbose banner (printed per restart) -- not parsed from stdout here, per
    # the brief's instruction not to build stdout parsing into the sweep. One
    # baseline fit, verbose, its printed numbers go in the report by hand.
    print("=== baseline objective-scale run (verbose=True) ===")
    fit_pure_pareto(**{**base, "verbose": True}, seed=SEEDS[0])
    print("=== end baseline objective-scale run ===")

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
                print(f"{knob}={label} seed={seed}: {len(F)} pts, {cached['seconds']:.1f}s [checkpoint]")
                continue
            t0 = time.perf_counter()
            front = fit_pure_pareto(**{**base, **over}, seed=seed)
            dt = time.perf_counter() - t0
            run = dict(knob=knob, level=label, seed=seed, F=front.F, seconds=dt)
            runs.append(run)
            _append_checkpoint(CHECKPOINT, {**run, "F": front.F.tolist()})
            print(f"{knob}={label} seed={seed}: {len(front.F)} pts, {dt:.1f}s")

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

    # ONE shared reference front and ONE shared ref_point for the whole sweep --
    # per-cell values would make the columns incomparable.
    ref = reference_front(*[r["F"] for r in runs], region=REGION)
    ref_point = np.max(np.vstack([r["F"] for r in runs]), axis=0) * 1.1

    rows = [
        {k: r[k] for k in ("knob", "level", "seed")}
        | {"seconds": r["seconds"]}
        | front_metrics(r["F"], reference=ref, ref_point=ref_point, region=REGION)
        for r in runs
    ]
    df = pl.DataFrame(rows)
    csv_path = OUT / "water_forte_knobs.csv"
    with open(csv_path, "w") as f:
        f.write(
            "# water Forte front (objectives=('psat','rho')) knob sweep, OFAT "
            "around pop_size=80 n_gen=80 refine=4 objectives=('psat','rho')\n"
        )
        f.write(
            f"# n_restarts={base['n_restarts']} held constant across every cell "
            f"(brief baseline is 8; see task-3-report.md for why this run used "
            f"{base['n_restarts']})\n"
        )
        f.write(
            "# every metric column is region-clipped to AARD_psat in [0, 3]% "
            "(AARD_rho unclipped) against one reference front and one ref_point "
            "shared across the whole sweep -- see task-3-report.md\n"
        )
        f.write(
            "# objective scale (from one verbose baseline run, not per-cell -- "
            "see task-3-report.md) is not a column here: it is constant across "
            "restarts of one run to within the reported precision, and per-cell "
            "values are not reachable without parsing stdout, which the brief "
            "asked not to build\n"
        )
        f.write("# read with pl.read_csv(path, comment_prefix='#')\n")
        df.write_csv(f)
    print(f"wrote {csv_path} ({len(df)} rows)")

    draw_figures(runs, df)
    print(f"wrote figures to {OUT}")

    CHECKPOINT.unlink(missing_ok=True)  # clean finish: no need to keep the raw checkpoint


if __name__ == "__main__":
    main()
