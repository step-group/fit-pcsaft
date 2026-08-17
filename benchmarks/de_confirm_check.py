"""Two follow-ups to ``variant_sft_check.py``'s result (``variant="de",
de_cr=1.0`` beat ``variant="sbx"`` in both objective modes, disjoint seeds).

**Part A -- the budget ladder.** Every comparison so far ran at one reduced
budget (2700 evals/cell), which cannot tell "SBX is worse" from "SBX
converges slower". This runs ``sbx`` and ``de-cr1.0`` at two more, larger
budgets (10 800 and 28 800 evals/cell), 2 seeds each, and reuses the existing
2700-eval ``sbx*``/``de-cr1.0`` rows from
``benchmarks/variant-sft/variant_sft.csv`` / ``fronts.json`` as the ladder's
first, cheapest rung -- **not re-run**. Converging lines in ``ladder_hv.png``
(hv vs. evals/cell, log-x, one line per operator, seed min-max as a band)
mean SBX is merely slower; persistently separated lines mean it is worse.

**Part B -- the DE parameter grid.** ``de_f`` was never varied (default 0.5
throughout) and ``de_cr=0.5`` was clearly worse than ``1.0``, so DE's
parameters demonstrably matter here -- 1.0 may not be optimal either. Grids
``de_f in (0.3, 0.5, 0.8)`` x ``de_cr in (0.5, 0.8, 1.0)`` at the *cheap*
budget only (same 2700 evals/cell as the ladder's reused rung), 2 seeds, 18
cells -- ``grid_heatmap.png`` is median hv over the F x CR grid.

Both parts share one reference front and one ref_point, built from Part A's
runs only (the ladder spans the full budget range, including a rung at the
same budget as the grid) so every hv number in this script -- ladder and
grid alike -- is on the same footing, per ``front_metrics``' own rule that
hv/igd_plus are only comparable across a shared reference/ref_point.

    uv run python benchmarks/de_confirm_check.py

Resumable exactly like ``variant_sft_check.py``: every finished
(part, label, seed) cell is appended to ``de-confirm/_checkpoint.csv``, and a
re-run skips whatever is already there. Delete that file to start over. Part
B (cheap, ~14 min) runs before Part A (~46 min) so an interruption still
banks the inexpensive half.

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
OUT = Path(__file__).parent / "de-confirm"
# Raw per-cell checkpoint -- crash recovery only, not a committed artefact
# (those are de_confirm.csv, fronts.json and the PNGs). See
# variant_sft_check.py, which this pattern is copied from verbatim.
CHECKPOINT = OUT / "_checkpoint.csv"
# Every cell's raw front, kept for good -- so a later rescore against a
# different reference costs nothing. Same row shape idea as the sibling
# scripts; "part" distinguishes ladder cells from grid cells.
FRONTS = OUT / "fronts.json"
CSV_PATH = OUT / "de_confirm.csv"

# The 2700-eval rung already exists -- read it, never re-run it.
EXISTING_RUN = REPO / "benchmarks" / "variant-sft" / "fronts.json"
EXISTING_RUNG_LABEL = "2.7k"
EXISTING_RUNG_EVALS = 30 * 30 * 3  # pop_size x n_gen x n_restarts of that earlier run
# variant_sft_check.py's level names don't match this script's operator
# names (baseline is "sbx*" there, "sbx" here) -- map explicitly so a rename
# on either side fails loudly (see the assert in _load_existing_rung)
# instead of silently dropping rows. "de-cr0.5" isn't part of this
# comparison and is left out on purpose.
EXISTING_LEVEL_MAP = {"sbx*": "sbx", "de-cr1.0": "de-cr1.0"}

BASE = dict(
    id="water",
    psat_path=DATA / "psat" / "water.csv",
    density_path=DATA / "density" / "water.csv",
    sft_path=DATA / "surface_tension" / "water.csv",  # required: "sft" is an objective
    na=1, nb=1,
    objectives=("vle", "sft"),
    bounds=[(0.8, 3.0), (2.0, 3.5), (150.0, 400.0), (1e-3, 0.35), (1500.0, 4000.0)],
    refine=4,
    verbose=False,
)
REGION = [(0.0, 5.0), None]  # AAD_vle [%] axis, as in variant_sft_check.py

# --- Part A: budget ladder --------------------------------------------------
LADDER = {                      # label -> (pop_size, n_gen, n_restarts)
    "10.8k": (40, 45, 6),       #  10 800 evals/cell
    "28.8k": (60, 80, 6),       #  28 800 evals/cell
}
# Both arms state their operator -- `sbx: {}` relied on the library default,
# which is now "de". See transfer_check.py's OPS.
LADDER_OPS = {"sbx": {"variant": "sbx"}, "de-cr1.0": {"variant": "de", "de_cr": 1.0}}
LADDER_SEEDS = (101, 202)

# --- Part B: DE parameter grid ----------------------------------------------
GRID_F = (0.3, 0.5, 0.8)
GRID_CR = (0.5, 0.8, 1.0)
GRID_SEEDS = (101, 202)
GRID_POP, GRID_GEN, GRID_RESTARTS = 30, 30, 3  # the cheap budget: 2700 evals/cell
GRID_EVALS = GRID_POP * GRID_GEN * GRID_RESTARTS

# Measured on this machine in this mode: 2700 evals took ~46s with the
# worker pool (see variant_sft_check.py's cells) -> ~17 ms/eval wall.
MS_PER_EVAL = 0.017


def ladder_cells():
    for op, op_over in LADDER_OPS.items():
        for rung, (pop, gen, restarts) in LADDER.items():
            for seed in LADDER_SEEDS:
                label = f"{op}@{rung}"
                over = {**op_over, "pop_size": pop, "n_gen": gen, "n_restarts": restarts}
                extra = {"op": op, "rung": rung, "evals": pop * gen * restarts}
                yield "ladder", label, seed, over, extra


def grid_cells():
    for f in GRID_F:
        for cr in GRID_CR:
            for seed in GRID_SEEDS:
                label = f"F{f}_CR{cr}"
                over = {
                    "variant": "de", "de_f": f, "de_cr": cr,
                    "pop_size": GRID_POP, "n_gen": GRID_GEN, "n_restarts": GRID_RESTARTS,
                }
                extra = {"de_f": f, "de_cr": cr, "evals": GRID_EVALS}
                yield "grid", label, seed, over, extra


# Smallest possible check that the design hasn't silently drifted -- cheap
# (no feos/pymoo touched), fine to run at import time in a spawned worker too.
assert len(list(ladder_cells())) == len(LADDER_OPS) * len(LADDER) * len(LADDER_SEEDS) == 8
assert len(list(grid_cells())) == len(GRID_F) * len(GRID_CR) * len(GRID_SEEDS) == 18


def _append_checkpoint(path: Path, row: dict) -> None:
    """Append one completed cell's raw result. Writes the header on first write."""
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["part", "label", "seed", "seconds", "F_json"])
        w.writerow([row["part"], row["label"], row["seed"], row["seconds"], json.dumps(row["F"])])
        f.flush()


def _load_checkpoint(path: Path) -> dict[tuple[str, str, int], dict]:
    """(part, label, seed) -> row, for whatever cells a prior run already finished."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {
            (row["part"], row["label"], int(row["seed"])): {
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
        _append_checkpoint(path, dict(part="ladder", label="l", seed=1, seconds=1.5, F=[[1.0, 2.0]]))
        _append_checkpoint(path, dict(part="grid", label="l2", seed=2, seconds=2.5, F=[[3.0, 4.0]]))
        loaded = _load_checkpoint(path)
        assert len(loaded) == 2
        assert loaded[("ladder", "l", 1)] == {"seconds": 1.5, "F": [[1.0, 2.0]]}
        assert loaded[("grid", "l2", 2)] == {"seconds": 2.5, "F": [[3.0, 4.0]]}


_checkpoint_roundtrip_selfcheck()


def _print_cost_estimate() -> None:
    ladder_evals = sum(extra["evals"] for _, _, _, _, extra in ladder_cells())
    grid_evals = sum(extra["evals"] for _, _, _, _, extra in grid_cells())
    total = ladder_evals + grid_evals
    print("cost estimate (measured ~17 ms/eval wall on this machine in this mode):")
    print(
        f"  Part B grid (runs first): {sum(1 for _ in grid_cells())} cells, {grid_evals} evals, "
        f"~{grid_evals * MS_PER_EVAL / 60:.1f} min"
    )
    print(
        f"  Part A ladder (new cells only -- {EXISTING_RUNG_LABEL} rung reused from "
        f"{EXISTING_RUN}, not run): {sum(1 for _ in ladder_cells())} cells, {ladder_evals} evals, "
        f"~{ladder_evals * MS_PER_EVAL / 60:.1f} min"
    )
    print(f"  total planned: {total} evals, ~{total * MS_PER_EVAL / 60:.1f} min")


def _load_existing_rung() -> list[dict]:
    """The 2700-eval sbx*/de-cr1.0 fronts already computed in
    variant-sft/fronts.json -- Part A's cheapest rung, never re-run."""
    with open(EXISTING_RUN) as f:
        data = json.load(f)
    runs = []
    for row in data:
        if row["level"] not in EXISTING_LEVEL_MAP:
            continue
        op = EXISTING_LEVEL_MAP[row["level"]]
        runs.append({
            "part": "ladder", "label": f"{op}@{EXISTING_RUNG_LABEL}", "seed": row["seed"],
            "F": np.asarray(row["F"], dtype=float), "seconds": row["seconds"],
            "op": op, "rung": EXISTING_RUNG_LABEL, "evals": EXISTING_RUNG_EVALS,
        })
    expected = len(EXISTING_LEVEL_MAP) * 3  # 2 levels x 3 seeds in the reused file
    assert len(runs) == expected, (
        f"expected {expected} rows reused from {EXISTING_RUN} (2 levels x 3 seeds), "
        f"got {len(runs)} -- EXISTING_LEVEL_MAP or the source file changed"
    )
    return runs


def _run_cells(cell_list: list[tuple], done: dict) -> list[dict]:
    runs = []
    for part, label, seed, over, extra in cell_list:
        key = (part, label, seed)
        if key in done:
            cached = done[key]
            F = np.asarray(cached["F"], dtype=float)
            runs.append({"part": part, "label": label, "seed": seed, "F": F, "seconds": cached["seconds"], **extra})
            print(f"[{part}] {label} seed={seed}: {len(F)} pts, {cached['seconds']:.1f}s [checkpoint]")
            continue
        t0 = time.perf_counter()
        front = fit_pure_pareto(**{**BASE, **over}, seed=seed)
        dt = time.perf_counter() - t0
        run = {"part": part, "label": label, "seed": seed, "F": front.F, "seconds": dt, **extra}
        runs.append(run)
        _append_checkpoint(CHECKPOINT, {"part": part, "label": label, "seed": seed, "seconds": dt, "F": front.F.tolist()})
        print(f"[{part}] {label} seed={seed}: {len(front.F)} pts, {dt:.1f}s")
    return runs


def _metrics_row(r: dict, metrics: dict) -> dict:
    row = {
        "part": r["part"], "label": r["label"], "seed": r["seed"],
        "op": r.get("op"), "rung": r.get("rung"),
        "de_f": r.get("de_f"), "de_cr": r.get("de_cr"),
        "evals": r["evals"], "seconds": r["seconds"],
    }
    return row | metrics


def _ladder_figure(df: pl.DataFrame, path: Path) -> None:
    """hv vs. evals/cell, log-x, one line per operator, seed min-max as a band."""
    ops = list(LADDER_OPS)
    colors = dict(zip(ops, plt.get_cmap("tab10").colors))
    ladder_df = df.filter(pl.col("part") == "ladder")
    rungs = sorted(ladder_df.select("rung", "evals").unique().rows(), key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(6, 5))
    for op in ops:
        xs, med, lo, hi = [], [], [], []
        for rung, evals in rungs:
            vals = ladder_df.filter((pl.col("op") == op) & (pl.col("rung") == rung))["hv"].drop_nulls().to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            xs.append(evals)
            med.append(np.median(vals))
            lo.append(vals.min())
            hi.append(vals.max())
        if not xs:
            continue
        xs, med, lo, hi = (np.asarray(a) for a in (xs, med, lo, hi))
        ax.plot(xs, med, "o-", color=colors[op], label=op)
        ax.fill_between(xs, lo, hi, color=colors[op], alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("evaluations per cell")
    ax.set_ylabel("hv")
    ax.set_title(
        "budget ladder: hv vs. evals/cell\n"
        "(shared Part A reference front/ref_point; band = seed min-max)"
    )
    ax.legend(title="operator", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grid_heatmap(df: pl.DataFrame, path: Path) -> None:
    """Median hv (2 seeds) over the de_f x de_cr grid, scored vs. Part A's reference."""
    grid_df = df.filter(pl.col("part") == "grid")
    grid = np.full((len(GRID_CR), len(GRID_F)), np.nan)
    for i, cr in enumerate(GRID_CR):
        for j, f in enumerate(GRID_F):
            vals = grid_df.filter((pl.col("de_f") == f) & (pl.col("de_cr") == cr))["hv"].drop_nulls().to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals):
                grid[i, j] = np.median(vals)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(GRID_F)))
    ax.set_xticklabels(GRID_F)
    ax.set_yticks(range(len(GRID_CR)))
    ax.set_yticklabels(GRID_CR)
    ax.set_xlabel("de_f")
    ax.set_ylabel("de_cr")
    for i in range(len(GRID_CR)):
        for j in range(len(GRID_F)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="median hv (2 seeds)")
    ax.set_title(
        f"DE grid, pop={GRID_POP} n_gen={GRID_GEN} n_restarts={GRID_RESTARTS} "
        f"({GRID_EVALS} evals/cell)\nscored vs. Part A's reference front/ref_point"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    _print_cost_estimate()

    done = _load_checkpoint(CHECKPOINT)
    if done:
        print(f"resuming from {CHECKPOINT}: {len(done)} cell(s) already completed")

    # Part B first (cheap): if interrupted, this half is already banked.
    grid_runs = _run_cells(list(grid_cells()), done)
    # Part A: new ladder cells, plus the reused 2700-eval rung (never re-run).
    ladder_runs = _run_cells(list(ladder_cells()), done) + _load_existing_rung()

    with open(FRONTS, "w") as f:
        json.dump(
            [
                {**{k: v for k, v in r.items() if k != "F"}, "F": np.asarray(r["F"]).tolist()}
                for r in grid_runs + ladder_runs
            ],
            f,
        )
    print(f"wrote {FRONTS} ({len(grid_runs) + len(ladder_runs)} cells)")

    # ONE shared reference front and ONE shared ref_point, built from Part A
    # alone (it spans the full budget range, including a rung at the same
    # budget as the grid) -- Part B is scored against these too, per the
    # task's requirement that every number in this script be comparable.
    ref = reference_front(*[r["F"] for r in ladder_runs], region=REGION)
    ref_point = np.max(np.vstack([r["F"] for r in ladder_runs]), axis=0) * 1.1

    rows = [
        _metrics_row(r, front_metrics(r["F"], reference=ref, ref_point=ref_point, region=REGION))
        for r in ladder_runs + grid_runs
    ]
    df = pl.DataFrame(rows)
    with open(CSV_PATH, "w") as f:
        f.write(
            "# de_confirm: budget ladder (Part A, does DE's win over SBX persist at "
            "larger budgets?) and DE F/CR grid (Part B) for objectives=('vle','sft')\n"
        )
        f.write(
            f"# Part A rungs: {EXISTING_RUNG_LABEL} (reused from variant-sft/, not run), "
            + ", ".join(f"{label} ({p}x{g}x{r}={p * g * r})" for label, (p, g, r) in LADDER.items())
            + "\n"
        )
        f.write(
            f"# Part B grid: de_f in {GRID_F} x de_cr in {GRID_CR}, "
            f"pop_size={GRID_POP} n_gen={GRID_GEN} n_restarts={GRID_RESTARTS} "
            f"({GRID_EVALS} evals/cell), 2 seeds\n"
        )
        f.write(
            "# axis 0 is AAD_vle [%] (eq 30), axis 1 is AAD_sft [mN/m] (eq 31) -- see "
            "variant_sft_check.py\n"
        )
        f.write(
            "# every metric column is region-clipped to AAD_vle in [0, 5]% (AAD_sft "
            "unclipped) against one reference front and one ref_point shared across "
            "every row in this file, Part A and Part B alike, built from Part A's runs\n"
        )
        f.write(
            "# refs was left at fit_pure_pareto's default for this mode, (2.0, 0.7), "
            "and is not a column here\n"
        )
        f.write("# op/rung are set for Part A rows, de_f/de_cr for Part B rows; the other pair is null\n")
        f.write("# read with pl.read_csv(path, comment_prefix='#')\n")
        df.write_csv(f)
    print(f"wrote {CSV_PATH} ({len(df)} rows)")

    _ladder_figure(df, OUT / "ladder_hv.png")
    _grid_heatmap(df, OUT / "grid_heatmap.png")
    print(f"wrote figures to {OUT}")

    CHECKPOINT.unlink(missing_ok=True)  # clean finish: no need to keep the raw checkpoint


if __name__ == "__main__":
    main()
