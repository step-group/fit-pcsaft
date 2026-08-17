"""Does the ``variant="de"`` win over the default ``variant="sbx"`` --
decisive on water at every budget (``de_confirm_check.py``) -- survive on an
*easier* problem?

The working explanation for water's result is that water has a complicated
Pareto set: strongly associating (na=1, nb=1), two disconnected parameter
basins, five fitted parameters -- exactly DE's target problem class. That
explanation predicts the advantage should shrink or vanish somewhere easier.
Camphor tests it, differing from water in every relevant way: non-associating
(na=0, nb=0, no association-scheme degeneracy), polar with mu=2.9 D *fixed*
rather than fitted (see ``examples/pure/06_camphor.py``), 3 fitted parameters
(m, sigma, epsilon_k) instead of 5, and 22 psat points but only 6 density
points. This can come out either way -- it is not built to confirm DE's win.

Camphor came out a near-null: hv bands overlap at every rung once clipped to
AARD <= 10%, DE lags badly at 2.7k, and by 28.8k the two reach the same
extremes with DE modestly ahead on pooled dominance. Two explanations fit that
equally well, and they have different consequences:

  (a) *non-associating, 3 parameters* -> the Pareto set is a single L-shaped
      elbow with no basin structure, so DE's exploration buys nothing; or
  (b) *only 6 density points* -> axis 1 barely constrains anything, so the
      "front" is close to a one-objective problem and any operator finds it.

**Carvone separates them.** It is the same shape as camphor on (a) -- also a
COMPONENTS terpene, also mu-fixed (2.8 D) and effectively non-associating, also
3 fitted parameters -- but not on (b): 26 psat points over 258-414 K and 20
density points over 278-368 K, against camphor's 6. If carvone reproduces
camphor's null, (a) holds and the mechanism claim survives. If carvone instead
looks like water, the camphor result was an artefact of a starved rho axis and
says nothing about association at all.

Carvone's data comes from TESIS ``data/processed/{psat,density}/carvone_fit.csv``,
**literature rows only**: the 5 "This work (2026)" density points are dropped
before the copy into ``examples/data/``, because this repo is still the public
``step-group/fit-pcsaft`` and no unpublished thesis measurement has ever been
committed to it (eugenol and camphor there are pure literature). Dropping them
cannot affect what is measured here -- the question is about the optimizer, not
about carvone's parameters.

Budget ladder x 2 operators x 3 seeds, ``objectives=("psat", "rho")`` (Forte
et al. 2018) -- no surface-tension data exists for camphor, so the default
``("vle", "sft")`` pair isn't available. ~3.7 ms/eval in this mode
(``fit_pure_pareto``'s own docstring, single-core) makes the whole ladder
cheap; see ``_print_cost_estimate``.

**bounds=None.** Camphor fits only [m, sigma, epsilon_k] (mu fixed, non-
associating), so the bounds list must have exactly 3 entries, not water's 5.
``fit_pure_pareto`` derives that automatically: ``fit_mu = mu is None`` is
False here (mu=2.9), ``is_associative = na > 0 and nb > 0`` is False (na=nb=0),
so ``_default_de_bounds(False, False)`` returns exactly ``_DE_BOUNDS_BASE``,
``[(1.0, 20.0), (2.0, 6.0), (50.0, 700.0)]`` for m/sigma/epsilon_k -- the
generic box, not narrowed. Narrowing it the way the water scripts narrow
theirs would be circular here: unlike water, there is no known camphor front
to draw a tighter box from, and ``fit_pure_pareto``'s own docstring warns
against exactly that for a new compound.

**extrapolate_psat is dropped from BASE, not passed.** ``fit_pure_pareto``
has no such parameter -- unlike ``fit_pure``/``fit_pure_de``, it hardcodes
``extrapolate_psat=False`` in its internal ``_setup_pure_fit`` call
(``pareto.py``, commented "Inert here"), because the search's own cost path
(``_evaluate_point``) never calls the analytical-Jacobian cost function that
flag controls (``_fit_utils.py``'s August-equation branch); it grades
near-critical psat failures through its own violation fraction
(``_MIN_VALID_FRACTION=0.5``) instead, which applies regardless of this flag.
Passing it raises ``TypeError: unexpected keyword argument`` -- caught by the
smoke fit below, not guessed at.

**REGION=None (no clip).** The mandated smoke fit (pop_size=8, n_gen=4,
n_restarts=1, refine=0, n_jobs=1 -- 32 evaluations) returned a 4-point front
spanning AARD_psat 4.4-100% and AARD_rho 5.8-161%: exactly the "cluster, not
a curve" regime ``fit_pure_pareto``'s own docstring describes below a few
thousand evaluations. Unlike water's benchmarks, which draw ``REGION`` from
an existing, properly-budgeted front, there is no prior camphor front to say
where this ladder's real rungs (2 700-28 800 evals) will land, and guessing a
tight clip from an unconverged 32-eval front risks exactly what the brief
warns against: a region that silently empties most of the real cells. Left
unclipped, every metric is computed over the whole front instead.

    uv run python benchmarks/transfer_check.py            # camphor (the default)
    uv run python benchmarks/transfer_check.py carvone

Resumable like the sibling scripts: every finished (op, rung, seed) triple is
appended to camphor-transfer/_checkpoint.csv, and a re-run skips whatever is
already there. Delete that file to start over.

n_jobs=-1 (fit_pure_pareto's default) spawns a worker pool; a worker
re-imports this module, so every fitting call must stay inside main() --
module-level fitting code makes every worker start its own pool.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fit_pcsaft import fit_pure_pareto
from fit_pcsaft._pure.front_quality import front_metrics, reference_front

REPO = Path(__file__).parent.parent
DATA = REPO / "examples" / "data"

# Per-compound overrides. Everything else (ladder, operators, seeds, metrics,
# figures) is shared, which is the point: the two runs differ only in the
# problem, so a difference in the result is a difference in the problem.
#
# `mu` is fixed rather than fitted in both, and both are non-associating, so
# both fit exactly 3 parameters [m, sigma, epsilon_k] and both take
# bounds=None -> _default_de_bounds(False, False); see the module docstring.
COMPOUNDS = {
    "camphor": dict(
        mu=2.9,        # fixed, as in examples/pure/06_camphor.py
        na=0, nb=0,    # non-associating
    ),
    "carvone": dict(
        mu=2.8,        # McLellan, benzene soln. -- as in TESIS s04_fit_carvone.py
        # TESIS fits na=None, nb=1: zero donors, so no A-B pair can form and
        # _setup_pure_fit's `na > 0 and nb > 0` is False. The lone acceptor
        # exists for cross-association with water in the binary fit and is
        # inert for the pure component, so this is a 3-parameter fit too.
        na=0, nb=1,
    ),
    # The control for camphor's null result. Carvone beat camphor on exactly one
    # axis -- 20 density points against 6 -- and won decisively where camphor
    # tied, so "camphor's rho axis was starved" is the obvious explanation and
    # also an uncontrolled one: the two compounds differ in T range and in which
    # literature sources they come from as well. This entry removes every
    # difference but the count, re-running *carvone itself* on 6 density points
    # evenly spaced across its own range (see `make_rho6` below). If DE's win
    # survives, point count is not what silenced camphor and the explanation has
    # to be found elsewhere; if it collapses, it is.
    "carvone-rho6": dict(
        id="carvone",  # PubChem lookup key -- must stay a real compound name
        mu=2.8, na=0, nb=1,
        psat_stem="carvone",
        density_stem="carvone_rho6",
    ),
}

# ponytail: argv, not an env var. multiprocessing spawn re-imports this module
# in each worker with a different sys.argv, so COMPOUND there falls back to the
# default -- harmless, because the workers evaluate fit_pcsaft's own picklable
# closure and never read these globals. main() is __main__-guarded, so it does
# not re-run either. Switch to an env var only if a worker ever needs the name.
COMPOUND = sys.argv[1] if len(sys.argv) > 1 else "camphor"
if COMPOUND not in COMPOUNDS:
    raise SystemExit(f"unknown compound {COMPOUND!r}; use one of {sorted(COMPOUNDS)}")

OUT = Path(__file__).parent / f"{COMPOUND}-transfer"
# Raw per-cell checkpoint -- crash recovery only, not a committed artefact
# (those are <compound>_transfer.csv, fronts.json and the PNGs). Same pattern as
# de_confirm_check.py / variant_sft_check.py, copied verbatim.
CHECKPOINT = OUT / "_checkpoint.csv"
# Every cell's raw front, kept for good -- so a later rescore against a
# different reference/region costs nothing.
FRONTS = OUT / "fronts.json"
CSV_PATH = OUT / f"{COMPOUND}_transfer.csv"

# `id` reaches PubChem through _fetch_compound, so it has to stay a real
# compound name even when the run label is not one (see "carvone-rho6").
_spec = dict(COMPOUNDS[COMPOUND])
_psat_stem = _spec.pop("psat_stem", COMPOUND)
_density_stem = _spec.pop("density_stem", COMPOUND)

BASE = dict(
    id=_spec.pop("id", COMPOUND),
    psat_path=DATA / "psat" / f"{_psat_stem}.csv",
    density_path=DATA / "density" / f"{_density_stem}.csv",
    objectives=("psat", "rho"),  # neither compound has surface-tension data
    bounds=None,                 # -> _default_de_bounds(False, False); see module docstring
    refine=4,
    verbose=False,
    # Deliberately NOT extrapolate_psat=True: fit_pure_pareto accepts no such
    # kwarg (it is hardcoded False and inert internally) -- see module docstring.
    **_spec,
)


def make_rho6() -> Path:
    """Carvone's density set thinned to 6 points, evenly spaced over its own range.

    Evenly spaced rather than clustered the way camphor's 6 happen to be: this
    is meant to isolate the *count*, so it gives the thinned set the best
    possible spread. If DE's advantage survives even the fairest 6-point set,
    the count is not the mechanism.
    """
    src = DATA / "density" / "carvone.csv"
    dst = DATA / "density" / "carvone_rho6.csv"
    rows = list(csv.DictReader(src.open()))
    key = next(k for k in rows[0] if "K" in k)
    rows.sort(key=lambda r: float(r[key]))
    idx = np.linspace(0, len(rows) - 1, 6).round().astype(int)
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([rows[i] for i in idx])
    return dst

LADDER = {"2.7k": (30, 30, 3), "10.8k": (40, 45, 6), "28.8k": (60, 80, 6)}
OPS = {"sbx": {}, "de-cr1.0": {"variant": "de", "de_cr": 1.0}}
SEEDS = (101, 202, 303)
REGION = None  # see module docstring

_SEED_MARKERS = {SEEDS[0]: "o", SEEDS[1]: "s", SEEDS[2]: "^"}

# Measured in fit_pure_pareto's own docstring: objectives=("psat", "rho")
# averaged 3.69 ms/eval at n_jobs=1. Actual wall time will be lower under the
# default n_jobs=-1 worker pool this script uses.
MS_PER_EVAL = 0.0037


def cells():
    for op, op_over in OPS.items():
        for rung, (pop, gen, restarts) in LADDER.items():
            for seed in SEEDS:
                label = f"{op}@{rung}"
                over = {**op_over, "pop_size": pop, "n_gen": gen, "n_restarts": restarts}
                extra = {"op": op, "rung": rung, "evals": pop * gen * restarts}
                yield label, seed, over, extra


# Smallest possible check that the design hasn't silently drifted -- cheap
# (no feos/pymoo touched), fine to run at import time in a spawned worker too.
assert len(list(cells())) == len(OPS) * len(LADDER) * len(SEEDS) == 18


def _append_checkpoint(path: Path, row: dict) -> None:
    """Append one completed cell's raw result. Writes the header on first write."""
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["op", "rung", "seed", "seconds", "F_json"])
        w.writerow([row["op"], row["rung"], row["seed"], row["seconds"], json.dumps(row["F"])])
        f.flush()


def _load_checkpoint(path: Path) -> dict[tuple[str, str, int], dict]:
    """(op, rung, seed) -> row, for whatever cells a prior run already finished."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {
            (row["op"], row["rung"], int(row["seed"])): {
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
        _append_checkpoint(path, dict(op="sbx", rung="2.7k", seed=101, seconds=1.5, F=[[1.0, 2.0]]))
        _append_checkpoint(path, dict(op="de-cr1.0", rung="10.8k", seed=202, seconds=2.5, F=[[3.0, 4.0]]))
        loaded = _load_checkpoint(path)
        assert len(loaded) == 2
        assert loaded[("sbx", "2.7k", 101)] == {"seconds": 1.5, "F": [[1.0, 2.0]]}
        assert loaded[("de-cr1.0", "10.8k", 202)] == {"seconds": 2.5, "F": [[3.0, 4.0]]}


_checkpoint_roundtrip_selfcheck()


def _print_cost_estimate() -> None:
    all_cells = list(cells())
    total_evals = sum(extra["evals"] for _, _, _, extra in all_cells)
    print(
        f"planned: {len(all_cells)} cells ({len(OPS)} ops x {len(LADDER)} rungs x "
        f"{len(SEEDS)} seeds), {total_evals} evaluations total"
    )
    print(
        f"cost estimate (~{MS_PER_EVAL * 1000:.2f} ms/eval single-core, "
        f"fit_pure_pareto's own docstring): ~{total_evals * MS_PER_EVAL / 60:.1f} min "
        "single-core equivalent; actual wall time lower under n_jobs=-1"
    )


def _clip(F: np.ndarray, region) -> np.ndarray:
    """Per-axis [(lo, hi) | None, ...] clip, same semantics as front_quality's.
    ``region=None`` (this script's REGION) leaves F unchanged."""
    F = np.asarray(F, dtype=float)
    if region is None:
        return F
    keep = np.ones(len(F), dtype=bool)
    for axis, bounds in enumerate(region):
        if bounds is None:
            continue
        lo, hi = bounds
        keep &= (F[:, axis] >= lo) & (F[:, axis] <= hi)
    return F[keep]


def _run_cells(done: dict) -> list[dict]:
    runs = []
    for label, seed, over, extra in cells():
        key = (extra["op"], extra["rung"], seed)
        if key in done:
            cached = done[key]
            F = np.asarray(cached["F"], dtype=float)
            runs.append({"label": label, "seed": seed, "F": F, "seconds": cached["seconds"], **extra})
            print(f"{label} seed={seed}: {len(F)} pts, {cached['seconds']:.1f}s [checkpoint]")
            continue
        t0 = time.perf_counter()
        front = fit_pure_pareto(**{**BASE, **over}, seed=seed)
        dt = time.perf_counter() - t0
        run = {"label": label, "seed": seed, "F": front.F, "seconds": dt, **extra}
        runs.append(run)
        _append_checkpoint(
            CHECKPOINT,
            {"op": extra["op"], "rung": extra["rung"], "seed": seed, "seconds": dt, "F": front.F.tolist()},
        )
        print(f"{label} seed={seed}: {len(front.F)} pts, {dt:.1f}s")
    return runs


def _ladder_figure(df: pl.DataFrame, path: Path) -> None:
    """hv vs. evals/cell, log-x, one line per operator, seed min-max as a band."""
    ops = list(OPS)
    colors = dict(zip(ops, plt.get_cmap("tab10").colors))
    rungs = sorted(df.select("rung", "evals").unique().rows(), key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(6, 5))
    for op in ops:
        xs, med, lo, hi = [], [], [], []
        for rung, evals in rungs:
            vals = df.filter((pl.col("op") == op) & (pl.col("rung") == rung))["hv"].drop_nulls().to_numpy()
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
        f"{COMPOUND} budget ladder: hv vs. evals/cell\n"
        "objectives=('psat','rho'); shared reference front/ref_point; band = seed min-max"
    )
    ax.legend(title="operator", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _overlay_figure(runs: list[dict], path: Path) -> None:
    """Every op@rung front on shared axes, region-clipped; a level's 3 seeds share one colour."""
    levels = [f"{op}@{rung}" for op in OPS for rung in LADDER]
    colors = dict(zip(levels, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in runs:
        F = _clip(r["F"], REGION)
        if len(F) == 0:
            continue
        ax.scatter(
            F[:, 0], F[:, 1], s=18, alpha=0.75,
            color=colors[r["label"]], marker=_SEED_MARKERS[r["seed"]],
            label=r["label"],
        )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[level], label=level)
        for level in levels
    ]
    ax.legend(handles=handles, title="op@rung", fontsize=7, ncol=2)
    ax.set_xlabel("AARD_psat [%]")
    ax.set_ylabel("AARD_rho [%]")
    ax.set_title(
        f"{COMPOUND}: variant=de vs sbx, objectives=('psat','rho') "
        "(no region clip; marker = seed)"
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

    runs = _run_cells(done)

    with open(FRONTS, "w") as f:
        json.dump(
            [
                {**{k: v for k, v in r.items() if k != "F"}, "F": np.asarray(r["F"]).tolist()}
                for r in runs
            ],
            f,
        )
    print(f"wrote {FRONTS} ({len(runs)} cells)")

    # ONE shared reference front and ONE shared ref_point across all 18
    # cells -- per-cell values would make the columns incomparable.
    ref = reference_front(*[r["F"] for r in runs], region=REGION)
    ref_point = np.max(np.vstack([r["F"] for r in runs]), axis=0) * 1.1

    rows = [
        {"op": r["op"], "rung": r["rung"], "seed": r["seed"], "evals": r["evals"], "seconds": r["seconds"]}
        | front_metrics(r["F"], reference=ref, ref_point=ref_point, region=REGION)
        for r in runs
    ]
    df = pl.DataFrame(rows)
    with open(CSV_PATH, "w") as f:
        f.write(
            f"# {COMPOUND} budget ladder -- does variant='de''s win over 'sbx' on water "
            "(de_confirm_check.py) transfer to an easier, non-associating, "
            "3-parameter problem?\n"
        )
        f.write(
            f"# objectives=('psat','rho') (Forte et al. 2018) -- {COMPOUND} has no "
            "surface-tension data, so ('vle','sft') is not available\n"
        )
        f.write(
            f"# data: {len(open(BASE['psat_path']).readlines()) - 1} psat points, "
            f"{len(open(BASE['density_path']).readlines()) - 1} density points; "
            f"mu={BASE['mu']} fixed, na={BASE['na']} nb={BASE['nb']}\n"
        )
        f.write(
            "# rungs: "
            + ", ".join(f"{label} ({p}x{g}x{r}={p * g * r})" for label, (p, g, r) in LADDER.items())
            + "; 3 seeds each, 2 operators (sbx, de-cr1.0); refine=4\n"
        )
        f.write(
            "# bounds=None -> fit_pure_pareto's own default box for a non-associating, "
            "mu-fixed compound: [(1,20),(2,6),(50,700)] for m/sigma/epsilon_k -- see "
            "module docstring\n"
        )
        f.write(
            "# axis 0 is AARD_psat [%], axis 1 is AARD_rho [%] -- NOT eq 30's pooled "
            "AAD_vle\n"
        )
        f.write("# REGION=None: no clip applied anywhere in this file -- see module docstring for why\n")
        f.write(
            "# refs was left at fit_pure_pareto's default for this mode, (2.0, 2.0), "
            "and is not a column here\n"
        )
        f.write(
            "# every metric column is against one reference front and one ref_point "
            "shared across all 18 cells\n"
        )
        f.write("# read with pl.read_csv(path, comment_prefix='#')\n")
        df.write_csv(f)
    print(f"wrote {CSV_PATH} ({len(df)} rows)")

    _ladder_figure(df, OUT / "ladder_hv.png")
    _overlay_figure(runs, OUT / f"overlay_{COMPOUND}.png")
    print(f"wrote figures to {OUT}")

    CHECKPOINT.unlink(missing_ok=True)  # clean finish: no need to keep the raw checkpoint


if __name__ == "__main__":
    main()
