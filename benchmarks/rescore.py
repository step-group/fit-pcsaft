"""Rescore knob_sweep.py's fronts.json against an alternative reference front.

No fitting, no network: reads the raw per-cell fronts knob_sweep.py already
saved and recomputes hv/igd_plus/spacing from them. Runs in seconds.

hv/igd_plus/spacing are not properties of a front by itself -- they measure
distance to (or coverage of) a *reference* front, and knob_sweep.py's own
water_forte_knobs.csv used one reference built from all 66 cells. That
choice can flatter a knob: the ``variant=de`` cells produce far denser
fronts than ``variant=sbx`` (~304 points vs ~65), so DE dominates the shared
reference and ends up partly scored against itself. ``--exclude-knob-level``
answers "how would this score against a reference that DE didn't get to
define" without a refit.

**The exclusion is asymmetric, on purpose.** ``--exclude-knob-level
variant=de-cr1.0`` removes that cell from the pool used to *build* the
reference front, but the cell itself is still scored against the resulting
reference, same as every other cell -- it just doesn't get a vote on the
yardstick. If that reads like a bug (an excluded cell missing from the
output), it isn't: an excluded knob/level with no rows in the CSV would make
this a filter, not a rescore.

    uv run python benchmarks/rescore.py
    uv run python benchmarks/rescore.py \\
        --exclude-knob-level variant=de-cr1.0 --exclude-knob-level variant=de-cr0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from fit_pcsaft._pure.front_quality import front_metrics, reference_front

OUT = Path(__file__).parent / "knob-sweep"
REGION = [(0.0, 3.0), None]  # AARD_psat <= 3%, the region of interest -- matches knob_sweep.py


def _parse_exclusion(spec: str) -> tuple[str, str]:
    knob, sep, level = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected KNOB=LEVEL, got {spec!r}")
    return knob, level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fronts", type=Path, default=OUT / "fronts.json", help="fronts.json to read")
    parser.add_argument(
        "--exclude-knob-level", type=_parse_exclusion, action="append", default=[],
        metavar="KNOB=LEVEL", dest="excluded",
        help="exclude this (knob, level) from the reference front only -- its cells are "
             "still scored against the resulting reference. Repeatable.",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT / "water_forte_knobs_rescored.csv", help="output CSV path",
    )
    args = parser.parse_args()

    records = json.loads(args.fronts.read_text())
    excluded = set(args.excluded)

    ref = reference_front(
        *[r["F"] for r in records if (r["knob"], r["level"]) not in excluded],
        region=REGION,
    )
    # One shared ref_point across every cell, including excluded ones -- matches
    # knob_sweep.py exactly. Unlike the reference front, ref_point only scales
    # hypervolume; it isn't the "yardstick" the exclusion is about.
    ref_point = np.max(np.vstack([r["F"] for r in records]), axis=0) * 1.1

    rows = [
        {k: r[k] for k in ("knob", "level", "seed")}
        | {"seconds": r["seconds"]}
        | front_metrics(r["F"], reference=ref, ref_point=ref_point, region=REGION)
        for r in records
    ]
    df = pl.DataFrame(rows)
    with open(args.out, "w") as f:
        f.write(
            "# water Forte front knob sweep, rescored from fronts.json with no refit "
            f"(source: {args.fronts})\n"
        )
        if excluded:
            excl_str = ", ".join(f"{k}={lvl}" for k, lvl in sorted(excluded))
            f.write(
                f"# reference front excludes: {excl_str} -- those cells are still scored "
                "below, they just did not help build the reference\n"
            )
        else:
            f.write("# reference front excludes: (none) -- built from every cell, same as water_forte_knobs.csv\n")
        f.write("# read with pl.read_csv(path, comment_prefix='#')\n")
        df.write_csv(f)
    print(f"wrote {args.out} ({len(df)} rows, reference excludes {sorted(excluded) or 'nothing'})")


if __name__ == "__main__":
    main()
