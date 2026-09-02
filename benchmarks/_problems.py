"""One source of truth for what each transfer-test problem *is*.

`transfer_check.py` runs a ladder and `rescore_transfer.py` reads it back. Both
need the same fit configuration, the same axis labels and the same clip, and they
drifted the moment there was a second compound -- a carvone run shipped its overlay
as `overlay_camphor.png`, because the filename lived in the script while the
compound lived in a registry. Everything per-problem lives here; neither script
defines any of it locally.

This module imports from `fit_pcsaft` but from neither script, so there is no cycle.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from fit_pcsaft import SurfaceTensionOptions

DATA = Path(__file__).parent.parent / "examples" / "data"


def requested(script: str, default: str = "camphor") -> str:
    """The problem named on the command line, or ``default``.

    Both benchmark scripts take the problem as ``argv[1]`` and resolve it at
    import time, and both are also *imported* rather than run -- by the tests,
    and by multiprocessing spawn re-importing ``__main__`` in every worker. In
    those cases ``argv[1]`` belongs to pytest or to the worker, not to us:
    reading it raised ``SystemExit: unknown problem
    'tests/test_benchmark_problems.py'`` and took the whole test session with it.

    So an unrecognised name is an error only when this file really is the
    program being run -- compared against ``argv[0]`` by resolved path, not by
    suffix, because ``python -m pytest``'s own ``argv[0]`` ends in ``.py`` too.
    """
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name in PROBLEMS:
            return name
        if Path(sys.argv[0]).resolve() == Path(script).resolve():
            raise SystemExit(
                f"unknown problem {name!r}; use one of {sorted(PROBLEMS)}"
            )
    return default


@dataclass(frozen=True)
class Problem:
    """Everything that differs between one transfer test and the next.

    `fit` is merged into `fit_pure_pareto`'s kwargs after the shared defaults and
    the paths, so it may override any of them. `rescore_cap` is a *read-time* clip
    applied by `rescore_transfer.py`; the ladder itself always runs unclipped and
    stores raw fronts, because on camphor a clip guessed before the front was known
    would have hidden most of it. `None` on an axis means that axis is not clipped.
    """

    fit: dict
    psat_stem: str
    density_stem: str
    axis_labels: tuple[str, str]
    ms_per_eval: float
    sft_stem: str | None = None
    rescore_cap: tuple[float | None, float | None] = (10.0, 10.0)
    select_refs: tuple[float, float] | None = None
    baseline_eq32: float | None = None

    def paths(self) -> dict:
        p = {
            "psat_path": DATA / "psat" / f"{self.psat_stem}.csv",
            "density_path": DATA / "density" / f"{self.density_stem}.csv",
        }
        if self.sft_stem is not None:
            p["sft_path"] = DATA / "surface_tension" / f"{self.sft_stem}.csv"
        return p


_FORTE_AXES = ("AARD_psat [%]", "AARD_rho [%]")

# Measured in fit_pure_pareto's own docstring: ("psat","rho") averages 0.108 ms/eval
# over a 600-evaluation search, batched in-process on feos's rayon threads (2026-09-01;
# it was 3.69 ms/eval single-core under the old per-point loop and pool).
_MS_FORTE = 0.00011
# Measured on this machine, wall, under the pool: de-confirm's ladder ran 2700 evals
# in ~48 s, 10800 in ~190 s and 28800 in ~460 s -> 16-18 ms/eval.
_MS_VLE_SFT = 0.017

# Thymol's production configuration, copied from TESIS
# analysis/scripts/pure/s04_fit_thymol.py. Do not widen this box to "be safe": that
# file's docstring records what the wide box did -- a 121-point front whose eq-32
# tangent was its own lowest-AAD_vle endpoint, because _objective_scale froze a 64x
# divisor from generation zero and squeezed the interesting region into a sliver of
# the normalised axis. This box is about 1/17 the volume and is drawn to make
# generation zero dense where good fits live.
_THYMOL_BOUNDS = [
    (2.0, 4.5),        # m
    (3.9, 4.9),        # sigma / A
    (260.0, 400.0),    # epsilon_k / K
    (1.0e-4, 0.4),     # kappa_ab -- the UPPER bound is load-bearing on thymol
    (2000.0, 3600.0),  # epsilon_k_ab / K
]

PROBLEMS: dict[str, Problem] = {
    "camphor": Problem(
        fit={"id": "camphor", "mu": 2.9, "na": 0, "nb": 0,
             "objectives": ("psat", "rho"), "bounds": None},
        psat_stem="camphor", density_stem="camphor",
        axis_labels=_FORTE_AXES, ms_per_eval=_MS_FORTE,
    ),
    "carvone": Problem(
        fit={"id": "carvone", "mu": 2.8, "na": 0, "nb": 1,
             "objectives": ("psat", "rho"), "bounds": None},
        psat_stem="carvone", density_stem="carvone",
        axis_labels=_FORTE_AXES, ms_per_eval=_MS_FORTE,
    ),
    # Control for camphor's null: carvone itself on 6 density points, evenly spaced
    # over its own range, so nothing but the point count changes.
    "carvone-rho6": Problem(
        fit={"id": "carvone", "mu": 2.8, "na": 0, "nb": 1,
             "objectives": ("psat", "rho"), "bounds": None},
        psat_stem="carvone", density_stem="carvone_rho6",
        axis_labels=_FORTE_AXES, ms_per_eval=_MS_FORTE,
    ),
    # The only problem here that TESIS actually runs through fit_pure_pareto.
    "thymol": Problem(
        fit={
            "id": "thymol",
            "na": 1, "nb": 1,   # 2B: one donor/acceptor pair on the phenolic -OH
            "objectives": ("vle", "sft"),
            "bounds": _THYMOL_BOUNDS,
            # REF_VLE/REF_SFT. Mostly inert for the search -- _objective_scale takes
            # its divisors from generation zero's feasible span and falls back to
            # refs only when fewer than _MIN_SCALE_SAMPLE=4 points are feasible --
            # but it is exactly what select() minimises, and matching production
            # costs nothing.
            "refs": (2.0, 3.0),
            # Seeds only the tanh fallback initializer for the density profile;
            # measured inert to 12 decimals across 22 K in s04_fit_thymol.py. Passed
            # because the default 500.0 K is a generic placeholder far below any
            # plausible value for a C10H14O phenol.
            "sft_options": SurfaceTensionOptions(critical_temperature_k=698.3),
        },
        psat_stem="thymol", density_stem="thymol", sft_stem="thymol",
        axis_labels=("AAD_vle [%]", "AAD_sft [mN/m]"),
        ms_per_eval=_MS_VLE_SFT,
        # The front spans AAD_vle ~0.8-8.9% (s04_fit_thymol.py's docstring), so 5% is
        # informed rather than guessed -- and it is the clip the water ("vle","sft")
        # benchmarks already use. AAD_sft is in mN/m and is not clipped.
        rescore_cap=(5.0, None),
        select_refs=(2.0, 3.0),
        # The committed thymol fit: AAD_vle 0.797099881129, AAD_sft 0.487513064616
        # -> 0.797/2.0 + 0.488/3.0. Produced by SBX at 80 x 120 x 10 = 96 000 evals.
        # (For scale: the plain fit_pure set it replaced scores 1.669.)
        baseline_eq32=0.561,
    ),
}
