"""Absolute front-quality indicators, delegated to pymoo.

``coverage`` (pareto.py) is pairwise: every point of B is checked against
*some* point of A, over the whole front. On a real water front here, one tail
point at (7.73, 0.524) weakly dominated 217 of 522 points and inverted the
verdict -- a single outlier decided the whole comparison. Hypervolume, IGD+
and spacing (this module) don't share that specific failure mode, but the
actual fix is ``region``: a per-axis ``[(lo, hi) | None, ...]`` clip applied
before anything else is computed, so a comparison can be scoped to the part
of the front that matters (e.g. AARD_psat <= 2%) instead of being decided by
an extreme point nobody would ever select. See
``test_the_region_of_interest_can_flip_the_verdict``.

``non_dominated`` and ``coverage`` are reused from ``pareto.py``, not
reimplemented. This module imports from ``pareto.py``, never the reverse.
``pareto.py``'s re-export of ``front_metrics``/``reference_front``/
``compare_fronts`` is a plain top-level ``from .front_quality import ...``,
placed after ``non_dominated``/``coverage`` are defined (this module needs
them at its own top level). That ordering is the only thing that matters --
not import order at the call site: ``fit_pcsaft/__init__.py`` unconditionally
imports ``pareto.py`` before anything else can reach ``front_quality.py``, so
this file's top-level code never runs before ``pareto.py`` has executed down
to the re-export line, regardless of which module a caller names first.
Verified directly: importing ``fit_pcsaft._pure.front_quality`` first, in a
fresh interpreter, still works, because that import drags in
``fit_pcsaft/__init__.py`` -> ``pareto.py`` first either way.

HV note: ``ref_point`` is passed with ``norm_ref_point=False``, but that flag
is inert on pymoo 0.6.2 as installed here -- confirmed by reading
``pymoo/indicators/hv.py`` and ``pymoo/util/normalization.py``:
``norm_ref_point`` only has an effect when ``Hypervolume`` also receives
``zero_to_one=True``, which nothing in this codebase passes, so
``self.normalization`` is always ``NoNormalization()`` and ``norm_ref_point``
changes nothing (confirmed empirically too: identical HV for the same front
under both settings). ``False`` is set anyway, purely as a defensive default:
it costs nothing today and means a future pymoo release that wires the flag
up for real can't silently start renormalizing ``ref_point`` per front, which
would break the thing that actually matters here -- one shared ``ref_point``
producing comparable hypervolumes across the ~45 fronts Task 3 sweeps. What
``test_hv_is_comparable_across_fronts_with_a_shared_ref_point`` pins is that
comparability itself (same ``ref_point``, strictly dominating front scores
strictly higher), not the ``norm_ref_point`` flag -- on this pymoo build the
flag has nothing to pin.

A hypervolume is only meaningful when compared against another HV computed
with the *same* ``ref_point`` -- that is the one easy way to get a
meaningless comparison, and it is why ``front_metrics``' default
(``F.max(axis=0) * 1.1``, taken after the region clip) is a per-call fallback
and not something to lean on when comparing fronts against each other; pass
an explicit shared ``ref_point`` for that.
"""

from __future__ import annotations

import numpy as np

from fit_pcsaft._pure.pareto import coverage, non_dominated


def _clip(F: np.ndarray, region) -> np.ndarray:
    """Per-axis ``[(lo, hi) | None, ...]`` clip -- the module's main point.

    Applied before anything else runs. A ``None`` entry leaves that axis
    unbounded; ``region=None`` (the default) leaves everything unbounded.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if F.shape[1] != 2:
        raise ValueError(
            f"front_quality is two-objective only, got {F.shape[1]} objectives: the "
            "hypervolume ref-point convention, extent_0/extent_1 and _max_gap's sort by "
            "F[:, 0] all assume a curve. pymoo's hv/igd_plus/spacing are n-D, so only "
            "those three need work to lift this."
        )
    if region is None:
        return F
    keep = np.ones(len(F), dtype=bool)
    for axis, bounds in enumerate(region):
        if bounds is None:
            continue
        lo, hi = bounds
        keep &= (F[:, axis] >= lo) & (F[:, axis] <= hi)
    return F[keep]


def _max_gap(F: np.ndarray) -> float:
    """Largest normalized Euclidean step between adjacent front points.

    Adjacent after sorting by the first objective; each axis normalized by
    its own range first -- the same normalization ``_densify`` uses
    (``pareto.py:628-630``) to decide how to spread interpolation budget
    across segments, so this number is on the same footing as that one.
    """
    if len(F) < 2:
        return float("nan")
    F = F[np.argsort(F[:, 0])]
    span = np.ptp(F, axis=0)
    span[span <= 0.0] = 1.0
    return float(np.max(np.linalg.norm(np.diff(F, axis=0) / span, axis=1)))


def front_metrics(
    F, *, reference=None, ref_point=None, region=None
) -> dict[str, float]:
    """Absolute quality indicators for one front, ``region``-clipped first.

    Returns ``{"n_points", "hv", "igd_plus", "spacing", "max_gap",
    "extent_0", "extent_1"}``.

    ``ref_point`` defaults to ``F.max(axis=0) * 1.1`` -- computed from ``F``
    *after* the region clip, so it is only ever comparable to another call
    that used the same explicit ``ref_point`` or clipped to the same region
    with the same front shape; pass ``ref_point`` explicitly to compare
    across fronts. ``igd_plus`` is ``NaN`` when ``reference`` is not given
    (nothing to measure distance to) or is empty after its own clip.
    ``spacing`` and ``max_gap`` need at least 2 points and are ``NaN``
    below that, including the ``n_points == 0`` case below.

    A clip that removes every point is reported, not raised: this is the
    axis Task 3's sweep writes one CSV row per cell along, and a cell whose
    whole front sits outside the region of interest is a legitimate,
    expected row -- ``n_points=0`` with every metric ``NaN``.
    """
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd_plus import IGDPlus
    from pymoo.indicators.spacing import SpacingIndicator

    F = _clip(np.asarray(F, dtype=float), region)
    n = len(F)
    if n == 0:
        return {
            "n_points": 0,
            "hv": float("nan"),
            "igd_plus": float("nan"),
            "spacing": float("nan"),
            "max_gap": float("nan"),
            "extent_0": float("nan"),
            "extent_1": float("nan"),
        }

    if ref_point is None:
        ref_point = F.max(axis=0) * 1.1
    hv = HV(ref_point=np.asarray(ref_point, dtype=float), norm_ref_point=False)(F)

    igd_plus = float("nan")
    if reference is not None:
        ref = _clip(np.asarray(reference, dtype=float), region)
        if len(ref):
            igd_plus = float(IGDPlus(ref)(F))

    spacing = float(SpacingIndicator()(F)) if n >= 2 else float("nan")

    return {
        "n_points": n,
        "hv": float(hv),
        "igd_plus": igd_plus,
        "spacing": spacing,
        "max_gap": _max_gap(F),
        "extent_0": float(np.ptp(F[:, 0])),
        "extent_1": float(np.ptp(F[:, 1])),
    }


def reference_front(*fronts, region=None) -> np.ndarray:
    """Non-dominated union of the given fronts, ``region``-clipped first.

    The yardstick ``front_metrics``' ``reference=`` / IGD+ measures distance
    to: everything any of the given fronts achieved, restricted to the
    region of interest, filtered down to what nothing else in the union
    dominates. Clipping each front before the union (rather than after) is
    what lets the region flip a verdict -- a point that only "wins" via an
    extreme outlier loses its dominating neighbour once that neighbour falls
    outside the region, and a point behind it becomes non-dominated in the
    clipped view. Empty when every front is empty after the clip.
    """
    parts = [_clip(np.asarray(f, dtype=float), region) for f in fronts]
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.empty((0, 2), dtype=float)
    F = np.vstack(parts)
    return F[non_dominated(F)]


def compare_fronts(A, B, *, reference=None, region=None) -> dict[str, float]:
    """Compare two fronts against a shared reference, ``region``-clipped first.

    Returns ``{"coverage_ab", "coverage_ba", "a": {...}, "b": {...}}``, the
    last two being ``front_metrics`` for each side against the shared
    ``reference``. ``coverage_ab``/``coverage_ba`` are the Zitzler C-metric
    (``pareto.coverage``), computed on the region-clipped points -- this is
    the actual fix for the whole-front C-metric's outlier problem described
    in the module docstring, not a different indicator.

    ``reference`` defaults to ``reference_front(A, B, region=region)``; pass
    one explicitly to compare against a bigger yardstick (e.g. the union
    over many runs, not just these two).
    """
    A = _clip(np.asarray(A, dtype=float), region)
    B = _clip(np.asarray(B, dtype=float), region)
    if reference is None:
        reference = reference_front(A, B)  # already region-clipped above
    else:
        reference = _clip(np.asarray(reference, dtype=float), region)

    cov_ab = coverage(A, B) if len(A) and len(B) else float("nan")
    cov_ba = coverage(B, A) if len(A) and len(B) else float("nan")

    return {
        "coverage_ab": cov_ab,
        "coverage_ba": cov_ba,
        # A/B/reference are already region-clipped above; no region= here.
        "a": front_metrics(A, reference=reference),
        "b": front_metrics(B, reference=reference),
    }
