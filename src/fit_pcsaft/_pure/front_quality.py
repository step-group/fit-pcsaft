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
``compare_fronts`` is a module-level ``__getattr__`` (PEP 562) rather than a
top-level ``from .front_quality import ...`` -- the straight-line version is
NOT safe here: the tests (and any other caller) import this module directly
as ``fit_pcsaft._pure.front_quality``, and when that happens first, Python
starts executing this file, hits its own top-level
``from fit_pcsaft._pure.pareto import coverage, non_dominated``, and begins
importing ``pareto.py`` from scratch -- which would then reach a top-level
``from .front_quality import front_metrics, ...`` line while this module is
still mid-import and hasn't defined ``front_metrics`` yet, raising
``ImportError: cannot import name 'front_metrics' from partially initialized
module``. The lazy ``__getattr__`` defers that import until something actually
does ``pareto.front_metrics(...)``, by which point both modules have finished
loading.

HV note: ``ref_point`` is deliberately passed with ``norm_ref_point=False``.
Reading pymoo 0.6.2's ``pymoo/indicators/hv.py`` and
``pymoo/util/normalization.py`` shows ``norm_ref_point`` only has an effect
when ``zero_to_one=True`` is also passed to ``Hypervolume`` -- which nothing
here does -- so ``self.normalization`` is always ``NoNormalization()`` and
``norm_ref_point`` is currently a no-op either way (confirmed empirically:
identical HV for the same front under both settings). ``False`` is set anyway
so a future pymoo release that makes the flag do something again can't
silently start renormalizing ``ref_point`` per front. That renormalization is
exactly what Task 3 cannot tolerate: it scores ~45 fronts against one shared
``ref_point`` specifically so the hypervolumes are comparable across them, and
a per-front renormalization would silently answer a different question. See
``test_hv_is_comparable_across_fronts_with_a_shared_ref_point``, which pins
it: two fronts, one strictly dominating the other, scored with the same
``ref_point``, must come back with the dominating front's HV strictly higher.

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

    cov_ab = coverage(A, B) if len(A) and len(B) else float("nan")
    cov_ba = coverage(B, A) if len(A) and len(B) else float("nan")

    return {
        "coverage_ab": cov_ab,
        "coverage_ba": cov_ba,
        "a": front_metrics(A, reference=reference, region=region),
        "b": front_metrics(B, reference=reference, region=region),
    }
