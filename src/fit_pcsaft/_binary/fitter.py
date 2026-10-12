"""Generic binary k_ij fitter accepting any combination of data types.

Usage::

    from fit_pcsaft import BinaryKijFitter
    import si_units as si

    result = (
        BinaryKijFitter("mibk", "water", params_path,
                        kij_order=1, kij_t_ref=333.15, induced_assoc=True)
        .add_vle(vle_path, pressure_unit=si.KILO * si.PASCAL)
        .add_lle(lle_path)
        .add_vlle(vlle_path, pressure_unit=si.KILO * si.PASCAL)
        .fit()
    )

All registered data types are fitted per-point independently (each yields a
set of (T, k_ij) pairs), then a single k_ij(T) polynomial is fitted to the
combined collection.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import si_units as si

from fit_pcsaft._binary._utils import (
    _apply_induced_association,
    _build_binary_eos,
    _comp_resid,
    _fit_kij_polynomial,
    _kij_at_T,
    _load_pure_records,
)
from fit_pcsaft._binary.result import BinaryFitResult


def _ard_at_coeffs(res: BinaryFitResult, kij_coeffs, kij_t_ref: float,
                   src: "dict | None" = None) -> float:
    """ARD [%] of a single-type result's data at the k_ij polynomial `kij_coeffs`.

    `res` is a vle/lle/vlle/sle result (its `data`, records and LLE/SLE settings
    are reused); `src` is a `BinaryKijFitter` source spec, read for
    temperature_unit, pressure_unit, require_both_phases, ucst_target and
    relative_residuals (defaults: K, kPa, True, False, True, as add_lle/add_vle).
    Every data row is scored, 100 * mean|residual| per point (VLLE: the T
    residual only), points below 0.01 % dropped as machine-precision exact fits,
    failed predictions counted as 100 %, as vle.py/lle.py's own post-polynomial
    ARD counts them; vlle.py and sle.py never scored their polynomial. k_ij(T)
    is evaluated in kelvin.
    """
    src = src or {}
    tu = src.get("temperature_unit", si.KELVIN)
    pu = src.get("pressure_unit", si.KILO * si.PASCAL)
    rel = src.get("relative_residuals", True)
    t_scale = float(tu / si.KELVIN)
    d, r1, r2, eq = res.data, res._record1, res._record2, res.equilibrium_type
    T_csv = d["T"].astype(float)
    ards: list[float] = []

    if eq == "vle":
        from fit_pcsaft._binary.vle import _residuals_vle_point
        for i, T in enumerate(T_csv):
            r = _residuals_vle_point(
                [_kij_at_T(kij_coeffs, T * t_scale, kij_t_ref)], T, float(d["P"][i]),
                float(d["x1"][i]), float(d["y1"][i]) if "y1" in d else None,
                r1, r2, tu, pu,
            )
            ards.append(100.0 * float(np.mean(np.abs(r))))  # failure: penalty of ones
    elif eq == "lle":
        from fit_pcsaft._binary.lle import (
            _LLE_FEEDS,
            _exp_feeds,
            _individual_lle_points,
            _residuals_at_T,
        )
        pts = _individual_lle_points(
            T_csv, d["x1_I"].astype(float) if "x1_I" in d else None,
            d["x1_II"].astype(float) if "x1_II" in d else None, t_scale,
        )
        if src.get("require_both_phases", True):
            pts = [p for p in pts if p[1] is not None and p[2] is not None]
        T_anchor_K = min(p[0] for p in pts)
        if src.get("ucst_target", False):
            T_ucst = max(p[0] for p in pts)
            pts = [p for p in pts if abs(p[0] - T_ucst) < 1e-3]
        warm: list = [None, None, None]
        for T_K, exp_I, exp_II in pts:
            r = _residuals_at_T(
                [_kij_at_T(kij_coeffs, T_K, kij_t_ref)], T_K, exp_I, exp_II, r1, r2,
                res.lle_pressure_bar * si.BAR, _exp_feeds(exp_I, exp_II) + _LLE_FEEDS,
                T_anchor_K=T_anchor_K, relative_residuals=rel,
                require_liquid_phases=res.lle_require_liquid_phases, warm=warm,
                binary_assoc=res._binary_assoc, record_at_T=res._record_at_T,
            )
            ards.append(100.0 * float(np.mean(np.abs(r))))  # failure: penalty of ones
    elif eq == "vlle":
        from fit_pcsaft._binary.vlle import _predict_vlle_point
        p_scale = float(pu / si.PASCAL)
        for i, T in enumerate(T_csv):
            T_K = T * t_scale
            T_pred = _predict_vlle_point(
                r1, r2, _kij_at_T(kij_coeffs, T_K, kij_t_ref), T_K,
                float(d["P"][i]) * p_scale,
                float(d["x1_I"][i]) if "x1_I" in d else 0.01,
                float(d["x1_II"][i]) if "x1_II" in d else 0.90,
                binary_assoc=res._binary_assoc,
            )[0]
            # T only, as fit_kij_vlle scores its points
            ards.append(100.0 * abs((T_pred - T_K) / T_K if rel else T_pred - T_K)
                        if np.isfinite(T_pred) else 100.0)
    elif eq == "sle":
        from fit_pcsaft._binary.sle import _predict_x1_for
        solids = [(res.solid_index, res.tm_K, res.delta_hfus_J)]
        if np.isfinite(res.tm2_K):
            solids.append((1 - res.solid_index, res.tm2_K, res.delta_hfus2_J))
        for T, x1 in zip(T_csv * t_scale, d["x1"].astype(float)):
            try:
                eos = _build_binary_eos(
                    r1, r2, _kij_at_T(kij_coeffs, T, kij_t_ref), res._binary_assoc,
                    T_K=T, record_at_T=res._record_at_T,
                )
                resid = min((_comp_resid(_predict_x1_for(eos, T, x1, *s), x1, False)
                             for s in solids), key=abs)
            except Exception:  # noqa: BLE001 -- solver failure counts as 100 %
                resid = 1.0
            ards.append(100.0 * abs(resid))
    else:
        raise NotImplementedError(f"no ARD evaluation for {eq!r}")

    arr = np.array(ards)
    if len(arr) == 0:
        return float("nan")
    meaningful = arr[arr > 0.01]
    return float(meaningful.mean()) if len(meaningful) > 0 else float(arr.mean())


class BinaryKijFitter:
    """Fluent builder for fitting k_ij(T) from one or more equilibrium datasets.

    Parameters
    ----------
    id1, id2 : str
        Component identifiers matching names in the params JSON file.
    params_path : Path | str | list
        Feos-compatible JSON parameter file(s).
    kij_order : int
        Polynomial order for k_ij(T): 0=constant, 1=linear, 2=quadratic.
    kij_t_ref : float
        Reference temperature for the k_ij polynomial [K].
    kij_bounds : tuple
        (lower, upper) bounds for k_ij at each data point.
    induced_assoc : bool
        If True, apply the induced-association rule. Requires exactly one
        self-associating component (with epsilon_k_ab > 0). The non-associating
        component receives kappa_ab copied from the self-associating component
        and epsilon_k_ab = `induced_epsilon_k_ab`.
    induced_sites : str
        "2B" (default): the non-associating component is rewritten as na = nb = 1.
        "own": its declared na/nb are kept (a ketone's acceptor-only site stays
        so). See `_apply_induced_association`.
    induced_epsilon_k_ab : float
        Site energy written on the non-associating component, in K. Through the
        arithmetic-mean combining rule it is the fitted cross-association
        parameter of Rehner, Bardow & Gross (2023). Default 0.0.
    """

    def __init__(
        self,
        id1: str,
        id2: str,
        params_path: "Path | str | list",
        *,
        kij_order: int = 0,
        kij_t_ref: float = 298.15,
        kij_bounds: tuple = (-0.3, 0.3),
        induced_assoc: bool = False,
        induced_sites: str = "2B",
        induced_epsilon_k_ab: float = 0.0,
    ):
        self.id1 = id1
        self.id2 = id2
        self.params_path = params_path
        self.kij_order = kij_order
        self.kij_t_ref = kij_t_ref
        self.kij_bounds = kij_bounds
        self.induced_assoc = induced_assoc
        self.induced_sites = induced_sites
        self.induced_epsilon_k_ab = induced_epsilon_k_ab
        self._sources: list[dict] = []

    # ------------------------------------------------------------------
    # add_* methods  (each returns self for method chaining)
    # ------------------------------------------------------------------

    def add_vle(
        self,
        path: "Path | str",
        *,
        temperature_unit=si.KELVIN,
        pressure_unit=si.KILO * si.PASCAL,
        t_min=None,
        t_max=None,
        relative_residuals: bool = True,
    ) -> "BinaryKijFitter":
        """Register a VLE bubble-point dataset."""
        self._sources.append(
            {
                "type": "vle",
                "path": path,
                "temperature_unit": temperature_unit,
                "pressure_unit": pressure_unit,
                "t_min": t_min,
                "t_max": t_max,
                "relative_residuals": relative_residuals,
            }
        )
        return self

    def add_lle(
        self,
        path: "Path | str",
        *,
        temperature_unit=si.KELVIN,
        pressure: "si.SIObject" = 1.01325 * si.BAR,
        t_min=None,
        t_max=None,
        require_both_phases: bool = True,
        ucst_target: bool = False,
        relative_residuals: bool = True,
    ) -> "BinaryKijFitter":
        """Register an LLE tie-line dataset."""
        self._sources.append(
            {
                "type": "lle",
                "path": path,
                "temperature_unit": temperature_unit,
                "pressure": pressure,
                "t_min": t_min,
                "t_max": t_max,
                "require_both_phases": require_both_phases,
                "ucst_target": ucst_target,
                "relative_residuals": relative_residuals,
            }
        )
        return self

    def add_vlle(
        self,
        path: "Path | str",
        *,
        temperature_unit=si.KELVIN,
        pressure_unit=si.KILO * si.PASCAL,
        t_min=None,
        t_max=None,
        relative_residuals: bool = True,
    ) -> "BinaryKijFitter":
        """Register a VLLE heteroazeotrope dataset."""
        self._sources.append(
            {
                "type": "vlle",
                "path": path,
                "temperature_unit": temperature_unit,
                "pressure_unit": pressure_unit,
                "t_min": t_min,
                "t_max": t_max,
                "relative_residuals": relative_residuals,
            }
        )
        return self

    def add_sle(
        self,
        path: "Path | str",
        tm: "si.SIObject",
        delta_hfus: "si.SIObject",
        *,
        solid_index: int = 0,
        tm2: "si.SIObject | None" = None,
        delta_hfus2: "si.SIObject | None" = None,
        temperature_unit=si.KELVIN,
        t_min=None,
        t_max=None,
        relative_residuals: bool = True,
    ) -> "BinaryKijFitter":
        """Register an SLE solubility dataset."""
        self._sources.append(
            {
                "type": "sle",
                "path": path,
                "tm": tm,
                "delta_hfus": delta_hfus,
                "solid_index": solid_index,
                "tm2": tm2,
                "delta_hfus2": delta_hfus2,
                "temperature_unit": temperature_unit,
                "t_min": t_min,
                "t_max": t_max,
                "relative_residuals": relative_residuals,
            }
        )
        return self

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(self) -> BinaryFitResult:
        """Run per-point fitting for all registered sources, then fit a single
        k_ij(T) polynomial to the combined (T, k_ij) pairs.

        Returns
        -------
        BinaryFitResult
            ``equilibrium_type`` is the ``"+"``-joined list of source types,
            e.g. ``"vle+lle"`` or ``"vle+lle+vlle"``.
        """
        if not self._sources:
            raise ValueError(
                "No data sources registered. Call add_vle(), add_lle(), etc."
            )

        t0 = time.perf_counter()

        from fit_pcsaft._binary.lle import fit_kij_lle
        from fit_pcsaft._binary.sle import fit_kij_sle
        from fit_pcsaft._binary.vle import fit_kij_vle
        from fit_pcsaft._binary.vlle import fit_kij_vlle

        T_all: list[np.ndarray] = []
        kij_all: list[np.ndarray] = []
        ard_all: list[np.ndarray] = []
        source_labels: list[np.ndarray] = []
        total_nfev = 0
        data_exp: dict[str, np.ndarray] = {}
        ard_by_type: dict[str, float] = {}
        types_seen: list[str] = []
        sub_fits: list = []

        # --- run per-point fitting for each source ---
        for src in self._sources:
            stype = src["type"]
            if stype not in types_seen:
                types_seen.append(stype)

            if stype == "vle":
                res = fit_kij_vle(
                    self.id1,
                    self.id2,
                    src["path"],
                    self.params_path,
                    kij_order=0,
                    kij_t_ref=self.kij_t_ref,
                    kij_bounds=self.kij_bounds,
                    temperature_unit=src["temperature_unit"],
                    pressure_unit=src["pressure_unit"],
                    t_min=src["t_min"],
                    t_max=src["t_max"],
                    kij_per_point=True,
                    induced_assoc=self.induced_assoc,
                    induced_sites=self.induced_sites,
                    induced_epsilon_k_ab=self.induced_epsilon_k_ab,
                    relative_residuals=src["relative_residuals"],
                )
                for k in ("T", "P", "x1", "y1"):
                    if k in res.data:
                        data_exp[f"vle_{k}"] = res.data[k]

            elif stype == "lle":
                res = fit_kij_lle(
                    self.id1,
                    self.id2,
                    src["path"],
                    self.params_path,
                    kij_order=0,
                    kij_t_ref=self.kij_t_ref,
                    kij_bounds=self.kij_bounds,
                    temperature_unit=src["temperature_unit"],
                    pressure=src["pressure"],
                    t_min=src["t_min"],
                    t_max=src["t_max"],
                    require_both_phases=src["require_both_phases"],
                    kij_per_point=True,
                    induced_assoc=self.induced_assoc,
                    induced_sites=self.induced_sites,
                    induced_epsilon_k_ab=self.induced_epsilon_k_ab,
                    ucst_target=src["ucst_target"],
                    relative_residuals=src["relative_residuals"],
                )
                for k in ("T", "x1_I", "x1_II"):
                    if k in res.data:
                        data_exp[f"lle_{k}"] = res.data[k]

            elif stype == "vlle":
                res = fit_kij_vlle(
                    self.id1,
                    self.id2,
                    src["path"],
                    self.params_path,
                    kij_order=0,
                    kij_t_ref=self.kij_t_ref,
                    kij_bounds=self.kij_bounds,
                    temperature_unit=src["temperature_unit"],
                    pressure_unit=src["pressure_unit"],
                    t_min=src["t_min"],
                    t_max=src["t_max"],
                    induced_assoc=self.induced_assoc,
                    induced_sites=self.induced_sites,
                    induced_epsilon_k_ab=self.induced_epsilon_k_ab,
                    relative_residuals=src["relative_residuals"],
                )
                for k in ("T", "P", "x1_I", "x1_II", "y1"):
                    if k in res.data:
                        data_exp[f"vlle_{k}"] = res.data[k]

            elif stype == "sle":
                res = fit_kij_sle(
                    self.id1,
                    self.id2,
                    src["path"],
                    self.params_path,
                    tm=src["tm"],
                    delta_hfus=src["delta_hfus"],
                    solid_index=src["solid_index"],
                    tm2=src["tm2"],
                    delta_hfus2=src["delta_hfus2"],
                    kij_order=0,
                    kij_t_ref=self.kij_t_ref,
                    kij_bounds=self.kij_bounds,
                    temperature_unit=src["temperature_unit"],
                    t_min=src["t_min"],
                    t_max=src["t_max"],
                    kij_per_point=True,
                )
                for k in ("T", "x1"):
                    if k in res.data:
                        data_exp[f"sle_{k}"] = res.data[k]

            else:
                raise ValueError(f"Unknown source type: {stype!r}")

            T_pts = res.data["T_kij"]
            kij_pts = res.data["kij_pointwise"]
            ard_pts = res.data["ard_pointwise"]

            T_all.append(T_pts)
            kij_all.append(kij_pts)
            ard_all.append(ard_pts)
            source_labels.append(np.full(len(T_pts), stype))
            sub_fits.append((src, res))
            total_nfev += res.scipy_result.nfev

        # --- combine and fit polynomial ---
        T_combined = np.concatenate(T_all)
        kij_combined = np.concatenate(kij_all)
        ard_combined = np.concatenate(ard_all)
        source = np.concatenate(source_labels)

        kij_coeffs, poly_resid = _fit_kij_polynomial(
            T_combined, kij_combined, ard_combined, self.kij_order, self.kij_t_ref
        )
        scipy_result = SimpleNamespace(
            x=kij_coeffs,
            fun=poly_resid,
            cost=float(np.sum(poly_resid**2)) / 2.0,
            success=True,
            nfev=total_nfev,
            message="Combined per-point fitting completed",
        )
        # Each source scored at the combined polynomial, not its own sub-fit's.
        for src, res in sub_fits:
            ard_by_type[src["type"]] = _ard_at_coeffs(res, kij_coeffs, self.kij_t_ref, src)

        # overall ARD: weighted by number of points per type
        counts = {t: int(np.sum(source == t)) for t in types_seen}
        n_total = len(T_combined)
        ard = sum(ard_by_type[t] * counts[t] for t in types_seen) / n_total

        # build EOS and records with induced assoc applied
        record1, record2 = _load_pure_records(self.params_path, self.id1, self.id2)
        if self.induced_assoc:
            record1, record2 = _apply_induced_association(
                record1,
                record2,
                sites=self.induced_sites,
                epsilon_k_ab=self.induced_epsilon_k_ab,
            )
        eos_ref = _build_binary_eos(record1, record2, float(kij_coeffs[0]))

        eq_type = "+".join(types_seen)

        return BinaryFitResult(
            kij_coeffs=kij_coeffs,
            kij_t_ref=self.kij_t_ref,
            id1=self.id1,
            id2=self.id2,
            equilibrium_type=eq_type,
            eos=eos_ref,
            data={
                "T_kij": T_combined,
                "kij_pointwise": kij_combined,
                "source": source,
                "ard_pointwise": ard_combined,
                **{f"ard_{t}": np.array([ard_by_type[t]]) for t in types_seen},
                **data_exp,
            },
            data_full={},
            ard=ard,
            scipy_result=scipy_result,
            time_elapsed=time.perf_counter() - t0,
            _record1=record1,
            _record2=record2,
        )
