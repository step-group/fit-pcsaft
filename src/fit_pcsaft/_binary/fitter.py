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
from scipy.optimize import least_squares

from fit_pcsaft._binary._utils import (
    _apply_induced_association,
    _build_binary_eos,
    _load_pure_records,
)
from fit_pcsaft._binary.result import BinaryFitResult


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
        Apply the induced-association mixing rule.
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
    ):
        self.id1 = id1
        self.id2 = id2
        self.params_path = params_path
        self.kij_order = kij_order
        self.kij_t_ref = kij_t_ref
        self.kij_bounds = kij_bounds
        self.induced_assoc = induced_assoc
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
    ) -> "BinaryKijFitter":
        """Register a VLE bubble-point dataset."""
        self._sources.append(
            dict(
                type="vle",
                path=path,
                temperature_unit=temperature_unit,
                pressure_unit=pressure_unit,
                t_min=t_min,
                t_max=t_max,
            )
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
    ) -> "BinaryKijFitter":
        """Register an LLE tie-line dataset."""
        self._sources.append(
            dict(
                type="lle",
                path=path,
                temperature_unit=temperature_unit,
                pressure=pressure,
                t_min=t_min,
                t_max=t_max,
                require_both_phases=require_both_phases,
            )
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
    ) -> "BinaryKijFitter":
        """Register a VLLE heteroazeotrope dataset."""
        self._sources.append(
            dict(
                type="vlle",
                path=path,
                temperature_unit=temperature_unit,
                pressure_unit=pressure_unit,
                t_min=t_min,
                t_max=t_max,
            )
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
    ) -> "BinaryKijFitter":
        """Register an SLE solubility dataset."""
        self._sources.append(
            dict(
                type="sle",
                path=path,
                tm=tm,
                delta_hfus=delta_hfus,
                solid_index=solid_index,
                tm2=tm2,
                delta_hfus2=delta_hfus2,
                temperature_unit=temperature_unit,
                t_min=t_min,
                t_max=t_max,
            )
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
            ard_by_type[stype] = float(np.mean(ard_pts))
            total_nfev += res.scipy_result.nfev

        # --- combine and fit polynomial ---
        T_combined = np.concatenate(T_all)
        kij_combined = np.concatenate(kij_all)
        ard_combined = np.concatenate(ard_all)
        source = np.concatenate(source_labels)

        effective_order = min(self.kij_order, len(T_combined) - 1)
        dT = T_combined - self.kij_t_ref

        ols_rev = np.polyfit(dT, kij_combined, effective_order)
        x0_poly = ols_rev[::-1]

        if effective_order == 0 or len(T_combined) == 1:
            kij_coeffs = x0_poly
            poly_resid = kij_combined - sum(c * dT**j for j, c in enumerate(kij_coeffs))
            scipy_result = SimpleNamespace(
                x=kij_coeffs,
                fun=poly_resid,
                cost=float(np.sum(poly_resid**2)) / 2.0,
                success=True,
                nfev=total_nfev,
                message="Combined per-point fitting completed",
            )
        else:

            def _poly_resid(coeffs):
                return sum(c * dT**j for j, c in enumerate(coeffs)) - kij_combined

            rob = least_squares(
                _poly_resid,
                x0_poly,
                loss="cauchy",
                f_scale=0.01,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            total_nfev += rob.nfev
            kij_coeffs = rob.x
            scipy_result = SimpleNamespace(
                x=kij_coeffs,
                fun=rob.fun,
                cost=rob.cost,
                success=rob.success,
                nfev=total_nfev,
                message="Combined per-point fitting completed",
            )

        # overall ARD: weighted by number of points per type
        counts = {t: int(np.sum(source == t)) for t in types_seen}
        n_total = len(T_combined)
        ard = sum(ard_by_type[t] * counts[t] for t in types_seen) / n_total

        # build EOS and records with induced assoc applied
        record1, record2 = _load_pure_records(self.params_path, self.id1, self.id2)
        if self.induced_assoc:
            record1, record2 = _apply_induced_association(record1, record2)
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
