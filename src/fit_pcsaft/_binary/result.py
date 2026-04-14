"""BinaryFitResult dataclass for k_ij fitting results."""
import json
from dataclasses import dataclass
from pathlib import Path

import feos
import numpy as np

from fit_pcsaft._binary._utils import _kij_at_T


@dataclass(frozen=True)
class BinaryFitResult:
    """Result of PC-SAFT binary k_ij fitting.

    Attributes
    ----------
    kij_coeffs : np.ndarray
        Polynomial coefficients [k_ij0, k_ij1, ...], length = kij_order+1.
    kij_t_ref : float
        Reference temperature for the polynomial [K].
    id1 : str
        Identifier of component 1.
    id2 : str
        Identifier of component 2.
    equilibrium_type : str
        One of 'vle', 'lle', 'sle'.
    eos : feos.EquationOfState
        EOS built at kij_coeffs[0] (evaluated at T=t_ref).
    data : dict
        Raw column arrays from CSV.
    ard : float
        ARD % for VLE/LLE or relative ARD on composition for SLE.
    scipy_result : object
        Raw scipy OptimizeResult from least_squares.
    time_elapsed : float
        Wall-clock time for the fit [s].
    """

    kij_coeffs: np.ndarray
    kij_t_ref: float
    id1: str
    id2: str
    equilibrium_type: str
    eos: feos.EquationOfState
    data: dict
    data_full: dict
    ard: float
    scipy_result: object
    time_elapsed: float
    # Henry-specific: pure solvent record (needed for molfrac plot)
    _solvent_record: object = None
    # LLE-specific: pure records needed to rebuild EOS at each T with k_ij(T)
    _record1: object = None
    _record2: object = None
    # SLE-specific (NaN / 0 for VLE/LLE)
    tm_K: float = float("nan")
    delta_hfus_J: float = float("nan")
    solid_index: int = 0
    # Second solid for eutectic systems (NaN when not applicable)
    tm2_K: float = float("nan")
    delta_hfus2_J: float = float("nan")
    # Temperature filter bounds used during fitting [K] (NaN = no bound)
    t_filter_min_K: float = float("nan")
    t_filter_max_K: float = float("nan")

    def kij_at(self, T: float) -> float:
        """Evaluate the k_ij polynomial at temperature T [K]."""
        return _kij_at_T(self.kij_coeffs, T, self.kij_t_ref)

    def to_json(self, path: "Path | str") -> None:
        """Append or update k_ij entry in a JSON file.

        Entry format:
        {"id1": "...", "id2": "...", "type": "vle", "k_ij0": ..., "k_ij1": ..., "t_ref": 293.15}

        Deduplicates on (id1, id2, type).
        """
        path = Path(path)
        entry = {
            "id1": self.id1,
            "id2": self.id2,
            "type": self.equilibrium_type,
            "t_ref": self.kij_t_ref,
        }
        for i, c in enumerate(self.kij_coeffs):
            entry[f"k_ij{i}"] = float(c)

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data = [
                d
                for d in data
                if not (
                    d.get("id1") == self.id1
                    and d.get("id2") == self.id2
                    and d.get("type") == self.equilibrium_type
                )
            ]
        else:
            data = []

        data.append(entry)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_csv(self, path: "Path | str", include_unfitted: bool = False) -> None:
        """Export experimental data and PC-SAFT model curves to CSV files.

        Writes into the directory *path* (created if needed):

        * ``{id1}_{id2}_exp.csv``        — experimental data used for fitting
        * ``{id1}_{id2}_model.csv``       — fitted model curve(s)
        * ``{id1}_{id2}_model_kij0.csv``  — k_ij=0 predictive curve (when
          ``include_unfitted=True`` and pure records are available)

        Columns depend on equilibrium type:

        * **VLE** isobaric:  ``T, x1_bubble, x1_dew``
        * **VLE** isothermal: ``P, x1_bubble, x1_dew``  (one block per T)
        * **LLE**:  ``T, x1_I, x1_II``
        * **SLE**:  ``T, x1``
        * **VLLE** / **VLE+LLE**: all relevant columns from each sub-type

        Parameters
        ----------
        path : Path or str
            Output directory.
        include_unfitted : bool
            If True, also write the k_ij=0 predictive model curve.
        """
        import polars as pl
        import si_units as si

        import feos

        from fit_pcsaft._binary._plot import (
            _find_eutectic,
            _find_heteroazeotrope,
            _lle_curve_kij_T,
            _lle_feed_z1,
            _normalize_result_for_type,
            _sle_fixed_point,
            _vle_branch_isobaric,
            _vlle_locus,
        )

        def _eos_kij0():
            if self._record1 is None or self._record2 is None:
                return None
            try:
                params = feos.Parameters.new_binary([self._record1, self._record2], k_ij=0.0)
                return feos.EquationOfState.pcsaft(params, max_iter_cross_assoc=100)
            except Exception:
                return None

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        stem = f"{self.id1}_{self.id2}"

        eq = self.equilibrium_type
        _tokens = frozenset(eq.replace("_", "+").split("+"))

        # --- experimental CSV ---
        _EXP_META = {"T_kij", "kij_pointwise", "ard_pointwise", "ard_pointwise_poly", "source"}
        exp_rows = {
            k: v.tolist()
            for k, v in self.data.items()
            if k not in _EXP_META and not k.startswith("ard_") and hasattr(v, "tolist")
        }
        if exp_rows:
            n = max(len(v) for v in exp_rows.values())
            padded = {k: v + [None] * (n - len(v)) for k, v in exp_rows.items()}
            pl.DataFrame(padded).write_csv(path / f"{stem}_exp.csv")

        # --- helpers ---
        def _write_model(curves: dict, suffix: str = "model") -> None:
            if not curves:
                return
            n = max(len(v) for v in curves.values())
            padded = {k: v + [None] * (n - len(v)) for k, v in curves.items()}
            pl.DataFrame(padded).write_csv(path / f"{stem}_{suffix}.csv")

        def _vle_curves(eos, pressure_unit=si.KILO * si.PASCAL) -> dict:
            import feos
            data = _normalize_result_for_type(self, "vle").data
            T_data = data["T"].astype(float)
            P_data = data["P"].astype(float)
            is_isobaric = np.std(P_data) / np.mean(P_data) < 0.05
            if is_isobaric:
                P_mean = float(np.mean(P_data))
                vle = feos.PhaseDiagram.binary_vle(eos, P_mean * pressure_unit, npoints=200)
                return {
                    "T":          (vle.liquid.temperature / si.KELVIN).tolist(),
                    "x1_bubble":  vle.liquid.molefracs[:, 0].tolist(),
                    "x1_dew":     vle.vapor.molefracs[:, 0].tolist(),
                }
            else:
                T_unique = np.unique(np.round(T_data, 4))
                T_out, x1_b_out, x1_d_out = [], [], []
                for T_iso in T_unique:
                    try:
                        vle = feos.PhaseDiagram.binary_vle(eos, T_iso * si.KELVIN, npoints=200)
                        T_out.extend((vle.liquid.temperature / si.KELVIN).tolist())
                        x1_b_out.extend(vle.liquid.molefracs[:, 0].tolist())
                        x1_d_out.extend(vle.vapor.molefracs[:, 0].tolist())
                    except Exception:
                        pass
                return {"T": T_out, "x1_bubble": x1_b_out, "x1_dew": x1_d_out}

        def _lle_curves(result_obj) -> dict:
            z1 = _lle_feed_z1(_normalize_result_for_type(result_obj, "lle"))
            data = _normalize_result_for_type(result_obj, "lle").data
            T_arr = data.get("T", np.array([])).astype(float)
            T_min = float(T_arr.min()) if len(T_arr) else 273.15
            T_max = float(T_arr.max()) if len(T_arr) else 373.15
            T_out, x_I, x_II = _lle_curve_kij_T(result_obj, z1, T_min, T_max)
            return {
                "T":     T_out.tolist(),
                "x1_I":  list(x_I),
                "x1_II": list(x_II),
            }

        def _sle_curves(result_obj) -> dict:
            Tm_K      = result_obj.tm_K
            dHfus_J   = result_obj.delta_hfus_J
            solid_idx = result_obj.solid_index
            eutectic  = not np.isnan(result_obj.tm2_K)
            data      = _normalize_result_for_type(result_obj, "sle").data
            T_data    = data["T"].astype(float)
            x1_data   = data["x1"].astype(float)
            curve_T_min = float(T_data.min()) * 0.995

            if eutectic:
                Tm2_K    = result_obj.tm2_K
                dHfus2_J = result_obj.delta_hfus2_J
                si_idx2  = 1 - solid_idx
                T_eut, x1_eut = _find_eutectic(
                    result_obj.eos, Tm_K, dHfus_J, solid_idx,
                    Tm2_K, dHfus2_J, si_idx2,
                )
                T_start = T_eut if not np.isnan(T_eut) else curve_T_min
                x0_eut  = x1_eut if not np.isnan(x1_eut) else float(x1_data[np.argmin(T_data)])

                T1_out, x1_out_1 = [], []
                T2_out, x1_out_2 = [], []
                for T_i in np.linspace(T_start, Tm_K, 120):
                    try:
                        x1 = _sle_fixed_point(result_obj.eos, T_i, Tm_K, dHfus_J, solid_idx, x0_eut)
                        if not np.isnan(x1) and 0.0 <= x1 <= 1.0:
                            T1_out.append(float(T_i)); x1_out_1.append(x1)
                    except Exception:
                        pass
                for T_i in np.linspace(T_start, Tm2_K, 120):
                    try:
                        x1 = _sle_fixed_point(result_obj.eos, T_i, Tm2_K, dHfus2_J, si_idx2, x0_eut)
                        if not np.isnan(x1) and 0.0 <= x1 <= 1.0:
                            T2_out.append(float(T_i)); x1_out_2.append(x1)
                    except Exception:
                        pass
                T_all  = T1_out  + T2_out
                x1_all = x1_out_1 + x1_out_2
                return {"T": T_all, "x1": x1_all}
            else:
                x0 = float(x1_data[np.argmin(T_data)])
                T_out, x1_out = [], []
                for T_i in np.linspace(curve_T_min, Tm_K, 120):
                    try:
                        x1 = _sle_fixed_point(result_obj.eos, T_i, Tm_K, dHfus_J, solid_idx, x0)
                        if not np.isnan(x1) and 0.0 <= x1 <= 1.0:
                            T_out.append(float(T_i)); x1_out.append(x1); x0 = x1
                    except Exception:
                        pass
                return {"T": T_out, "x1": x1_out}

        def _vlle_curves(result_obj, pressure_unit=si.KILO * si.PASCAL) -> dict:
            data = result_obj.data
            P_key = next((k for k in ("P", "vlle_P") if k in data), None)
            P_arr = data[P_key].astype(float) if P_key else np.array([101.325])
            P_si  = np.sort(np.unique(P_arr * pressure_unit / si.PASCAL))[::-1]  # descending

            x_I_exp  = float(np.nanmean(data.get("x1_I", data.get("vlle_x1_I", np.array([0.05])))))
            x_II_exp = float(np.nanmean(data.get("x1_II", data.get("vlle_x1_II", np.array([0.90])))))
            T_init   = float(np.nanmean(data.get("T", data.get("vlle_T", np.array([350.0])))))

            T_arr, xI_arr, xII_arr, y1_arr = _vlle_locus(
                result_obj, P_si, x_I_init=x_I_exp, x_II_init=x_II_exp, T_init=T_init
            )
            P_plot = P_si[::-1] / (pressure_unit / si.PASCAL)
            return {
                "T":    T_arr[::-1].tolist(),
                "P":    P_plot.tolist(),
                "x1_I":  xI_arr[::-1].tolist(),
                "x1_II": xII_arr[::-1].tolist(),
                "y1":    y1_arr[::-1].tolist(),
            }

        # --- dispatch ---
        pressure_unit = si.KILO * si.PASCAL  # consistent with default plot units

        if _tokens == {"vle"}:
            _write_model(_vle_curves(self.eos, pressure_unit))
            if include_unfitted:
                eos0 = _eos_kij0()
                if eos0 is not None:
                    _write_model(_vle_curves(eos0, pressure_unit), "model_kij0")

        elif _tokens == {"lle"}:
            _write_model(_lle_curves(self))
            if include_unfitted and _eos_kij0() is not None:
                import dataclasses
                self0 = dataclasses.replace(self, kij_coeffs=np.array([0.0]))
                _write_model(_lle_curves(self0), "model_kij0")

        elif _tokens == {"sle"}:
            _write_model(_sle_curves(self))

        elif _tokens == {"vlle"}:
            _write_model(_vlle_curves(self, pressure_unit))

        elif "vle" in _tokens:
            # VLE+LLE (and variants)
            vle_c = _vle_curves(self.eos, pressure_unit)
            lle_c = _lle_curves(self)
            combined = {f"vle_{k}": v for k, v in vle_c.items()}
            combined.update({f"lle_{k}": v for k, v in lle_c.items()})
            _write_model(combined)
            if include_unfitted and _eos_kij0() is not None:
                import dataclasses
                self0 = dataclasses.replace(self, kij_coeffs=np.array([0.0]))
                lle0  = _lle_curves(self0)
                eos0  = _eos_kij0()
                vle0  = _vle_curves(eos0, pressure_unit) if eos0 is not None else {}
                combined0 = {f"vle_{k}": v for k, v in vle0.items()}
                combined0.update({f"lle_{k}": v for k, v in lle0.items()})
                _write_model(combined0, "model_kij0")

        elif "lle" in _tokens:
            _write_model(_lle_curves(self))

    def plot(self, path=None, temperature_unit=None, pressure_unit=None, henry_unit=None, plot_unfitted: bool = False):
        """Plot binary equilibrium diagram with experimental data overlay.

        Parameters
        ----------
        path : str or Path, optional
            Save path for the figure.
        temperature_unit : si.SIObject, optional
            Unit of the T column in the CSV (default: si.KELVIN).
        pressure_unit : si.SIObject, optional
            Unit of the P column in the CSV, VLE only (default: si.KILO * si.PASCAL).
        plot_unfitted : bool, optional
            If True, overlay a "predictive" curve computed with k_ij = 0.0.
        """
        import si_units as si

        from fit_pcsaft._binary._plot import _plot_binary

        return _plot_binary(
            self,
            path=path,
            temperature_unit=si.KELVIN if temperature_unit is None else temperature_unit,
            pressure_unit=si.KILO * si.PASCAL if pressure_unit is None else pressure_unit,
            henry_unit=si.MEGA * si.PASCAL if henry_unit is None else henry_unit,
            plot_unfitted=plot_unfitted,
        )

    def plot_kij(self, path=None):
        """Plot point-wise k_ij vs T and the polynomial fit.

        Available when kij_per_point=True was used (LLE or VLE).

        Parameters
        ----------
        path : str or Path, optional
            Save path for the figure.
        """
        if "T_kij" not in self.data or "kij_pointwise" not in self.data:
            raise ValueError(
                "No point-wise k_ij data available. Re-run with kij_per_point=True."
            )
        from fit_pcsaft._binary._plot import _plot_kij_vs_T

        return _plot_kij_vs_T(
            self.data["T_kij"],
            self.data["kij_pointwise"],
            self.kij_coeffs,
            self.kij_t_ref,
            self.id1,
            self.id2,
            equilibrium_type=self.equilibrium_type,
            ard_pw=self.data.get("ard_pointwise"),
            source=self.data.get("source"),
            path=path,
        )

    def __str__(self) -> str:
        """Pretty-print k_ij result."""
        lines = [
            f"Binary k_ij fit ({self.equilibrium_type.upper()}): {self.id1} + {self.id2}",
            "Fitted k_ij polynomial:",
            f"  k_ij0 (constant):    {self.kij_coeffs[0]:.6f}",
        ]
        for i in range(1, len(self.kij_coeffs)):
            lines.append(f"  k_ij{i} (T^{i} coeff):  {self.kij_coeffs[i]:.6e}")
        if len(self.kij_coeffs) > 1:
            lines.append(f"  T_ref:               {self.kij_t_ref:.2f} K")

        n = sum(len(v) for v in self.data.values() if hasattr(v, "__len__")) // max(
            1, len(self.data)
        )
        rms = np.sqrt(2.0 * self.scipy_result.cost / max(1, len(self.scipy_result.fun)))
        lines.extend(["", "Fitting quality:"])
        if "ard_vle" in self.data and "ard_lle" in self.data:
            lines.append(f"  ARD VLE:             {float(self.data['ard_vle'][0]):.2f}%")
            lines.append(f"  ARD LLE:             {float(self.data['ard_lle'][0]):.2f}%")
            lines.append(f"  ARD combined:        {self.ard:.2f}%")
        else:
            lines.append(f"  ARD:                 {self.ard:.2f}%")
        lines.extend([
            f"  RMS weighted resid.: {rms:.4f}",
            f"  Converged:           {self.scipy_result.success}",
            f"  Function evals:      {self.scipy_result.nfev}",
            f"  Time elapsed:        {self.time_elapsed:.2f} s",
        ])
        return "\n".join(lines)
