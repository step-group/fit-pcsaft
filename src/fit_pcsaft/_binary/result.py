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
        """Plot point-wise k_ij vs T and the polynomial fit (LLE only).

        Parameters
        ----------
        path : str or Path, optional
            Save path for the figure.
        """
        if self.equilibrium_type != "lle":
            raise ValueError("plot_kij is only available for LLE fits")
        if "T_kij" not in self.data or "kij_pointwise" not in self.data:
            raise ValueError("No point-wise k_ij data available in this result")
        from fit_pcsaft._binary._plot import _plot_kij_vs_T

        return _plot_kij_vs_T(
            self.data["T_kij"],
            self.data["kij_pointwise"],
            self.kij_coeffs,
            self.kij_t_ref,
            self.id1,
            self.id2,
            ard_pw=self.data.get("ard_pointwise"),
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
        lines.extend(
            [
                "",
                "Fitting quality:",
                f"  ARD:                 {self.ard:.2f}%",
                f"  RMS weighted resid.: {rms:.4f}",
                f"  Converged:           {self.scipy_result.success}",
                f"  Function evals:      {self.scipy_result.nfev}",
                f"  Time elapsed:        {self.time_elapsed:.2f} s",
            ]
        )
        return "\n".join(lines)
