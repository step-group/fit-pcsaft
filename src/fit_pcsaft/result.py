import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import feos
import numpy as np

from fit_pcsaft._types import Compound, ModelSpec, PureData, Units


@dataclass(frozen=True)
class FitResult:
    """Result of PC-SAFT pure component fitting.

    Attributes
    ----------
        params : dict
            Dictionary of fitted parameters {"m", "sigma", "epsilon_k", ...}

        eos : object
            Fitted feos.EquationOfState object

        data : PureData
            Experimental data used for fitting

        compound : Compound
            Compound identifier and molar weight

        spec : ModelSpec
            Model specification (mu, q, na, nb)

        units : Units
            Units used for experimental data

        ard_psat : float
            Average relative deviation for vapor pressure (%)

        ard_rho : float
            Average relative deviation for liquid density (%)

        ard_hvap : float
            Average relative deviation for enthalpy of vaporization (%). nan if no hvap data.

        scipy_result : object
            Raw scipy OptimizeResult from least_squares

    Methods
    -------
        to_json(path: Path | str) -> None
            Append fitted parameters to a feos-compatible JSON parameter file.
        ```
    """

    params: dict
    eos: feos.EquationOfState
    data: PureData
    compound: Compound
    spec: ModelSpec
    units: Units
    ard_psat: float
    ard_rho: float
    ard_hvap: float
    scipy_result: object
    time_elapsed: float
    input_name: str = ""

    def to_json(self, path: "Path | str") -> None:
        """Append or update fitted parameters in a feos-compatible JSON parameter file.

        If an entry with the same CAS number or name already exists, it is replaced.
        If the file does not exist, a new single-entry list is created.
        Parent directories are created automatically if needed.

        Parameters
        ----------
        path : Path or str
            Path to the JSON parameter file.

        Raises
        ------
        json.JSONDecodeError
            If the existing file contains invalid JSON.

        Examples
        --------
        ```python
        >>> result = fit_pure(
        ...     id="ethanol",
        ...     psat_path=psat_path,
        ...     density_path=density_path,
        ...     na=1,
        ...     nb=1,
        ... )
        >>> result.to_json("path/to/parameters.json")
        ```

        ```json
        {
            "identifier": {
            "cas": "64-17-5",
            "name": "ethanol",
            "iupac_name": "ethanol",
            "smiles": "CCO",
            "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            "formula": "C2H6O"
            },
            "molarweight": 46.07,
            "m": 3.6856059608152645,
            "sigma": 2.718370938956805,
            "epsilon_k": 175.508463577238,
            "association_sites": [
            {
                "na": 1.0,
                "nb": 1.0,
                "kappa_ab": 0.11116132011425092,
                "epsilon_k_ab": 2182.115702815104
            }
            ]
        }
        ```
        """
        identifier = self.compound.identifier
        mu = self.spec.mu if "mu" not in self.params else self.params["mu"]
        q = self.spec.q
        na = self.spec.na
        nb = self.spec.nb

        path = Path(path)
        entry = {
            "identifier": {
                "cas": identifier.cas,
                "name": self.input_name or identifier.name,
                "iupac_name": identifier.iupac_name,
                "smiles": identifier.smiles,
                "inchi": identifier.inchi,
                "formula": identifier.formula,
            },
            "molarweight": self.compound.mw,
            "m": self.params["m"],
            "sigma": self.params["sigma"],
            "epsilon_k": self.params["epsilon_k"],
        }

        if mu != 0.0:
            entry["mu"] = mu
        if q != 0.0:
            entry["q"] = q
        if na is not None:
            site = {"na": float(na), "nb": float(nb)}
            if "kappa_ab" in self.params:
                site["kappa_ab"] = self.params["kappa_ab"]
                site["epsilon_k_ab"] = self.params["epsilon_k_ab"]
            entry["association_sites"] = [site]

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # Remove existing entries with same CAS or name
            name_to_match = self.input_name or identifier.name
            data = [
                d
                for d in data
                if d.get("identifier", {}).get("cas") != identifier.cas
                and d.get("identifier", {}).get("name") != name_to_match
            ]
        else:
            data = []

        data.append(entry)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_csv(
        self,
        path: "Path | str",
        T_min: float | None = None,
        T_max: float | None = None,
        n_points: int = 501,
    ) -> None:
        """Export experimental data and PC-SAFT model curves to CSV files.

        Writes two files into *path* (created if it doesn't exist):

        * ``{name}_exp.csv``   — experimental data used for fitting
        * ``{name}_model.csv`` — phase envelope from PC-SAFT

        Parameters
        ----------
        path : Path or str
            Directory to write the CSV files into.
        T_min : float, optional
            Lower bound for the model curve in the same temperature units as
            the experimental data. Default: T_exp_min - 5.
        T_max : float, optional
            Upper bound for the model curve (rows above this are dropped).
            Default: T_exp_max + 5.
        n_points : int, optional
            Number of points along the phase envelope. Default: 501.
        """
        import polars as pl
        import feos

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        name = self.input_name or self.compound.identifier.name

        tu = self.units.temperature
        pu = self.units.pressure
        du = self.units.density

        # --- experimental CSV ---
        n_psat = len(self.data.T_psat)
        n_rho  = len(self.data.T_rho)
        n_hvap = len(self.data.T_hvap)
        n_exp  = max(n_psat, n_rho, n_hvap)

        def _pad(arr, n):
            if len(arr) == n:
                return arr.tolist()
            return arr.tolist() + [None] * (n - len(arr))

        exp_data = {
            "T_psat": _pad(self.data.T_psat, n_exp),
            "p_psat": _pad(self.data.p_psat, n_exp),
            "T_rho":  _pad(self.data.T_rho,  n_exp),
            "rho":    _pad(self.data.rho,     n_exp),
        }
        if n_hvap > 0:
            exp_data["T_hvap"] = _pad(self.data.T_hvap, n_exp)
            exp_data["hvap"]   = _pad(self.data.hvap,   n_exp)

        pl.DataFrame(exp_data).write_csv(path / f"{name}_exp.csv")

        # --- model curve CSV ---
        all_T = [self.data.T_psat, self.data.T_rho, self.data.T_hvap]
        T_exp_min = float(min(T.min() for T in all_T if len(T) > 0))
        T_exp_max = float(max(T.max() for T in all_T if len(T) > 0))
        if T_min is None:
            T_min = T_exp_min - 5.0
        if T_max is None:
            T_max = T_exp_max + 5.0

        phase_diagram = feos.PhaseDiagram.pure(self.eos, T_min * tu, n_points)

        (
            pl.DataFrame({
                "T":       (phase_diagram.vapor.temperature / tu).tolist(),
                "p_sat":   (phase_diagram.vapor.pressure    / pu).tolist(),
                "rho_liq": (phase_diagram.liquid.mass_density / du).tolist(),
                "rho_vap": (phase_diagram.vapor.mass_density  / du).tolist(),
            })
            .filter(pl.col("T") <= T_max)
            .write_csv(path / f"{name}_model.csv")
        )

    def plot(
        self,
        path=None,
        color: str = "red",
        line_color: str = "black",
        linestyle: str = "-",
        scatter_kw: dict | None = None,
        line_kw: dict | None = None,
    ):
        """Two-panel phase diagram with experimental data overlay.

        Parameters
        ----------
        path : str or Path, optional
            If given, save the figure to this path.
        color : str
            Colour for experimental data points. Preset name ("red", "blue",
            "green", "orange", "purple", "cyan", "black") or any matplotlib
            colour string. Default: "red".
        line_color : str
            Colour for the PC-SAFT curves. Same preset names or any matplotlib
            colour string. Default: "black".
        linestyle : str
            Line style for PC-SAFT curves, e.g. "-", "--", "-.", ":".
            Default: "-".
        scatter_kw : dict, optional
            Extra kwargs merged into scatter calls (overrides defaults).
        line_kw : dict, optional
            Extra kwargs merged into line plot calls (overrides defaults).

        Returns
        -------
        fig, axes
        """
        from fit_pcsaft._plot import _plot_pure

        return _plot_pure(
            self,
            path=path,
            color=color,
            line_color=line_color,
            linestyle=linestyle,
            scatter_kw=scatter_kw,
            line_kw=line_kw,
        )

    def __str__(self) -> str:
        """Pretty-print fitted parameters and quality metrics."""
        mu = self.params.get("mu", self.spec.mu or 0.0)
        q = self.spec.q
        na = self.spec.na
        nb = self.spec.nb
        n_psat = len(self.data.T_psat)
        n_rho = len(self.data.T_rho)
        n_hvap = len(self.data.T_hvap)

        lines = [
            "Fitted parameters:",
            f"  m (segments):            {self.params['m']:.4f}",
            f"  σ (diameter):            {self.params['sigma']:.4f} Å",
            f"  ε/k (energy):            {self.params['epsilon_k']:.2f} K",
        ]

        if mu != 0.0:
            lines.append(f"  μ (dipole):              {mu:.4f} D")
        if q != 0.0:
            lines.append(f"  q (quadrupole):          {q:.4f} DÅ")
        if "kappa_ab" in self.params:
            lines.append(f"  κ_ab (assoc. volume):    {self.params['kappa_ab']:.6f}")
        if "epsilon_k_ab" in self.params:
            lines.append(
                f"  ε_ab/k (assoc. energy):  {self.params['epsilon_k_ab']:.2f} K"
            )

        if na is not None:
            scheme = _assoc_scheme_name(na, nb)
            lines.append(f"\nAssociation scheme:        {scheme} (na={na}, nb={nb})")

        rms = np.sqrt(2.0 * self.scipy_result.cost / len(self.scipy_result.fun))
        ard_total = sum(
            v for v in [self.ard_psat, self.ard_rho, self.ard_hvap] if not np.isnan(v)
        )
        quality_lines = [
            "",
            "Fitting quality:",
            f"  ARD vapor pressure:      {self.ard_psat:.2f}%  (n={n_psat})",
            f"  ARD liquid density:      {self.ard_rho:.2f}%  (n={n_rho})",
        ]
        if n_hvap > 0:
            quality_lines.append(
                f"  ARD enthalpy of vap.:    {self.ard_hvap:.2f}%  (n={n_hvap})"
            )
        quality_lines.append(f"  ARD total:               {ard_total:.2f}%")
        quality_lines.extend(
            [
                f"  RMS weighted resid.:     {rms:.4f}",
                f"  Converged:               {self.scipy_result.success}",
                f"  Function evals:          {self.scipy_result.nfev}",
                f"  Time elapsed:            {self.time_elapsed:.2f} s",
            ]
        )
        lines.extend(quality_lines)

        return "\n".join(lines)


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating PC-SAFT parameters against experimental data.

    Attributes
    ----------
        params : dict
            Parameters that were evaluated {"m", "sigma", "epsilon_k", ...}
        eos : feos.EquationOfState
            Constructed EOS object
        data : PureData
            Experimental data used for evaluation
        compound : Compound
            Compound identifier and molar weight
        spec : ModelSpec
            Model specification (mu, q, na, nb)
        units : Units
            Units for experimental data
        ard_psat : float
            Average relative deviation for vapor pressure (%)
        ard_rho : float
            Average relative deviation for liquid density (%)
        ard_hvap : float
            Average relative deviation for enthalpy of vaporization (%). nan if no data.
        input_name : str
            Compound name as supplied by the user
    """

    params: dict
    eos: feos.EquationOfState
    data: PureData
    compound: Compound
    spec: ModelSpec
    units: Units
    ard_psat: float
    ard_rho: float
    ard_hvap: float
    input_name: str = ""

    def to_csv(
        self,
        path: "Path | str",
        T_min: float | None = None,
        T_max: float | None = None,
        n_points: int = 501,
    ) -> None:
        """Export experimental data and PC-SAFT model curves to CSV files.

        Writes two files into *path* (created if it doesn't exist):

        * ``{name}_exp.csv``   — experimental data used for evaluation
        * ``{name}_model.csv`` — phase envelope from PC-SAFT

        Parameters
        ----------
        path : Path or str
            Directory to write the CSV files into.
        T_min : float, optional
            Lower bound for the model curve in the same temperature units as
            the experimental data. Default: T_exp_min - 5.
        T_max : float, optional
            Upper bound for the model curve (rows above this are dropped).
            Default: T_exp_max + 5.
        n_points : int, optional
            Number of points along the phase envelope. Default: 501.
        """
        import polars as pl
        import feos

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        name = self.input_name or self.compound.identifier.name

        tu = self.units.temperature
        pu = self.units.pressure
        du = self.units.density

        n_psat = len(self.data.T_psat)
        n_rho  = len(self.data.T_rho)
        n_hvap = len(self.data.T_hvap)
        n_exp  = max(n_psat, n_rho, n_hvap)

        def _pad(arr, n):
            if len(arr) == n:
                return arr.tolist()
            return arr.tolist() + [None] * (n - len(arr))

        exp_data = {
            "T_psat": _pad(self.data.T_psat, n_exp),
            "p_psat": _pad(self.data.p_psat, n_exp),
            "T_rho":  _pad(self.data.T_rho,  n_exp),
            "rho":    _pad(self.data.rho,     n_exp),
        }
        if n_hvap > 0:
            exp_data["T_hvap"] = _pad(self.data.T_hvap, n_exp)
            exp_data["hvap"]   = _pad(self.data.hvap,   n_exp)

        pl.DataFrame(exp_data).write_csv(path / f"{name}_exp.csv")

        all_T = [self.data.T_psat, self.data.T_rho, self.data.T_hvap]
        T_exp_min = float(min(T.min() for T in all_T if len(T) > 0))
        T_exp_max = float(max(T.max() for T in all_T if len(T) > 0))
        if T_min is None:
            T_min = T_exp_min - 5.0
        if T_max is None:
            T_max = T_exp_max + 5.0

        phase_diagram = feos.PhaseDiagram.pure(self.eos, T_min * tu, n_points)

        (
            pl.DataFrame({
                "T":       (phase_diagram.vapor.temperature / tu).tolist(),
                "p_sat":   (phase_diagram.vapor.pressure    / pu).tolist(),
                "rho_liq": (phase_diagram.liquid.mass_density / du).tolist(),
                "rho_vap": (phase_diagram.vapor.mass_density  / du).tolist(),
            })
            .filter(pl.col("T") <= T_max)
            .write_csv(path / f"{name}_model.csv")
        )

    def plot(
        self,
        path=None,
        color: str = "red",
        line_color: str = "black",
        linestyle: str = "-",
        scatter_kw: Optional[dict] = None,
        line_kw: Optional[dict] = None,
    ):
        """Two-panel phase diagram with experimental data overlay."""
        from fit_pcsaft._plot import _plot_pure

        return _plot_pure(
            self,
            path=path,
            color=color,
            line_color=line_color,
            linestyle=linestyle,
            scatter_kw=scatter_kw,
            line_kw=line_kw,
        )

    def __str__(self) -> str:
        mu = self.params.get("mu", self.spec.mu or 0.0)
        q = self.spec.q
        na = self.spec.na
        nb = self.spec.nb
        n_psat = len(self.data.T_psat)
        n_rho = len(self.data.T_rho)
        n_hvap = len(self.data.T_hvap)

        lines = [
            "Parameters:",
            f"  m (segments):            {self.params['m']:.4f}",
            f"  σ (diameter):            {self.params['sigma']:.4f} Å",
            f"  ε/k (energy):            {self.params['epsilon_k']:.2f} K",
        ]
        if mu != 0.0:
            lines.append(f"  μ (dipole):              {mu:.4f} D")
        if q != 0.0:
            lines.append(f"  q (quadrupole):          {q:.4f} DÅ")
        if "kappa_ab" in self.params:
            lines.append(f"  κ_ab (assoc. volume):    {self.params['kappa_ab']:.6f}")
        if "epsilon_k_ab" in self.params:
            lines.append(
                f"  ε_ab/k (assoc. energy):  {self.params['epsilon_k_ab']:.2f} K"
            )
        if na is not None:
            scheme = _assoc_scheme_name(na, nb)
            lines.append(f"\nAssociation scheme:        {scheme} (na={na}, nb={nb})")

        ard_total = sum(
            v for v in [self.ard_psat, self.ard_rho, self.ard_hvap] if not np.isnan(v)
        )
        lines += [
            "",
            "ARD% vs experimental data:",
            f"  ARD vapor pressure:      {self.ard_psat:.2f}%  (n={n_psat})",
            f"  ARD liquid density:      {self.ard_rho:.2f}%  (n={n_rho})",
        ]
        if n_hvap > 0:
            lines.append(f"  ARD enthalpy of vap.:    {self.ard_hvap:.2f}%  (n={n_hvap})")
        lines.append(f"  ARD total:               {ard_total:.2f}%")

        return "\n".join(lines)


def _assoc_scheme_name(na: int, nb: int) -> str:
    """Return common association scheme name for given na and nb."""
    _schemes = {
        (1, 0): "1A",
        (0, 1): "1A",
        (1, 1): "2B",
        (1, 2): "3B",
        (2, 1): "3B",
        (2, 2): "4C",
    }
    return _schemes.get((na, nb), "custom")
