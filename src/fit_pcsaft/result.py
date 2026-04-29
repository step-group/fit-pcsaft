import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import feos
import numpy as np

from fit_pcsaft._types import Compound, ModelSpec, PureData, Units


def _predict_per_property(eos, data, units):
    """Return (model_arr, exp_arr) per property, NaN where feos raises.

    Returns dict keyed by "psat", "rho", "hvap". Each value is a tuple
    (model: np.ndarray, exp: np.ndarray) of equal length. Empty datasets
    produce zero-length arrays. Never raises.
    """
    tu = units.temperature
    pu = units.pressure
    du = units.density
    eu = units.enthalpy

    def _psat_model(T_vals, p_exp_vals):
        model = np.full(len(T_vals), np.nan)
        for i, T in enumerate(T_vals):
            try:
                model[i] = float(feos.PhaseEquilibrium.vapor_pressure(eos, float(T) * tu)[0] / pu)
            except Exception:
                pass
        return model, np.asarray(p_exp_vals, dtype=float)

    def _rho_model(T_vals, rho_exp_vals):
        model = np.full(len(T_vals), np.nan)
        for i, T in enumerate(T_vals):
            try:
                model[i] = float(
                    feos.PhaseEquilibrium.pure(eos, float(T) * tu).liquid.mass_density() / du
                )
            except Exception:
                pass
        return model, np.asarray(rho_exp_vals, dtype=float)

    def _hvap_model(T_vals, hvap_exp_vals):
        model = np.full(len(T_vals), np.nan)
        for i, T in enumerate(T_vals):
            try:
                vle = feos.PhaseEquilibrium.pure(eos, float(T) * tu)
                model[i] = float(
                    (vle.vapor.molar_enthalpy(feos.Contributions.Residual)
                     - vle.liquid.molar_enthalpy(feos.Contributions.Residual)) / eu
                )
            except Exception:
                pass
        return model, np.asarray(hvap_exp_vals, dtype=float)

    return {
        "psat": _psat_model(data.T_psat, data.p_psat),
        "rho":  _rho_model(data.T_rho, data.rho),
        "hvap": _hvap_model(data.T_hvap, data.hvap),
    }


def _compute_pure_metrics(eos, data, units):
    """Compute per-property Metrics from an EOS + experimental data.

    Returns dict keyed by "psat", "rho", "hvap". Properties with no input rows
    return Metrics.empty(0).
    """
    from fit_pcsaft._metrics import compute_metrics_from_arrays
    preds = _predict_per_property(eos, data, units)
    return {
        prop: compute_metrics_from_arrays(model, exp, n_total=len(exp))
        for prop, (model, exp) in preds.items()
    }


def _compute_per_point_rd(eos, data, units):
    """Compute per-point signed RD% and ARD% for all experimental datasets."""
    import polars as pl

    preds = _predict_per_property(eos, data, units)

    rows = []
    prop_labels = {"psat": (data.T_psat, data.p_psat),
                   "rho":  (data.T_rho,  data.rho),
                   "hvap": (data.T_hvap, data.hvap)}
    for prop, (model_arr, exp_arr) in preds.items():
        T_arr = prop_labels[prop][0]
        for T_val, exp_val, model_val in zip(T_arr, exp_arr, model_arr):
            rd = (float(model_val) - float(exp_val)) / float(exp_val) * 100.0 if np.isfinite(model_val) else float("nan")
            rows.append({
                "property": prop,
                "T": float(T_val),
                "exp": float(exp_val),
                "model": float(model_val),
                "rd_pct": rd,
                "ard_pct": abs(rd) if np.isfinite(rd) else float("nan"),
            })

    if not rows:
        return pl.DataFrame(schema={
            "property": pl.Utf8, "T": pl.Float64, "exp": pl.Float64,
            "model": pl.Float64, "rd_pct": pl.Float64, "ard_pct": pl.Float64,
        })
    return pl.DataFrame(rows)


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

        metrics : dict
            Per-property Metrics panels keyed by "psat", "rho", "hvap".
            Access via result.metrics["psat"].aard_pct, .rmsd_pct, .r2, etc.

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
    metrics: dict
    scipy_result: object
    time_elapsed: float
    input_name: str = ""

    # --- backwards-compat shims -------------------------------------------
    @property
    def ard_psat(self) -> float:
        return self.metrics["psat"].aard_pct

    @property
    def ard_rho(self) -> float:
        return self.metrics["rho"].aard_pct

    @property
    def ard_hvap(self) -> float:
        return self.metrics["hvap"].aard_pct

    @property
    def metrics_psat(self):
        return self.metrics["psat"]

    @property
    def metrics_rho(self):
        return self.metrics["rho"]

    @property
    def metrics_hvap(self):
        return self.metrics["hvap"]

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

        * ``{name}_exp.csv``   — per-point residuals (long/tidy):
          ``property, T, exp, model, rd_pct, ard_pct``
        * ``{name}_model.csv`` — smooth phase envelope:
          ``T, inv_T, p_sat, ln_psat, rho_liq, rho_vap, hvap``

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
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        name = self.input_name or self.compound.identifier.name

        # --- experimental CSV (long/tidy: one row per data point) ---
        _compute_per_point_rd(self.eos, self.data, self.units).write_csv(
            path / f"{name}_exp.csv"
        )

        # --- model curve CSV ---
        _model_curve_df(
            self.eos, self.data, self.units, T_min, T_max, n_points
        ).write_csv(path / f"{name}_model.csv")

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

    def residuals(self):
        """Per-point signed RD% and absolute ARD% as a polars DataFrame.

        Columns: ``property`` ("psat"/"rho"/"hvap"), ``T``, ``exp``, ``model``,
        ``rd_pct`` = (model − exp)/exp × 100, ``ard_pct`` = |rd_pct|.

        Export: ``result.residuals().write_csv("out.csv")``.
        """
        return _compute_per_point_rd(self.eos, self.data, self.units)

    def plot_residuals(self, path=None):
        """RD% vs temperature for each property.

        Parameters
        ----------
        path : str or Path, optional
            If given, save the figure to this path.

        Returns
        -------
        fig, ax
        """
        from fit_pcsaft._plot import _plot_residuals_pure
        return _plot_residuals_pure(self, path=path)

    def metrics_table(self):
        """Per-property metrics as a tidy polars DataFrame (one row per property)."""
        import polars as pl
        rows = []
        for prop in ("psat", "rho", "hvap"):
            m = self.metrics[prop]
            rows.append({
                "property": prop, "n": m.n, "n_total": m.n_total,
                "aard_pct": m.aard_pct, "rmsd_pct": m.rmsd_pct,
                "bias_pct": m.bias_pct, "mard_pct": m.mard_pct,
                "mae": m.mae, "rmsd": m.rmsd, "r2": m.r2,
            })
        return pl.DataFrame(rows)

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
        quality_lines = ["", "Fitting quality:"]
        for prop, label in [("psat", "Vapor pressure   "),
                            ("rho",  "Liquid density   "),
                            ("hvap", "Hvap             ")]:
            m = self.metrics[prop]
            if m.n_total > 0:
                quality_lines.append(f"  {label} {m}")
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
        metrics : dict
            Per-property Metrics panels keyed by "psat", "rho", "hvap".
        input_name : str
            Compound name as supplied by the user
    """

    params: dict
    eos: feos.EquationOfState
    data: PureData
    compound: Compound
    spec: ModelSpec
    units: Units
    metrics: dict
    input_name: str = ""

    # --- backwards-compat shims -------------------------------------------
    @property
    def ard_psat(self) -> float:
        return self.metrics["psat"].aard_pct

    @property
    def ard_rho(self) -> float:
        return self.metrics["rho"].aard_pct

    @property
    def ard_hvap(self) -> float:
        return self.metrics["hvap"].aard_pct

    @property
    def metrics_psat(self):
        return self.metrics["psat"]

    @property
    def metrics_rho(self):
        return self.metrics["rho"]

    @property
    def metrics_hvap(self):
        return self.metrics["hvap"]

    def to_csv(
        self,
        path: "Path | str",
        T_min: float | None = None,
        T_max: float | None = None,
        n_points: int = 501,
    ) -> None:
        """Export experimental data and PC-SAFT model curves to CSV files.

        Writes two files into *path* (created if it doesn't exist):

        * ``{name}_exp.csv``   — per-point residuals (long/tidy):
          ``property, T, exp, model, rd_pct, ard_pct``
        * ``{name}_model.csv`` — smooth phase envelope:
          ``T, inv_T, p_sat, ln_psat, rho_liq, rho_vap, hvap``

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
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        name = self.input_name or self.compound.identifier.name

        # --- experimental CSV (long/tidy: one row per data point) ---
        _compute_per_point_rd(self.eos, self.data, self.units).write_csv(
            path / f"{name}_exp.csv"
        )

        _model_curve_df(
            self.eos, self.data, self.units, T_min, T_max, n_points
        ).write_csv(path / f"{name}_model.csv")

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

    def residuals(self):
        """Per-point signed RD% and absolute ARD% as a polars DataFrame.

        Columns: ``property`` ("psat"/"rho"/"hvap"), ``T``, ``exp``, ``model``,
        ``rd_pct`` = (model − exp)/exp × 100, ``ard_pct`` = |rd_pct|.

        Export: ``result.residuals().write_csv("out.csv")``.
        """
        return _compute_per_point_rd(self.eos, self.data, self.units)

    def plot_residuals(self, path=None):
        """RD% vs temperature for each property.

        Parameters
        ----------
        path : str or Path, optional
            If given, save the figure to this path.

        Returns
        -------
        fig, ax
        """
        from fit_pcsaft._plot import _plot_residuals_pure
        return _plot_residuals_pure(self, path=path)

    def metrics_table(self):
        """Per-property metrics as a tidy polars DataFrame (one row per property)."""
        import polars as pl
        rows = []
        for prop in ("psat", "rho", "hvap"):
            m = self.metrics[prop]
            rows.append({
                "property": prop, "n": m.n, "n_total": m.n_total,
                "aard_pct": m.aard_pct, "rmsd_pct": m.rmsd_pct,
                "bias_pct": m.bias_pct, "mard_pct": m.mard_pct,
                "mae": m.mae, "rmsd": m.rmsd, "r2": m.r2,
            })
        return pl.DataFrame(rows)

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

        lines += ["", "Fitting quality:"]
        for prop, label in [("psat", "Vapor pressure   "),
                            ("rho",  "Liquid density   "),
                            ("hvap", "Hvap             ")]:
            m = self.metrics[prop]
            if m.n_total > 0:
                lines.append(f"  {label} {m}")

        return "\n".join(lines)


def _model_curve_df(eos, data, units, T_min, T_max, n_points):
    """Build the model-curve polars DataFrame for to_csv.

    T ranges are computed per-dataset (psat / rho / hvap independently),
    so a wide rho dataset does not inflate the psat column range and vice
    versa.  Each column is nulled outside its own dataset's [T_min, T_max].
    The overall phase diagram spans the union of all per-dataset ranges.
    """
    import polars as pl
    import feos

    tu = units.temperature
    pu = units.pressure
    du = units.density
    eu = units.enthalpy

    def _bounds(arr, override_min, override_max):
        """Return (lo, hi) in data units, or (None, None) if dataset empty."""
        if len(arr) == 0:
            return None, None
        lo = override_min if override_min is not None else float(arr.min()) - 5.0
        hi = override_max if override_max is not None else float(arr.max()) + 5.0
        return lo, hi

    psat_lo, psat_hi = _bounds(data.T_psat, T_min, T_max)
    rho_lo,  rho_hi  = _bounds(data.T_rho,  T_min, T_max)
    hvap_lo, hvap_hi = _bounds(data.T_hvap, T_min, T_max)

    all_los = [v for v in [psat_lo, rho_lo, hvap_lo] if v is not None]
    all_his = [v for v in [psat_hi, rho_hi, hvap_hi] if v is not None]
    overall_min = min(all_los)
    overall_max = max(all_his)

    phase_diagram = feos.PhaseDiagram.pure(eos, overall_min * tu, n_points)
    T_col = (phase_diagram.vapor.temperature / tu).tolist()

    def _mask(col_vals, lo, hi):
        if lo is None:
            return col_vals
        return [v if lo <= t <= hi else None for t, v in zip(T_col, col_vals)]

    import math
    p_sat_vals   = (phase_diagram.vapor.pressure      / pu).tolist()
    rho_liq_vals = (phase_diagram.liquid.mass_density / du).tolist()
    rho_vap_vals = (phase_diagram.vapor.mass_density  / du).tolist()
    try:
        hvap_arr = (
            phase_diagram.vapor.molar_enthalpy(feos.Contributions.Residual)
            - phase_diagram.liquid.molar_enthalpy(feos.Contributions.Residual)
        ) / eu
        hvap_vals = hvap_arr.tolist()
    except Exception:
        hvap_vals = [None] * len(T_col)

    masked_ln_psat = _mask(
        [math.log(v) if v is not None and v > 0 else None for v in p_sat_vals],
        psat_lo, psat_hi,
    )
    masked_inv_T = _mask(
        [1000.0 / t if t else None for t in T_col],
        psat_lo, psat_hi,
    )

    return (
        pl.DataFrame({
            "T":        T_col,
            "inv_T":    masked_inv_T,
            "p_sat":    _mask(p_sat_vals, psat_lo, psat_hi),
            "ln_psat":  masked_ln_psat,
            "rho_liq":  _mask(rho_liq_vals, rho_lo, rho_hi),
            "rho_vap":  _mask(rho_vap_vals, rho_lo, rho_hi),
            "hvap":     _mask(hvap_vals, hvap_lo, hvap_hi),
        })
        .filter(pl.col("T") <= overall_max)
    )


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
