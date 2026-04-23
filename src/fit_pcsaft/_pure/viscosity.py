"""Entropy scaling viscosity correlation fitting for pure PC-SAFT components.

Correlation model (Lötgering-Lin & Gross 2018, Ind. Eng. Chem. Res. 57, 4095):

    ln(η / η_CE) = A + B·s + C·s² + D·s³,   s = s_res / (R·m)

where η_CE is the Chapman-Enskog reference viscosity, s_res the residual molar
entropy, R the gas constant, and m the number of SAFT segments.

Fitting procedure (faithful to LL 2018):

- **D** is always fixed from the molar-mass correlation (LL 2018, eq. 14):
      D = 1 / (−1.25594 − 888.1232 / M[g/mol])
- **A** is fixed from group contribution when ``groups`` is supplied (LL 2015,
  eq. 11 + page 4097 of LL 2018):
      A_gc = Σ_α (n_α · m_α · σ_α³ · A_α)
      A_i  = A_gc + 0.5 · ln(1 / m_i)
  The ln shift bridges the 2015 vs 2018 Chapman-Enskog reference conventions.
  Alternatively pass ``a_gc=<float>`` to supply A_gc directly.
- **Only B and C** are fit from data (or A, B, C when no GC source is given).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import feos
import numpy as np
import polars as pl
import scipy.optimize
import si_units as si

from fit_pcsaft._csv import SCHEMA_VISCOSITY, load_csv
from fit_pcsaft._binary._utils import _apply_induced_association
from fit_pcsaft._pure.viscosity_gc import compute_a_gc


@dataclass(frozen=True)
class ViscosityFitResult:
    """Result of entropy scaling viscosity parameter fitting.

    Attributes
    ----------
    viscosity_params : list
        Fitted ``[A, B, C, D]`` correlation parameters.
    eos : feos.EquationOfState or None
        Rebuilt EOS with viscosity parameters set; ``None`` if rebuild failed.
    ard : float
        Mean absolute relative deviation of viscosity vs experimental data (%).
    n_points : int
        Number of experimental data points used.
    input_name : str
        Compound name used by :meth:`to_json` to locate the JSON entry.
    """

    viscosity_params: list
    eos: Optional[object]
    ard: float
    n_points: int
    input_name: str = ""
    a_fixed: bool = False
    d_fixed: bool = True
    groups: Optional[dict] = None
    a_gc_used: Optional[float] = None
    # Internal data stored for plotting / export (not shown in __str__)
    _T_K: list = field(default_factory=list, repr=False, compare=False)
    _P_MPa: list = field(default_factory=list, repr=False, compare=False)
    _eta_exp_Pa_s: list = field(default_factory=list, repr=False, compare=False)
    _s_vals: list = field(default_factory=list, repr=False, compare=False)
    _y_vals: list = field(default_factory=list, repr=False, compare=False)

    def to_json(self, path: "Path | str", name: str = "") -> None:
        """Add or update the ``viscosity`` field in a feos-compatible JSON file.

        The entry is found by matching ``name`` (or ``self.input_name``) against
        the ``identifier.name`` or ``identifier.cas`` of each record.

        Parameters
        ----------
        path : Path | str
            Existing feos JSON parameter file that already contains the compound.
        name : str, optional
            Compound name or CAS override.  Defaults to ``self.input_name``.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If no matching entry is found.
        ValueError
            If no name is known (neither *name* nor ``self.input_name`` set).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        search = name or self.input_name
        if not search:
            raise ValueError(
                "Provide a compound name via the 'name' argument "
                "or pass name=... to fit_viscosity_entropy_scaling()."
            )

        records = json.loads(path.read_text(encoding="utf-8"))
        matched = False
        for entry in records:
            ident = entry.get("identifier", {})
            if ident.get("name") == search or ident.get("cas") == search:
                entry["viscosity"] = self.viscosity_params
                matched = True
                break

        if not matched:
            found = [e.get("identifier", {}).get("name", "?") for e in records]
            raise KeyError(
                f"No entry matching '{search}' in {path}.\n"
                f"Entries found: {found}"
            )

        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def to_csv(self, path: "Path | str") -> None:
        """Write experimental and predicted viscosity to a CSV file.

        Columns: ``T`` (K), ``P`` (MPa), ``eta_exp`` (Pa·s),
        ``eta_pred`` (Pa·s), ``ard_pct`` (%).

        ``eta_pred`` is computed from the rebuilt EOS when available;
        otherwise from the entropy scaling polynomial directly.

        Parameters
        ----------
        path : Path | str
            Output CSV path.
        """
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        A, B, C, D = self.viscosity_params
        rows = []
        for T, P, eta_e in zip(self._T_K, self._P_MPa, self._eta_exp_Pa_s):
            eta_p = float("nan")
            if self.eos is not None:
                try:
                    state = feos.State(
                        self.eos,
                        temperature=T * si.KELVIN,
                        pressure=P * si.MEGA * si.PASCAL,
                        total_moles=si.MOL,
                        density_initialization="liquid",
                    )
                    eta_p = float(state.viscosity() / (si.PASCAL * si.SECOND))
                except Exception:
                    pass
            ard_pct = abs(eta_p - eta_e) / eta_e * 100 if np.isfinite(eta_p) else float("nan")
            rows.append({"T": T, "P": P, "eta_exp": eta_e, "eta_pred": eta_p, "ard_pct": ard_pct})

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["T", "P", "eta_exp", "eta_pred", "ard_pct"])
            w.writeheader()
            w.writerows(rows)

    def __str__(self) -> str:
        A, B, C, D = self.viscosity_params
        if self.a_fixed:
            if self.groups is not None:
                a_src = f"(fixed from GC: groups={self.groups})"
            else:
                a_src = "(fixed from a_gc)"
        else:
            a_src = "(fit)"
        d_src = "(fixed from molar-mass correlation)" if self.d_fixed else "(fit)"
        lines = [
            "Entropy scaling viscosity parameters [A, B, C, D]:",
            f"  A: {A:+.6f}  {a_src}",
            f"  B: {B:+.6f}  (fit)",
            f"  C: {C:+.6f}  (fit)",
            f"  D: {D:+.6f}  {d_src}",
            "",
            "Fitting quality:",
            f"  ARD viscosity: {self.ard:.2f}%  (n={self.n_points})",
        ]
        if self.eos is None:
            lines.append("\n  Note: EOS rebuild failed — viscosity_params are still valid.")
        return "\n".join(lines)

    def plot(self, path=None):
        """Two-panel plot: η vs T (left) and entropy scaling fit (right).

        Parameters
        ----------
        path : str or Path, optional
            If given, save the figure to this path (dpi=300).

        Returns
        -------
        fig, axes
        """
        return _plot_viscosity_pure(self, path=path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_viscosity_csv(path: "Path | str"):
    """Return (T, P, eta, phases) from a viscosity CSV. P is None when the column is absent."""
    data = load_csv(path, SCHEMA_VISCOSITY)
    T = data["T"]
    P = data.get("P")  # None if column absent — caller uses P_sat instead
    eta = data["eta"]

    # Phase is a string column — read the raw DF to extract it separately.
    df = pl.read_csv(Path(path), infer_schema_length=9999, truncate_ragged_lines=True)
    phase_col = next(
        (c for c in df.columns if c.strip().lower() == "phase"), None
    )
    if phase_col is not None:
        phases = [str(v) if v is not None else None for v in df[phase_col].to_list()]
    else:
        phases = [None] * len(T)

    return T, P, eta, phases



def _D_from_molar_mass(mw_g_mol: float) -> float:
    """LL & Gross 2018, eq. 14: D = 1 / (−1.25594 − 888.1232 / M[g/mol])."""
    return 1.0 / (-1.25594 - 888.1232 / mw_g_mol)


def _rebuild_eos_with_viscosity(source, viscosity_list: list) -> Optional[object]:
    """Rebuild feos.EquationOfState with viscosity params added.

    *source* is a FitResult or a feos.Parameters object.  Returns None on failure.
    """
    try:
        if hasattr(source, 'pure_records'):
            # feos.Parameters
            rec = source.pure_records[0]
            mr = rec.model_record
            kw = {
                'identifier': rec.identifier,
                'molarweight': rec.molarweight,
                'm': mr['m'],
                'sigma': mr['sigma'],
                'epsilon_k': mr['epsilon_k'],
                'viscosity': viscosity_list,
            }
            for opt in ('mu', 'q'):
                if opt in mr:
                    kw[opt] = mr[opt]
            if rec.association_sites:
                kw['association_sites'] = list(rec.association_sites)
        else:
            # FitResult
            fr = source
            p = fr.params
            spec = fr.spec
            mu_val = p.get('mu', spec.mu)
            if mu_val is None:
                mu_val = 0.0
            na, nb = spec.na, spec.nb
            assoc = na is not None and nb is not None and na > 0 and nb > 0
            kw = {
                'identifier': fr.compound.identifier,
                'molarweight': fr.compound.mw,
                'm': p['m'],
                'sigma': p['sigma'],
                'epsilon_k': p['epsilon_k'],
                'mu': float(mu_val),
                'q': float(spec.q),
                'viscosity': viscosity_list,
            }
            if assoc:
                kw['association_sites'] = [{
                    'na': na, 'nb': nb,
                    'kappa_ab': p['kappa_ab'],
                    'epsilon_k_ab': p['epsilon_k_ab'],
                }]

        new_rec = feos.PureRecord(**kw)
        return feos.EquationOfState.pcsaft(feos.Parameters.new_pure(new_rec))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_viscosity_entropy_scaling(
    source: "FitResult | feos.Parameters",
    viscosity_path: "Path | str",
    name: str = "",
    temperature_unit: "si.SIObject" = si.KELVIN,
    pressure_unit: "si.SIObject" = si.MEGA * si.PASCAL,
    viscosity_unit: "si.SIObject" = si.PASCAL * si.SECOND,
    groups: "dict[str, int] | None" = None,
    a_gc: "float | None" = None,
    fix_d: bool = True,
    loss: str = "linear",
    f_scale: float = 0.1,
) -> ViscosityFitResult:
    """Fit entropy scaling viscosity correlation ``[A, B, C, D]`` to experimental data.

    Implements the Lötgering-Lin & Gross (2018) procedure faithfully:

    - **D** is always computed from the molar-mass correlation (eq. 14).
    - **A** is fixed from group contribution when ``groups`` or ``a_gc`` is
      given; otherwise A is also fit from data.
    - Only **B and C** (or A, B, C without GC) are regressed by OLS.

    Parameters
    ----------
    source : FitResult or feos.Parameters
        PC-SAFT parameters — either a :class:`~fit_pcsaft.FitResult` returned
        by :func:`~fit_pcsaft.fit_pure`, or a ``feos.Parameters`` object (e.g.
        loaded via ``feos.Parameters.from_json``).
    viscosity_path : Path | str
        CSV with columns ``T``, ``P``, ``eta`` (temperature, pressure, viscosity).
        An optional ``phase`` string column (``'liquid'`` / ``'vapor'``) guides
        the EOS to the correct density root near two-phase conditions.
        Default units: K, MPa, Pa·s — override with the ``*_unit`` parameters.
    name : str
        Compound name stored in the result and used by
        :meth:`ViscosityFitResult.to_json`. Defaults to the name already in
        *source*.
    temperature_unit : si.SIObject
        Unit for temperature column. Default: K.
    pressure_unit : si.SIObject
        Unit for pressure column. Default: MPa.
    viscosity_unit : si.SIObject
        Unit for viscosity column. Default: Pa·s.
    groups : dict[str, int], optional
        Segment name → count for group-contribution A calculation, e.g.
        ``{"CH3": 2, "CH2": 4}`` for hexane.  Uses the Sauer 2014 + LL 2015
        parameter table shipped with fit-pcsaft.
    a_gc : float, optional
        Pre-computed A_gc value (before the 0.5·ln(1/m) shift).  Use when
        group parameters come from an external source.  Ignored if ``groups``
        is given.
    fix_d : bool
        If ``True`` (default), fix D from the molar-mass correlation
        (LL 2018, eq. 14).  If ``False``, D is also regressed from data —
        useful when the molar-mass correlation is a poor fit or for
        associating compounds where the cubic term behaves differently.
    loss : str
        Loss function: ``'linear'`` (default, fast lstsq) or any robust loss
        accepted by ``scipy.optimize.least_squares`` (``'huber'``, ``'cauchy'``,
        ``'arctan'``, ``'soft_l1'``).
    f_scale : float
        Residual scale for robust loss.  Ignored when ``loss='linear'``.

    Returns
    -------
    ViscosityFitResult
        Fitted ``viscosity_params``, rebuilt EOS, ARD%, and point count.

    Raises
    ------
    RuntimeError
        If fewer than 3 data points could be evaluated successfully.

    Notes
    -----
    Individual data points at which the EOS fails to converge are silently
    skipped and do not contribute to the fit.
    """
    # --- Resolve EOS, m, and mw from source ----------------------------------
    if hasattr(source, 'pure_records'):
        rec = source.pure_records[0]
        m = float(rec.model_record['m'])
        mw = float(rec.molarweight)
        eos = feos.EquationOfState.pcsaft(source)
        compound_name = name or rec.identifier.name
    else:
        eos = source.eos
        m = source.params['m']
        mw = float(source.compound.mw)
        compound_name = name or source.input_name

    # --- Fix D from molar-mass correlation; resolve A from GC if available --
    D_fixed = _D_from_molar_mass(mw)
    a_gc_val: "float | None" = None
    if a_gc is not None:
        a_gc_val = float(a_gc)
    elif groups is not None:
        a_gc_val = compute_a_gc(groups)
    A_fixed: "float | None" = (a_gc_val + 0.5 * np.log(1.0 / m)) if a_gc_val is not None else None

    # --- Load experimental data -------------------------------------------
    T_data, P_data, eta_data, phase_data = _load_viscosity_csv(viscosity_path)

    # --- Compute (s, eta_CE, eta_exp) triples from EOS -----------------------
    s_vals: list[float] = []
    eta_CE_vals: list[float] = []
    eta_exp_Pa_s_vals: list[float] = []
    no_pressure_col = P_data is None

    for i, (T, eta_exp, phase) in enumerate(zip(T_data, eta_data, phase_data)):
        try:
            kw: dict = {'temperature': T * temperature_unit, 'total_moles': si.MOL}
            is_liquid = isinstance(phase, str) and phase.lower() == 'liquid'

            if no_pressure_col:
                if is_liquid:
                    P_sat_list = feos.PhaseEquilibrium.vapor_pressure(eos, T * temperature_unit)
                    if not P_sat_list:
                        continue
                    kw['pressure'] = P_sat_list[0]
                    kw['density_initialization'] = 'liquid'
                else:
                    kw['pressure'] = 0.1 * si.MEGA * si.PASCAL
                    if isinstance(phase, str) and phase.lower() in ('vapor', 'vapour'):
                        kw['density_initialization'] = 'vapor'
            else:
                kw['pressure'] = float(P_data[i]) * pressure_unit
                if is_liquid or isinstance(phase, str) and phase.lower() in ('vapor', 'vapour'):
                    kw['density_initialization'] = phase

            state = feos.State(eos, **kw)

            # If a pressure was given and a liquid was requested but feos
            # settled on a vapor root, retry at P_sat.
            if not no_pressure_col and is_liquid:
                V_mol = float(state.volume / si.METER**3)
                if V_mol > 1e-3:
                    P_sat_list = feos.PhaseEquilibrium.vapor_pressure(eos, T * temperature_unit)
                    if P_sat_list:
                        kw['pressure'] = P_sat_list[0]
                        state = feos.State(eos, **kw)

            s = float(state.molar_entropy(feos.Contributions.Residual) / si.RGAS / m)
            eta_CE = float(state.viscosity_reference() / (si.PASCAL * si.SECOND))
            eta_exp_Pa_s = float(eta_exp) * float(viscosity_unit / (si.PASCAL * si.SECOND))

            if np.isfinite(s) and eta_CE > 0 and eta_exp_Pa_s > 0:
                s_vals.append(s)
                eta_CE_vals.append(eta_CE)
                eta_exp_Pa_s_vals.append(eta_exp_Pa_s)
        except Exception:
            continue

    n = len(s_vals)
    n_free_params = (2 if fix_d else 3) + (0 if A_fixed is not None else 1)
    if n < n_free_params:
        raise RuntimeError(
            f"Only {n} valid data points (need ≥ {n_free_params}). "
            "Check that T/P conditions are within the EOS validity range."
        )

    s_arr = np.array(s_vals)
    eta_CE_arr = np.array(eta_CE_vals)
    eta_exp_arr = np.array(eta_exp_Pa_s_vals)
    y_arr = np.log(eta_exp_arr / eta_CE_arr)

    # --- Shifted-y linear solve (LL 2018 faithful) ---------------------------
    # Four cases: {A fixed | A free} × {D fixed | D free}.
    #
    # A-fixed: raw monomial basis — no intercept column so κ ≤ 80 (safe).
    #   D-fixed: fit [B, C]    via [s, s²]
    #   D-free:  fit [B, C, D] via [s, s², s³]
    #
    # A-free: Chebyshev basis over t ∈ [−1,1] — intercept in [1,s,s²] drives
    #   κ > 7000 on narrow-span datasets; Chebyshev keeps κ ≤ 8 universally.
    #   D-fixed: fit [A,B,C]   via T0–T2 (degree 2)
    #   D-free:  fit [A,B,C,D] via T0–T3 (degree 3)
    #
    # Chebyshev back-transform (t=a·s+b, a=2/(s_hi−s_lo), b=−(s_hi+s_lo)/(s_hi−s_lo)):
    #   A = c0+c1b+c2(2b²−1)+c3(4b³−3b)
    #   B = a(c1+4c2b+3c3(4b²−1))
    #   C = 2a²(c2+6c3b)
    #   D = 4a³c3         (c3=0 and dropped when D is fixed)
    if fix_d:
        y_shift = y_arr - D_fixed * s_arr**3
    else:
        y_shift = y_arr.copy()

    if A_fixed is not None:
        y_shift = y_shift - A_fixed
        if fix_d:
            Phi_fit = np.column_stack([s_arr, s_arr**2])
            x0 = np.array([-2.6, -0.5])
        else:
            Phi_fit = np.column_stack([s_arr, s_arr**2, s_arr**3])
            x0 = np.array([-2.6, -0.5, -0.1])
    else:
        s_lo, s_hi = float(s_arr.min()), float(s_arr.max())
        _a = 2.0 / (s_hi - s_lo)
        _b = -(s_hi + s_lo) / (s_hi - s_lo)
        t_cheb = _a * s_arr + _b
        cheb_deg = 2 if fix_d else 3
        Phi_fit = np.polynomial.chebyshev.chebvander(t_cheb, cheb_deg)
        x0 = np.array([-1.2, -2.6, -0.5]) if fix_d else np.array([-1.2, -2.6, -0.5, -0.1])

    if loss == "linear":
        theta, *_ = np.linalg.lstsq(Phi_fit, y_shift, rcond=None)
    else:
        def _residuals(p: np.ndarray) -> np.ndarray:
            return Phi_fit @ p - y_shift

        theta = scipy.optimize.least_squares(
            _residuals, x0=x0, loss=loss, f_scale=f_scale, method="trf",
        ).x

    if A_fixed is not None:
        A = A_fixed
        if fix_d:
            B, C = theta.tolist()
            D = D_fixed
        else:
            B, C, D = theta.tolist()
    else:
        if fix_d:
            c0, c1, c2 = theta.tolist()
            c3 = 0.0
        else:
            c0, c1, c2, c3 = theta.tolist()
        A = c0 + c1*_b + c2*(2.0*_b**2 - 1.0) + c3*(4.0*_b**3 - 3.0*_b)
        B = _a * (c1 + 4.0*c2*_b + 3.0*c3*(4.0*_b**2 - 1.0))
        C = 2.0*_a**2 * (c2 + 6.0*c3*_b)
        D = 4.0*_a**3 * c3 if not fix_d else D_fixed

    viscosity_list = [float(A), float(B), float(C), float(D)]
    eos_final = _rebuild_eos_with_viscosity(source, viscosity_list)

    # Compute ARD from the rebuilt EOS so the reported number reflects actual
    # prediction quality (catches any rebuild mismatch).  Fall back to the
    # polynomial residual only when EOS rebuild failed.
    if eos_final is not None:
        ard_vals: list[float] = []
        for i, (T, eta_exp, phase) in enumerate(zip(T_data, eta_data, phase_data)):
            try:
                kw2: dict = {'temperature': T * temperature_unit, 'total_moles': si.MOL}
                is_liq2 = isinstance(phase, str) and phase.lower() == 'liquid'
                if no_pressure_col:
                    if is_liq2:
                        P_sat_list = feos.PhaseEquilibrium.vapor_pressure(eos_final, T * temperature_unit)
                        if not P_sat_list:
                            continue
                        kw2['pressure'] = P_sat_list[0]
                        kw2['density_initialization'] = 'liquid'
                    else:
                        kw2['pressure'] = 0.1 * si.MEGA * si.PASCAL
                        if isinstance(phase, str) and phase.lower() in ('vapor', 'vapour'):
                            kw2['density_initialization'] = 'vapor'
                else:
                    kw2['pressure'] = float(P_data[i]) * pressure_unit
                    if isinstance(phase, str) and phase.lower() in ('liquid', 'vapor', 'vapour'):
                        kw2['density_initialization'] = phase
                state2 = feos.State(eos_final, **kw2)
                eta_pred = float(state2.viscosity() / (si.PASCAL * si.SECOND))
                eta_exp_Pas = float(eta_exp) * float(viscosity_unit / (si.PASCAL * si.SECOND))
                if np.isfinite(eta_pred) and eta_exp_Pas > 0:
                    ard_vals.append(abs(eta_pred - eta_exp_Pas) / eta_exp_Pas)
            except Exception:
                continue
        ard = 100.0 * float(np.mean(ard_vals)) if ard_vals else float("nan")
    else:
        Phi = np.column_stack([np.ones(n), s_arr, s_arr**2, s_arr**3])
        ard = 100.0 * float(np.mean(np.abs(np.expm1(Phi @ np.array(viscosity_list) - y_arr))))

    p_MPa_for_result = (
        [float('nan')] * len(T_data) if no_pressure_col
        else [float(P) * float(pressure_unit / (si.MEGA * si.PASCAL)) for P in P_data]
    )
    return ViscosityFitResult(
        viscosity_params=viscosity_list,
        eos=eos_final,
        ard=ard,
        n_points=n,
        input_name=compound_name,
        a_fixed=(A_fixed is not None),
        d_fixed=fix_d,
        groups=(dict(groups) if groups is not None else None),
        a_gc_used=a_gc_val,
        _T_K=[float(T) * float(temperature_unit / si.KELVIN) for T in T_data],
        _P_MPa=p_MPa_for_result,
        _eta_exp_Pa_s=[float(e) * float(viscosity_unit / (si.PASCAL * si.SECOND)) for e in eta_data],
        _s_vals=s_vals,
        _y_vals=y_arr.tolist(),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_EXP_COLOR = "#E32F2F"
_LINE_COLOR = "#000000"


def _scatter_kw(color: str, marker: str = "o") -> dict:
    return dict(s=40, marker=marker, facecolors="white", edgecolors=color,
                linewidths=1.2, zorder=5)


def _plot_viscosity_pure(result: ViscosityFitResult, path=None):
    """Two-panel: η vs T (left) and entropy scaling fit ln(η/η_CE) vs s (right)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    name = result.input_name or "compound"
    A, B, C, D = result.viscosity_params

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Left: η vs T ---
    ax = axes[0]
    T_exp = np.array(result._T_K)
    eta_exp_mPas = np.array(result._eta_exp_Pa_s) * 1e3

    ax.scatter(T_exp, eta_exp_mPas, label="Experiment", **_scatter_kw(_EXP_COLOR))

    if result.eos is not None:
        T_min, T_max = T_exp.min(), T_exp.max()
        T_pad = (T_max - T_min) * 0.05
        T_curve = np.linspace(T_min - T_pad, T_max + T_pad, 80)
        eta_curve = []
        for T in T_curve:
            try:
                state = feos.State(result.eos, temperature=T * si.KELVIN,
                                   pressure=0.1 * si.MEGA * si.PASCAL,
                                   total_moles=si.MOL,
                                   density_initialization="liquid")
                eta_curve.append(float(state.viscosity() / (si.PASCAL * si.SECOND)) * 1e3)
            except Exception:
                eta_curve.append(float("nan"))
        eta_curve = np.array(eta_curve)
        valid = np.isfinite(eta_curve)
        ax.plot(T_curve[valid], eta_curve[valid], color=_LINE_COLOR, label="PC-SAFT")

    ax.set_yscale("log")
    ax.set_xlabel("$T$ / K")
    ax.set_ylabel(r"$\eta$ / mPa·s")
    ax.set_title(f"Viscosity — {name}  (ARD {result.ard:.2f}%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="small")

    # --- Right: entropy scaling fit ---
    ax = axes[1]
    s_arr = np.array(result._s_vals)
    y_arr = np.array(result._y_vals)

    if len(s_arr) > 0:
        s_lo, s_hi = s_arr.min(), s_arr.max()
        s_pad = (s_hi - s_lo) * 0.05
        s_curve = np.linspace(s_lo - s_pad, s_hi + s_pad, 200)
        y_curve = A + B * s_curve + C * s_curve**2 + D * s_curve**3

        ax.scatter(s_arr, y_arr, label="Experiment", **_scatter_kw(_EXP_COLOR))
        ax.plot(s_curve, y_curve, color=_LINE_COLOR, label="Fit")

    ax.set_xlabel(r"$s = s_\mathrm{res}/(R\,m)$")
    ax.set_ylabel(r"$\ln(\eta/\eta_\mathrm{CE})$")
    ax.set_title(f"Entropy scaling fit — {name}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="small")

    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_viscosity_binary(
    params_mix,
    csv_path: "Path | str",
    id1: str = "component 1",
    id2: str = "component 2",
    path=None,
    csv_out=None,
    pressure_unit: "si.SIObject" = si.MEGA * si.PASCAL,
    viscosity_unit: "si.SIObject" = si.PASCAL * si.SECOND,
    induced_association: bool = False,
):
    """Plot binary mixture viscosity: η vs x₁ at each isotherm.

    Parameters
    ----------
    params_mix : feos.Parameters or feos.EquationOfState
        Binary parameters (or pre-built EOS) with viscosity set for both components.
    csv_path : Path | str
        CSV with columns ``T`` (K), optionally ``P`` (MPa), ``x_2pe`` or ``x1`` (mole
        fraction of component 1), ``eta`` (Pa·s).  If P is absent, 0.1 MPa is used.
    id1 : str
        Label for component 1 (x → 1).
    id2 : str
        Label for component 2 (x → 0).
    path : str or Path, optional
        If given, save the figure to this path (dpi=300).
    csv_out : str or Path, optional
        If given, write a CSV with columns T, P, x1, eta_exp, eta_pred, ard_pct.
    pressure_unit : si.SIObject
        Unit of the P column. Default: MPa.
    viscosity_unit : si.SIObject
        Unit of the eta column. Default: Pa·s.
    induced_association : bool
        If True, apply the induced-association mixing rule (solvation) before
        building the EOS — one component must be self-associating and the other
        not. Mirrors the same flag in BinaryKijFitter. Default: False.

    Returns
    -------
    fig, ax
    """
    import csv as _csv
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("talk")
    sns.set_style("ticks")

    # Build EOS — apply induced association if requested
    if hasattr(params_mix, 'pure_records'):
        if induced_association:
            rec1, rec2 = params_mix.pure_records
            rec1, rec2 = _apply_induced_association(rec1, rec2)
            params_mix = feos.Parameters.new_binary([rec1, rec2])
        eos_mix = feos.EquationOfState.pcsaft(params_mix)
    else:
        eos_mix = params_mix  # already an EOS

    df_raw = pl.read_csv(Path(csv_path))
    x_col = next((c for c in df_raw.columns if c.lower() in ("x_2pe", "x1", "x_1")), None)
    if x_col is None:
        raise ValueError(f"No mole fraction column found in {csv_path}. Expected x_2pe or x1.")

    T_arr = df_raw["T"].to_numpy()
    P_arr = df_raw["P"].to_numpy() if "P" in df_raw.columns else np.full(len(T_arr), 0.1)
    x_arr = df_raw[x_col].to_numpy()
    eta_arr = df_raw["eta"].to_numpy() * float(viscosity_unit / (si.PASCAL * si.SECOND)) * 1e3  # → mPa·s

    unique_T = np.unique(np.round(T_arr, 1))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(1, len(unique_T) - 1)) for i in range(len(unique_T))]

    fig, ax = plt.subplots(figsize=(9, 6))
    x_smooth = np.linspace(0.0, 1.0, 100)

    for k, T_iso in enumerate(unique_T):
        color = colors[k]
        iso_mask = np.abs(T_arr - T_iso) < 0.5
        P_iso = float(P_arr[iso_mask][0]) if iso_mask.any() else 0.1

        eta_model = []
        for x1 in x_smooth:
            try:
                state = feos.State(eos_mix, temperature=T_iso * si.KELVIN,
                                   pressure=P_iso * pressure_unit,
                                   total_moles=si.MOL,
                                   molefracs=np.array([x1, 1.0 - x1]))
                eta_model.append(float(state.viscosity() / (si.PASCAL * si.SECOND)) * 1e3)
            except Exception:
                eta_model.append(float("nan"))
        eta_model = np.array(eta_model)
        valid = np.isfinite(eta_model)
        ax.plot(x_smooth[valid], eta_model[valid], color=color, label=f"{T_iso:.1f} K")

        ax.scatter(x_arr[iso_mask], eta_arr[iso_mask],
                   s=40, marker="o", facecolors="white",
                   edgecolors=color, linewidths=1.2, zorder=5)

    ax.set_xlabel(rf"$x_1$ ({id1})")
    ax.set_ylabel(r"$\eta$ / mPa·s")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f"Viscosity: {id1} + {id2}")
    ax.legend(fontsize="small", title="$T$",
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(unique_T))
    sns.despine(offset=10)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")

    if csv_out is not None:
        csv_out = Path(csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        rows_out = []
        for T_iso, P_iso_csv, x1, eta_e_mPas in zip(T_arr, P_arr, x_arr, eta_arr):
            try:
                state = feos.State(eos_mix, temperature=T_iso * si.KELVIN,
                                   pressure=float(P_iso_csv) * pressure_unit,
                                   total_moles=si.MOL,
                                   molefracs=np.array([x1, 1.0 - x1]))
                eta_p_mPas = float(state.viscosity() / (si.PASCAL * si.SECOND)) * 1e3
            except Exception:
                eta_p_mPas = float("nan")
            eta_e_Pas = eta_e_mPas * 1e-3
            eta_p_Pas = eta_p_mPas * 1e-3
            ard = abs(eta_p_Pas - eta_e_Pas) / eta_e_Pas * 100 if np.isfinite(eta_p_Pas) else float("nan")
            rows_out.append({"T": T_iso, "P": float(P_iso_csv), "x1": x1,
                             "eta_exp": eta_e_Pas, "eta_pred": eta_p_Pas, "ard_pct": ard})
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["T", "P", "x1", "eta_exp", "eta_pred", "ard_pct"])
            w.writeheader()
            w.writerows(rows_out)

    return fig, ax
