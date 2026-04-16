"""Entropy scaling viscosity correlation fitting for pure PC-SAFT components.

Correlation model (Lötgering-Lin & Gross 2018, Ind. Eng. Chem. Res.):

    ln(η / η_CE) = A + B·s + C·s² + D·s³,   s = s_res / (R·m)

where η_CE is the Chapman-Enskog reference viscosity (computed from SAFT
σ and ε/k), s_res the residual molar entropy, R the gas constant, and m
the number of SAFT segments.  The four parameters [A, B, C, D] are fitted
by linear least squares.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import feos
import numpy as np
import polars as pl
import si_units as si

from fit_pcsaft._csv import SCHEMA_VISCOSITY, load_csv


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

    def __str__(self) -> str:
        A, B, C, D = self.viscosity_params
        lines = [
            "Entropy scaling viscosity parameters [A, B, C, D]:",
            f"  A: {A:+.6f}",
            f"  B: {B:+.6f}",
            f"  C: {C:+.6f}",
            f"  D: {D:+.6f}",
            "",
            "Fitting quality:",
            f"  ARD viscosity: {self.ard:.2f}%  (n={self.n_points})",
        ]
        if self.eos is None:
            lines.append("\n  Note: EOS rebuild failed — viscosity_params are still valid.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_viscosity_csv(path: "Path | str"):
    """Return (T, P, eta, phases) arrays from a viscosity CSV."""
    data = load_csv(path, SCHEMA_VISCOSITY)
    T = data["T"]
    P = data["P"]
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
            if 'association_sites' in mr:
                kw['association_sites'] = mr['association_sites']
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
) -> ViscosityFitResult:
    """Fit entropy scaling viscosity correlation ``[A, B, C, D]`` to experimental data.

    Uses linear least squares to fit:

        ln(η / η_CE) = A + B·s + C·s² + D·s³,   s = s_res / (R·m)

    where η_CE is the Chapman-Enskog reference viscosity from the PC-SAFT EOS.

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

    Returns
    -------
    ViscosityFitResult
        Fitted ``viscosity_params``, rebuilt EOS, ARD%, and point count.

    Raises
    ------
    RuntimeError
        If fewer than 4 data points could be evaluated successfully.

    Notes
    -----
    Individual data points at which the EOS fails to converge are silently
    skipped and do not contribute to the fit.
    """
    # --- Resolve EOS and m from source ------------------------------------
    if hasattr(source, 'pure_records'):
        # feos.Parameters
        rec = source.pure_records[0]
        m = float(rec.model_record['m'])
        eos = feos.EquationOfState.pcsaft(source)
        compound_name = name or rec.identifier.name
    else:
        # FitResult
        eos = source.eos
        m = source.params['m']
        compound_name = name or source.input_name

    # --- Load experimental data -------------------------------------------
    T_data, P_data, eta_data, phase_data = _load_viscosity_csv(viscosity_path)

    # --- Compute (s, y) pairs from EOS ------------------------------------
    s_vals: list[float] = []
    y_vals: list[float] = []

    for T, P, eta_exp, phase in zip(T_data, P_data, eta_data, phase_data):
        try:
            kw: dict = {
                'temperature': T * temperature_unit,
                'pressure': P * pressure_unit,
                'total_moles': si.MOL,
            }
            if isinstance(phase, str) and phase.lower() in ('liquid', 'vapor', 'vapour'):
                kw['density_initialization'] = phase

            state = feos.State(eos, **kw)
            s = state.molar_entropy(feos.Contributions.Residual) / si.RGAS / m
            y = float(np.log(eta_exp * viscosity_unit / state.viscosity_reference()))

            if np.isfinite(s) and np.isfinite(y):
                s_vals.append(s)
                y_vals.append(y)
        except Exception:
            continue

    n = len(s_vals)
    if n < 4:
        raise RuntimeError(
            f"Only {n} valid data points (need ≥ 4). "
            "Check that T/P conditions are within the EOS validity range."
        )

    s_arr = np.array(s_vals)
    y_arr = np.array(y_vals)

    # --- Linear regression: y = A + B*s + C*s^2 + D*s^3 ------------------
    Phi = np.column_stack([np.ones(n), s_arr, s_arr**2, s_arr**3])
    coeffs, _, _, _ = np.linalg.lstsq(Phi, y_arr, rcond=None)

    ard = 100.0 * float(np.mean(np.abs(np.expm1(Phi @ coeffs - y_arr))))
    viscosity_list = [float(v) for v in coeffs]

    eos_final = _rebuild_eos_with_viscosity(source, viscosity_list)

    return ViscosityFitResult(
        viscosity_params=viscosity_list,
        eos=eos_final,
        ard=ard,
        n_points=n,
        input_name=compound_name,
    )
