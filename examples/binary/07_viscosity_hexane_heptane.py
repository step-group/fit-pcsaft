"""Example: entropy scaling viscosity for a hexane + heptane mixture.

PC-SAFT entropy scaling uses linear mole-fraction mixing for viscosity:

    A_mix = x₁·A₁ + x₂·A₂,  (same for B, C, D)
    m_mix = x₁·m₁ + x₂·m₂
    s = s_res / (R · m_mix)
    ln(η / η_CE) = A_mix + B_mix·s + C_mix·s² + D_mix·s³

So once the pure component viscosity parameters are fitted, mixture
viscosity follows with no additional fitting.

Workflow
--------
1. Fit viscosity for each pure component (see examples/pure/09_viscosity_hexane.py).
2. Build a binary EOS combining both records.
3. Compute mixture viscosity and compare to experimental data.

Data
----
The placeholder CSVs were generated from model predictions as a consistency
check. Replace with real measurements for production use.
"""

from pathlib import Path

import feos
import numpy as np
import polars as pl
import si_units as si

from fit_pcsaft import fit_viscosity_entropy_scaling

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"


def _make_params(cas, name, iupac, smiles, inchi, formula, mw, m, sigma, eps):
    ident = feos.Identifier(
        cas=cas, name=name, iupac_name=iupac,
        smiles=smiles, inchi=inchi, formula=formula,
    )
    rec = feos.PureRecord(identifier=ident, molarweight=mw, m=m, sigma=sigma, epsilon_k=eps)
    return feos.Parameters.new_pure(rec)


def _add_viscosity(record: feos.PureRecord, viscosity: list) -> feos.PureRecord:
    """Rebuild a PureRecord with a viscosity field added."""
    mr = record.model_record
    kwargs = {
        "identifier": record.identifier,
        "molarweight": record.molarweight,
        "m": mr["m"],
        "sigma": mr["sigma"],
        "epsilon_k": mr["epsilon_k"],
        "viscosity": viscosity,
    }
    for opt in ("mu", "q"):
        if opt in mr:
            kwargs[opt] = mr[opt]
    if "association_sites" in mr:
        kwargs["association_sites"] = mr["association_sites"]
    return feos.PureRecord(**kwargs)


params_hexane = _make_params(
    cas="110-54-3", name="hexane", iupac="hexane",
    smiles="CCCCCC",
    inchi="InChI=1S/C6H14/c1-3-5-6-4-2/h3-6H2,1-2H3",
    formula="C6H14",
    mw=86.177, m=3.0576, sigma=3.7983, eps=236.77,
)
params_heptane = _make_params(
    cas="142-82-5", name="heptane", iupac="heptane",
    smiles="CCCCCCC",
    inchi="InChI=1S/C7H16/c1-3-5-7-6-4-2/h3-7H2,1-2H3",
    formula="C7H16",
    mw=100.204, m=3.4831, sigma=3.8049, eps=238.40,
)


def main() -> None:
    visc_hexane  = data_dir / "viscosity" / "hexane_viscosity.csv"
    visc_heptane = data_dir / "viscosity" / "heptane_viscosity.csv"

    print("=== Hexane viscosity fit ===")
    res_hex = fit_viscosity_entropy_scaling(params_hexane, visc_hexane, name="hexane")
    print(res_hex)

    print("\n=== Heptane viscosity fit ===")
    res_hep = fit_viscosity_entropy_scaling(params_heptane, visc_heptane, name="heptane")
    print(res_hep)

    # Accessing eos.parameters.pure_records on an EOS that already carries
    # viscosity params triggers a feos deserialization panic — rebuild instead.
    binary_params_visc = feos.Parameters.new_binary(
        [
            _add_viscosity(params_hexane.pure_records[0], res_hex.viscosity_params),
            _add_viscosity(params_heptane.pure_records[0], res_hep.viscosity_params),
        ],
        k_ij=0.0,  # k_ij ≈ 0 for similar n-alkanes; adjust for other systems
    )
    eos_mix = feos.EquationOfState.pcsaft(binary_params_visc)

    mix_data_path = data_dir / "viscosity" / "hexane_heptane_viscosity.csv"
    df = pl.read_csv(mix_data_path)

    T_col   = df["T"].to_numpy()
    P_col   = df["P"].to_numpy() if "P" in df.columns else None
    x1_col  = df["x_hexane"].to_numpy()
    eta_exp = df["eta"].to_numpy()

    print("\n=== Mixture viscosity: experiment vs prediction ===")
    print(f"{'T/K':>7}  {'x_hex':>6}  {'η_exp/mPa·s':>12}  {'η_pred/mPa·s':>13}  {'ARD/%':>7}")

    ard_vals = []
    P_iter = P_col if P_col is not None else [0.1] * len(T_col)
    for T, P, x1, eta_e in zip(T_col, P_iter, x1_col, eta_exp):
        try:
            state = feos.State(
                eos_mix,
                temperature=T * si.KELVIN,
                pressure=P * si.MEGA * si.PASCAL,
                total_moles=si.MOL,
                molefracs=np.array([x1, 1.0 - x1]),
            )
            eta_p = float(state.viscosity() / (si.PASCAL * si.SECOND))
            ard = abs(eta_p - eta_e) / eta_e * 100
            ard_vals.append(ard)
            print(f"{T:>7.2f}  {x1:>6.2f}  {eta_e*1e3:>12.4f}  {eta_p*1e3:>13.4f}  {ard:>7.2f}")
        except Exception:
            print(f"{T:>7.2f}  {x1:>6.2f}  {'(failed)':>12}")

    if ard_vals:
        print(f"\nMARD: {np.mean(ard_vals):.2f}%  (n={len(ard_vals)})")


if __name__ == "__main__":
    main()
