"""Example: fit entropy scaling viscosity parameters for hexane.

Fits [A, B, C, D] in:

    ln(η / η_CE) = A + B·s + C·s²  + D·s³,   s = s_res / (R·m)

Procedure (Lötgering-Lin & Gross 2018):

- D is fixed from the molar-mass correlation (eq. 14):
      D = 1 / (−1.25594 − 888.1232 / M)  →  −0.0865 for hexane
- A is fixed from group contribution (LL 2015, eq. 11):
      A_gc = Σ_α (n_α · m_α · σ_α³ · A_α)
      A_i  = A_gc + 0.5·ln(1/m)           →  ≈ −1.205 for hexane
- Only B and C are fit from experimental viscosity data.

Reference (Lötgering-Lin & Gross 2018):
  A=-1.2035, B=-2.5958, C=-0.4816, D=-0.0865
"""

import json
from pathlib import Path

import feos
import si_units as si

from fit_pcsaft import fit_viscosity_entropy_scaling

data_dir = Path(__file__).parent.parent / "data"
out_dir = Path(__file__).parent.parent / "out"

# PC-SAFT parameters for hexane (Gross & Sadowski 2001)
ident = feos.Identifier(
    cas="110-54-3",
    name="hexane",
    iupac_name="hexane",
    smiles="CCCCCC",
    inchi="InChI=1S/C6H14/c1-3-5-6-4-2/h3-6H2,1-2H3",
    formula="C6H14",
)
record = feos.PureRecord(
    identifier=ident,
    molarweight=86.177,
    m=3.0576,
    sigma=3.7983,
    epsilon_k=236.77,
)
params = feos.Parameters.new_pure(record)

visc_path = data_dir / "viscosity" / "hexane_viscosity.csv"


def main() -> None:
    result = fit_viscosity_entropy_scaling(
        params,
        visc_path,
        name="hexane",
        groups={"CH3": 2, "CH2": 4},
    )
    print(result)

    # ViscosityFitResult.to_json requires an existing entry for "hexane".
    # Bootstrap a minimal JSON for standalone use (hexane is not in examples_pure.json).
    json_out = out_dir / "hexane_with_viscosity.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not json_out.exists():
        entry = {
            "identifier": {k: getattr(ident, k) for k in
                           ("cas", "name", "iupac_name", "smiles", "inchi", "formula")},
            "molarweight": record.molarweight,
            **{k: record.model_record[k] for k in ("m", "sigma", "epsilon_k")},
        }
        json_out.write_text(json.dumps([entry], indent=2))

    result.to_json(json_out)
    print(f"\nViscosity parameters written to {json_out}")

    state = feos.State(
        result.eos,
        temperature=298.15 * si.KELVIN,
        pressure=0.1 * si.MEGA * si.PASCAL,
        total_moles=si.MOL,
        density_initialization="liquid",
    )
    eta_pred = state.viscosity() / (si.PASCAL * si.SECOND)
    print(f"Predicted viscosity at 298.15 K, 0.1 MPa (liquid): {eta_pred*1e3:.3f} mPa·s")


if __name__ == "__main__":
    main()
