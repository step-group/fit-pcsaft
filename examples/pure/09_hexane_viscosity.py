"""
Example: Fit entropy scaling viscosity parameters for hexane.

Fits [A, B, C, D] in:

    ln(η / η_CE) = A + B·s + C·s²  + D·s³,   s = s_res / (R·m)

Procedure (Lötgering-Lin & Gross 2018):

- D is fixed from the molar-mass correlation (eq. 14):
      D = 1 / (−1.25594 − 888.1232 / M)  →  −0.0865 for hexane
- A is fixed from group contribution (LL 2015, eq. 11):
      A_gc = Σ_α (n_α · m_α · σ_α³ · A_α)
      A_i  = A_gc + 0.5·ln(1/m)           →  ≈ −1.205 for hexane
- Only B and C are fit from experimental viscosity data.

Workflow
--------
1. Load PC-SAFT parameters.
2. Provide viscosity data as a CSV with columns T (K), P (MPa), eta (Pa·s).
3. Call fit_viscosity_entropy_scaling with groups={"CH3": 2, "CH2": 4}.
4. Write the result back into a feos-compatible JSON.

Reference (Lötgering-Lin & Gross 2018):
  A=-1.2035, B=-2.5958, C=-0.4816, D=-0.0865
"""

from pathlib import Path

import feos

from fit_pcsaft import fit_viscosity_entropy_scaling

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "out"

# --------------------------------------------------------------------------
# 1. Load PC-SAFT parameters
#    Option A: from a JSON written by fit_pure (recommended)
#        params = feos.Parameters.from_json(["hexane"], OUT_DIR / "examples_pure.json")
#
#    Option B: build manually with known values
# --------------------------------------------------------------------------
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
    # no viscosity field yet — that is what we are fitting
)
params = feos.Parameters.new_pure(record)

# --------------------------------------------------------------------------
# 2. Viscosity data
#    CSV columns: T (K), P (MPa), eta (Pa·s), phase (liquid/vapor, optional)
#    Replace with your actual experimental data.
# --------------------------------------------------------------------------
visc_path = DATA_DIR / "viscosity" / "hexane_viscosity.csv"


def main() -> None:
    result = fit_viscosity_entropy_scaling(
        params,
        visc_path,
        name="hexane",
        groups={"CH3": 2, "CH2": 4},
        # Alternative: pass a_gc=<float> if you computed A_gc externally
    )
    print(result)

    # ------------------------------------------------------------------
    # 3. Write viscosity params into a feos-compatible JSON
    #    The JSON must already contain an entry for "hexane"
    #    (created by fit_pure → result.to_json, or written manually).
    # ------------------------------------------------------------------
    json_out = OUT_DIR / "hexane_with_viscosity.json"

    # Bootstrap a minimal JSON if the file doesn't exist yet
    import json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not json_out.exists():
        entry = {
            "identifier": {
                "cas": ident.cas,
                "name": ident.name,
                "iupac_name": ident.iupac_name,
                "smiles": ident.smiles,
                "inchi": ident.inchi,
                "formula": ident.formula,
            },
            "molarweight": record.molarweight,
            "m": record.model_record["m"],
            "sigma": record.model_record["sigma"],
            "epsilon_k": record.model_record["epsilon_k"],
        }
        json_out.write_text(json.dumps([entry], indent=2))

    result.to_json(json_out)
    print(f"\nViscosity parameters written to {json_out}")

    # ------------------------------------------------------------------
    # 4. Quick sanity check: predict viscosity at a single state point
    # ------------------------------------------------------------------
    import si_units as si
    state = feos.State(
        result.eos,
        temperature=298.15 * si.KELVIN,
        pressure=0.1 * si.MEGA * si.PASCAL,
        total_moles=si.MOL,
        density_initialization="liquid",
    )
    eta_pred = state.viscosity() / (si.PASCAL * si.SECOND)
    print(f"\nPredicted viscosity at 298.15 K, 0.1 MPa (liquid): {eta_pred*1e3:.3f} mPa·s")


if __name__ == "__main__":
    main()
