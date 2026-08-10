"""Regenerate the water reference data used by examples/pure/11_water_pareto.py.

Run:  uv run python examples/data/generate_water_reference.py

Vapour pressure and saturated liquid density come from **IAPWS-95**, evaluated
through feos' own multiparameter Helmholtz equation of state
(``EquationOfState.multiparameter``) using the CoolProp coefficient database
vendored at ``examples/data/parameters/coolprop_multiparameter.json``. No extra
dependency: feos is already required, and this is the same library that
evaluates PC-SAFT, so reference and model are computed by one code path.

Surface tension is not a quantity IAPWS-95 provides — it is a separate
correlation, IAPWS R1-76(2014), reproduced below. Its four constants and the
values it produces were checked against the published release: the equation
reproduces Table 1 column 4 at 0.01, 25, 200, 250, 300, 350 and 370 degC.
    https://iapws.org/public/documents/CH-L9/Surf-H2O-2014.pdf

Temperature ranges follow Rehner & Gross (2020), with one deviation: bulk data
stops at Tr = 0.958. Classical PC-SAFT has no critical scaling, so saturated
density deviation reaches 20% by Tr = 0.99 and would swamp a relative objective
with the model's structural error rather than its parameter error. Surface
tension is well behaved there and runs to Tr = 0.989.
"""

from pathlib import Path

import feos
import numpy as np
import si_units as si

DATA = Path(__file__).parent
PARAMS = DATA / "parameters" / "coolprop_multiparameter.json"

TC = 647.096  # K, IAPWS critical temperature

N_BULK, T_BULK = 23, (280.0, 620.0)   # Tr <= 0.958
N_SFT, T_SFT = 25, (280.0, 640.0)     # Tr <= 0.989


def surface_tension_mN_m(T):
    """IAPWS R1-76(2014): sigma = B tau^mu (1 - b tau), tau = 1 - T/Tc."""
    tau = 1.0 - np.asarray(T, dtype=float) / TC
    return 235.8 * tau**1.256 * (1.0 - 0.625 * tau)


def _write(path: Path, header: str, T, vals, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{t:.2f},{v:{fmt}}" for t, v in zip(T, vals))
    path.write_text(f"{header}\n{rows}\n")
    print(f"  wrote {path.relative_to(DATA.parent.parent)}  ({len(T)} points)")


def main() -> None:
    eos = feos.EquationOfState.multiparameter(
        feos.Parameters.from_json(
            ["Water"], str(PARAMS), identifier_option=feos.IdentifierOption.Name
        )
    )

    T = np.linspace(*T_BULK, N_BULK)
    psat = np.array([
        float(feos.PhaseEquilibrium.vapor_pressure(eos, t * si.KELVIN)[0]
              / (si.KILO * si.PASCAL))
        for t in T
    ])
    rho = np.array([
        float(feos.PhaseEquilibrium.pure(eos, t * si.KELVIN).liquid.mass_density()
              / (si.KILOGRAM / si.METER**3))
        for t in T
    ])
    _write(DATA / "psat" / "water.csv", "T,psat", T, psat, ".5f")
    _write(DATA / "density" / "water.csv", "T,rho", T, rho, ".3f")

    Tg = np.linspace(*T_SFT, N_SFT)
    _write(DATA / "surface_tension" / "water.csv", "T,sft",
           Tg, surface_tension_mN_m(Tg), ".4f")

    # Anchors that are definitional or published, so a bad regeneration is loud.
    p_nbp = float(feos.PhaseEquilibrium.vapor_pressure(eos, 373.124 * si.KELVIN)[0]
                  / (si.KILO * si.PASCAL))
    print("\nchecks:")
    print(f"  psat(373.124 K) = {p_nbp:.5f} kPa   (normal boiling point: 101.325)")
    print(f"  sigma(300 degC) = {surface_tension_mN_m(573.15):.2f} mN/m "
          f"(R1-76 Table 1 col 4: 14.36)")


if __name__ == "__main__":
    main()
