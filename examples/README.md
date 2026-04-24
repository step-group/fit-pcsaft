# fit-pcsaft examples

## Quickstart

```bash
uv sync
uv run python examples/pure/01_propane.py
```

## Where to start

| Task | File |
|---|---|
| Non-associating pure component | `pure/01_propane.py` |
| Polar pure component (dipole) | `pure/02_acetone.py` |
| Associating pure component (2B) | `pure/03_ethanol.py` |
| Viscosity, pure | `pure/09_viscosity_hexane.py` |
| Simple binary VLE | `binary/01_vle_ethanol_water.py` |
| LLE screening over water models | `binary/02_lle_water_octanol.py` |
| Solid–liquid equilibrium | `binary/03_sle_ccl4_2-undecanone.py` |
| Henry's law constant | `binary/04_henry_2m3b2oh_water.py` |
| Combined VLE + LLE (`BinaryKijFitter`) | `binary/05_vle_lle_mibk_water.py` |
| Heteroazeotrope / VLLE (`BinaryKijFitter`) | `binary/06_vlle_2m2boh_water.py` |
| Binary viscosity mixing | `binary/07_viscosity_hexane_heptane.py` |

## Data layout

```
examples/data/
  psat/          — vapor pressure CSVs  (T [K], P [kPa])
  density/       — liquid density CSVs  (T [K], rho [kg/m³])
  hvap/          — enthalpy of vap. CSVs  (T [K], H [kJ/mol])
  vle/           — VLE bubble-point CSVs  (T, P, x1, y1)
  lle/           — LLE tie-line CSVs  (T, x1_I, x1_II)
  sle/           — SLE solubility CSVs  (T, x1)
  henry/         — Henry's law CSVs  (T, H)
  vle_lle/       — combined VLE+LLE and VLLE CSVs
  viscosity/     — viscosity CSVs  (T [K], P [MPa], eta [Pa·s])
  parameters/    — feos-compatible JSON parameter files
    examples_pure.json   — pure records written by the pure/ examples
    binary_params.json   — binary k_ij records + water and solvent models
    water_models.json    — 27 PC-SAFT water parameter sets
    alkanols_lle.json    — alkanol records used by the LLE sweep examples
```

## Output layout

`examples/out/` holds curated results checked into git. Running any example
regenerates the corresponding PNG(s) and writes/updates the shared JSON files
(`examples_pure.json`, `binary_kij.json`) via upsert — running multiple
examples accumulates all results in the same files without overwriting.

## Prerequisites for viscosity examples

`pure/09_viscosity_hexane.py` builds hexane PC-SAFT parameters inline (hexane
is not in `examples_pure.json`). `binary/07_viscosity_hexane_heptane.py` does
the same for both hexane and heptane. No prior steps are required.
