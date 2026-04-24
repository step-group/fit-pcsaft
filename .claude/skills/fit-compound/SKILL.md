---
name: fit-compound
description: Use when starting to fit a new pure PC-SAFT compound from scratch, when uncertain which ModelSpec flags to use, or when a fit converges to a bad local minimum and needs diagnosis.
---

## Overview

Structured checklist for fitting a new pure component: select ModelSpec flags → prepare CSVs → run `fit_pure` → evaluate ARD → validate parameter ranges → write to JSON.

## ModelSpec Flags (decide first)

| Question | Flag |
|---|---|
| Dipole moment > 0.5 D? (ketone, ester, nitrile…) | `fit_mu=True` |
| Alcohol, carboxylic acid, water, amine? | `na=1, nb=1` (or `na=2, nb=2` for water) + optionally `fit_kappa_ab=True` |
| Quadrupole significant? (CO₂, benzene, naphthalene) | `q=<fixed float>` |
| None of the above | no flags needed |

## Workflow

1. **Confirm CSV paths** — two-column, no header, default units K / kPa / kg/m³. Must be under `data/`.
2. **Run `fit_pure`:**
   ```python
   from fit_pcsaft import fit_pure
   result = fit_pure(
       compound="<name or CAS>",
       psat_path="data/<name>_psat.csv",
       rho_path="data/<name>_rho.csv",
       # ModelSpec flags here, e.g. fit_mu=True
   )
   print(result)
   result.plot()
   ```
3. **Check ARD** — flag if psat ARD > 5 % or density ARD > 2 %; likely a local minimum.
4. **Validate parameter ranges:**

| Parameter | Expected range | Flag if… |
|---|---|---|
| m | 1–15 | > 10 for MW < 300 g/mol |
| σ / Å | 2.5–5.5 | outside |
| ε/k / K | 100–600 | outside |
| κ_ab | 0.001–0.20 | < 0.001 (wrong min) |
| ε_ab/k / K | 1000–4000 | outside |

5. **Write to JSON:** `result.to_json("feos/<params_file>.json")` — upserts by CAS or name, safe to call repeatedly.

## Gotchas

- **mu must never start at 0.0** — the dipole Jacobian is identically zero there; `fit_pure` handles this automatically (never pass `mu=0`).
- **Bad associating fit?** Switch to `fit_pure_de` (differential evolution) as a global-search fallback.
- **Weights:** defaults are `psat=3.0`, `rho=2.0` — only override if one data set is clearly noisy.
