---
name: fit-compound
description: Use when starting to fit a new pure PC-SAFT compound from scratch, when uncertain which fit_pure arguments a compound needs, or when a fit converges to a bad local minimum and needs diagnosis.
---

## Overview

Structured checklist for fitting a new pure component: choose the arguments → prepare
CSVs → run `fit_pure` → evaluate ARD → validate parameter ranges → write to JSON.

## Arguments to decide first

All are keyword arguments of `fit_pure` / `fit_pure_de`. **There are no `fit_*` booleans**
— what gets fitted is decided by the value you pass.

| Question | Argument |
|---|---|
| Dipole moment > 0.5 D? (ketone, ester, nitrile…) | `mu=None` — passing `None` is what *fits* it |
| Dipole known, and you want it held there | `mu=<float>` |
| Non-polar | `mu=0.0` (the default) |
| Alcohol, carboxylic acid, amine? | `na=1, nb=1` |
| Water | `na=2, nb=2` |
| Quadrupole significant? (CO₂, benzene, naphthalene) | `q=<float>` — always fixed, never fitted |
| None of the above | nothing to pass |

Association (`kappa_ab`, `epsilon_k_ab`) is fitted whenever `na`/`nb` are set. There is no
separate switch for it.

Verify against the source rather than trusting this table if a call raises `TypeError`:

```bash
uv run python -c "import inspect, fit_pcsaft; print(inspect.signature(fit_pcsaft.fit_pure))"
```

## Workflow

1. **Prepare the CSVs** — a **header row is required**. Column names are normalised through
   `_COL_ALIASES` in `_csv.py` and validated against a `CsvSchema`; unrecognised columns are
   dropped silently. A unit suffix in a name (`psat_kPa`, `rho_kg_m3`) is a hint only — **no
   numeric conversion happens in the loaders**. Units are carried entirely by the `*_unit`
   keyword arguments, defaulting to K / kPa / kg m⁻³. Any path works; the files need not be
   under `data/`.

2. **Run `fit_pure`:**

   ```python
   import fit_pcsaft

   result = fit_pcsaft.fit_pure(
       id="thymol",                       # name, SMILES or InChI — PubChem lookup
       psat_path="data/thymol_psat.csv",
       density_path="data/thymol_rho.csv",
       na=1,
       nb=1,
       mu=0.0,
       loss="cauchy",                     # optional: robust loss for noisy literature data
   )
   print(result.params)
   result.plot()
   ```

   `id` is positional-first and is the compound identifier — there is no `compound=`
   argument. The density argument is `density_path`, not `rho_path`.

3. **Check ARD** — flag if psat ARD > 5 % or density ARD > 2 %; likely a local minimum.

4. **Validate parameter ranges:**

| Parameter | Expected range | Flag if… |
|---|---|---|
| m | 1–15 | > 10 for MW < 300 g/mol |
| σ / Å | 2.5–5.5 | outside |
| ε/k / K | 100–600 | outside |
| κ_ab | 0.001–0.20 | < 0.001 (wrong min) |
| ε_ab/k / K | 1000–4000 | outside |

5. **Write to JSON:** `result.to_json(path)` — appends or updates a list of feos
   `PureRecord`-compatible dicts, upserting by CAS or name, so it is safe to call
   repeatedly. Any path.

## Gotchas

- **`mu=0.0` is the default and means "non-polar, held at zero"** — correct for an alkane or
  a terpene hydrocarbon. To *fit* the dipole, pass `mu=None`. The optimizer never
  *initialises* it at 0.0 in that case, because the dipole Jacobian is identically zero
  there; `fit_pure` handles that internally, so there is nothing for you to do.
- **Bad associating fit?** Switch to `fit_pure_de` (differential evolution) as a
  global-search fallback. Same arguments.
- **Weights:** defaults are `psat_weight=3.0`, `density_weight=2.0` — only override if one
  data set is clearly noisy.
- **Near the critical point** classical PC-SAFT has no critical scaling. Data above about
  Tr = 0.96 carries the model's structural error rather than parameter error and will
  dominate a relative objective.
