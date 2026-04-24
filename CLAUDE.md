# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run an example
uv run python examples/pure/01_propane.py
uv run python examples/binary/01_ethanol_water_vle.py

# Install/sync dependencies
uv sync

# Run tests (if any exist)
uv run pytest
```

Use `uv` for all package management — not `pip`.

## Architecture

`fit-pcsaft` wraps the [FeOs](https://github.com/feos-org/feos) PC-SAFT implementation with fitting logic powered by `scipy.optimize`. The public API is entirely in `src/fit_pcsaft/__init__.py`.

### Pure component fitting (`src/fit_pcsaft/_pure/`)

- **`fit.py`**: `fit_pure()`, `fit_pure_de()`, `eval_pure()` — the main entry points. `fit_pure` runs multi-start Levenberg-Marquardt (6–8 initial sets); `fit_pure_de` uses differential evolution (global optimizer). Both call `_setup_pure_fit()` to load data/build the cost function, then minimize and return a `FitResult`.
- **`jacobian.py`**: Analytical Jacobian via feos AD (`_make_f_and_df`). Used when neither `hvap_path` nor `q != 0` is given; otherwise falls back to numerical 2-point diff (`_make_f_and_df_numerical` in `_fit_utils.py`).

### Viscosity fitting (`src/fit_pcsaft/_pure/viscosity.py`)

Implements the Lötgering-Lin & Gross (LL 2018) entropy scaling correlation:

```
ln(η / η_CE) = A + B·s + C·s² + D·s³,   s = s_res / (R·m)
```

- **`fit_viscosity_entropy_scaling(source, viscosity_path, ...)`**: Main entry point. `source` is a `FitResult` or `feos.Parameters`. Returns `ViscosityFitResult` with fitted `[A, B, C, D]`. D is always fixed from the molar-mass correlation (LL 2018, eq. 14); A can be fixed via group contribution (`groups=` dict or `a_gc=` float); only B and C (or A, B, C) are regressed. Uses OLS by default (`loss='linear'`); robust losses (`'huber'`, `'cauchy'`, etc.) are supported.
- **`ViscosityFitResult`**: Frozen dataclass. `.to_json(path)` writes the `viscosity` field into a feos-compatible JSON file. `.to_csv(path)` exports predicted vs. experimental values. `.plot()` shows η vs T and the entropy-scaling fit.
- **`plot_viscosity_binary(params_mix, csv_path, ...)`**: Standalone function for plotting binary mixture viscosity η vs x₁ at each isotherm (also exported from the top-level package).
- **Viscosity CSV format**: Columns `T`, `P`, `eta` (pressure optional; if absent, P_sat is used for liquid). An optional `phase` string column (`'liquid'`/`'vapor'`) guides the EOS density root selection.
- **`viscosity_gc.py`**: `compute_a_gc(groups)` computes A_gc = Σ(n_α · m_α · σ_α³ · A_α) from LL 2015 segment parameters bundled in `viscosity_gc_data/loetgeringlin2015_homo.json`. `available_groups()` lists segments with viscosity parameters.

### Binary k_ij fitting (`src/fit_pcsaft/_binary/`)

- **`fitter.py`**: `BinaryKijFitter` — fluent builder. Chain `.add_vle()`, `.add_lle()`, `.add_vlle()`, `.add_sle()`, then call `.fit()`. Runs per-point k_ij fitting for each source, then fits a single k_ij(T) polynomial to the combined dataset.
- **`vle.py`, `lle.py`, `vlle.py`, `sle.py`**: Standalone `fit_kij_*` functions for single-data-type fitting (also used internally by `BinaryKijFitter`).
- **`vle_lle.py`**: `fit_kij_vle_lle()` — combined VLE+LLE two-stage fitting: fits per-point k_ij independently for VLE and LLE, then fits a single k_ij(T) polynomial to the combined pairs. Standalone only (not available via `BinaryKijFitter`).
- **`henry.py`**: `fit_kij_henry()` — fits k_ij from Henry's law constant data. Component 1 is the solute, component 2 the solvent. Supports `henry_unit="molfrac"` for dimensionless K = y₁/x₁ data (converts via K = H_feos / P_vap_solvent). Standalone only.
- **`_utils.py`**: Shared helpers — `_load_pure_records`, `_build_binary_eos`, `_kij_at_T`, `_apply_induced_association`, `_make_binary_jac_fn`, `_fit_kij_polynomial`.
- **`result.py`**: `BinaryFitResult` dataclass. Has `.plot()` and `.to_json()`.

### Shared utilities

- **`_fit_utils.py`**: `_fetch_compound` (PubChem lookup → `feos.Identifier`), `_build_eos` (assembles `feos.PureRecord` + `feos.EquationOfState`), `_make_cost_fn` (weighted relative residuals), `_make_f_and_df_numerical`.
- **`_types.py`**: Domain dataclasses — `PureData`, `Compound`, `ModelSpec`, `Units`, `FitConfig`.
- **`result.py`**: `FitResult` and `EvalResult`. `FitResult.to_json()` writes/updates a feos-compatible JSON parameter file (upserts by CAS or name).
- **`_csv.py`**: CSV loaders (`load_psat_csv`, `load_density_csv`, `load_hvap_csv`). Schema constants (`SCHEMA_HENRY`, `SCHEMA_VISCOSITY`, etc.) define column expectations for each data type.
- **`_plot.py`** / **`_binary/_plot.py`**: Plotting helpers for pure and binary results.

### Parameter transformation

The optimizer works on **sqrt-transformed parameters** (`x_internal = sqrt(params)`), so all parameters remain positive without explicit bounds. The cost function squares them internally (`params = x**2`). This applies to `fit_pure` / `fit_pure_de` only; binary fitting uses bounds directly.

### FeOs SI units pattern

All feos calls require SI quantities created as `value * si_unit`:
```python
feos.PhaseEquilibrium.vapor_pressure(eos, T * si.KELVIN)
feos.PhaseEquilibrium.pure(eos, T * si.KELVIN).liquid.mass_density() / (si.KILOGRAM / si.METER**3)
```

### Defaults

- Weights: `psat=3.0`, `rho=2.0`, `hvap=1.0`
- Default CSV units: K, kPa, kg/m³, kJ/mol; viscosity: K, MPa, Pa·s
- Multi-start initial sets: 6 for non-associating, 8 for associating
- mu is **never initialized at 0.0** (dipole Jacobian is identically zero there)

### Data format

CSVs: first column = temperature, second column = property. No header required; `polars` reads them directly.

### JSON parameter files

`FitResult.to_json(path)` appends/updates a list of feos `PureRecord`-compatible dicts. These files are loaded by `BinaryKijFitter` via `params_path`. `ViscosityFitResult.to_json(path)` patches the `viscosity` field into an existing entry in the same file format.
