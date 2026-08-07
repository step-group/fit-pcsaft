# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run an example
uv run python examples/pure/01_propane.py
uv run python examples/binary/01_ethanol_water_vle.py
uv run python examples/pure/11_water_pareto.py   # ~4 min

# Install/sync dependencies
uv sync

# Run tests
uv run python -m pytest
```

`uv run pytest` fails to spawn in this checkout; use `uv run python -m pytest`.

Use `uv` for all package management — not `pip`.

## Architecture

`fit-pcsaft` wraps the [FeOs](https://github.com/feos-org/feos) PC-SAFT implementation with fitting logic powered by `scipy.optimize`. The public API is entirely in `src/fit_pcsaft/__init__.py`.

### Pure component fitting (`src/fit_pcsaft/_pure/`)

- **`fit.py`**: `fit_pure()`, `fit_pure_de()`, `eval_pure()` — the main entry points. `fit_pure` runs multi-start Levenberg-Marquardt (6–8 initial sets); `fit_pure_de` uses differential evolution (global optimizer). Both call `_setup_pure_fit()` to load data/build the cost function, then minimize and return a `FitResult`.
- **`jacobian.py`**: Analytical Jacobian via feos AD (`_make_f_and_df`). Used unless `sft_path` is given or `q != 0`; otherwise falls back to numerical 2-point diff (`_make_f_and_df_numerical` in `_fit_utils.py`). Surface tension has no feos AD path at all, so `_make_f_and_df` raises if `sft` data is present rather than silently returning a short residual vector.

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

### Surface tension and pareto fitting (`src/fit_pcsaft/_pure/`)

Implements Rehner & Gross (*J. Chem. Eng. Data* 2020, 65, 5698–5707): surface tension as a second, independent objective, to break the parameter degeneracy that vapour-pressure-plus-density fits leave behind for associating fluids.

- **`surface_tension.py`**: `predict_surface_tension(functional, T_vals, units, options)` — γ for a pure component from feos DFT, pDGT-initialized. Returns NaN wherever the VLE or interface solve fails; never raises. `SurfaceTensionOptions(n_grid=256)` is already grid-converged (~5 ms/point; `n_grid=2048` gives the same six digits at 8× the cost).
  - feos hardcodes `psi_dft = 1.3862` and its Python `PlanarInterface.from_pdgt` discards the bare pDGT surface tension, so results track the paper's **AAD_DFT** column, not AAD_pDGT.
  - Catches `BaseException`: feos signals unsupported input with `pyo3_runtime.PanicException`, which is not an `Exception`.
- **`pareto.py`**: `fit_pure_pareto(...)` → `ParetoResult`, solved with pymoo NSGA-II (derivative-free — large regions of parameter space have no VLE at all, handled as a constraint violation).
  - Objectives: `AAD_vle = mean(AARD(psat), AARD(rho))` in %, and `AAD_sft = mean|Δγ|` in mN/m. γ is an **absolute** deviation because it goes to zero at the critical point, where a relative one diverges.
  - `ParetoResult.select(ref_vle, ref_sft)` returns the eq-32 tangent point as an ordinary `FitResult`, so `to_json`/`plot`/`metrics_table` all work. Paper weights: water `(2%, 0.7)`, small alcohols `(2%, 1.5)`, C5+ alcohols `(2%, 3.0)`.
  - Infeasible parameter sets report a **graded** constraint violation (`0.5 - worst_valid_fraction`), not a flat penalty. About 80% of the default bounds box has no stable interface for an associating fluid; collapsing all of it onto one `(1e6, 1e6)` leaves NSGA-II unable to tell "failed two of 25 gamma points" from "no VLE at all".
  - `n_jobs=-1` (default) evaluates the population across worker processes — every core bar two. **Processes, not threads: feos does not release the GIL** (measured 0.89x on a 16-thread pool, i.e. slower than serial). `feos.Identifier` is also unpicklable, so pymoo's `StarmapParallelization` cannot be used — the problem is a vectorized `Problem` that maps the population itself, and workers rebuild their own feos objects in an initializer. The pool uses **spawn**, since forking a process with feos' live rayon threads risks a child deadlock, and calls `feos.set_num_threads(1)` per worker to avoid oversubscription. Measured 4.3x on 14 workers with identical results.
  - `quiet_solver=True` (default) silences fd-level stderr during the search; feos panics from Rust worker threads on infeasible parameter sets and Python cannot intercept those writes.
- **Surface tension CSV format**: columns `T`, `sft` (aliases `gamma`, `surface_tension`, `sigma_st`, `st`). Default unit mN/m. `FitResult.aad_sft` is the paper's AAD_sft; `ard_sft` is the relative AARD.
- **Fitting near the critical point**: classical PC-SAFT has no critical scaling. Bulk data above about Tr = 0.96 carries the model's structural error rather than parameter error and will dominate a relative objective — the bundled water quasi-data stops at 620 K for psat/rho (Tr = 0.958) while γ runs to 640 K.
- **Validated** in `tests/test_pareto.py` against Tables 1 and 2 of the paper: water 2B/3B/4C reproduce AAD_DFT = 1.62/1.16/1.84 mN/m against the published 1.59/1.14/1.81, and the scheme ranking (4C best bulk, 3B best interface) holds.

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

- Weights: `psat=3.0`, `rho=2.0`, `hvap=1.0`, `sft=1.0`
- Default surface tension unit: mN/m
- Default CSV units: K, kPa, kg/m³, kJ/mol; viscosity: K, MPa, Pa·s
- Multi-start initial sets: 6 for non-associating, 8 for associating
- mu is **never initialized at 0.0** (dipole Jacobian is identically zero there)

### Data format

CSVs **require a header**. Column names are normalised through `_COL_ALIASES` in `_csv.py` and validated against a `CsvSchema`; unrecognised columns are dropped silently. Aliases carrying unit suffixes (`psat_kPa`, `rho_kg_m3`) are name hints only — **no numeric conversion happens in the loaders**. Units are carried entirely by the `*_unit` keyword arguments into the `Units` dataclass and applied at prediction time.

### Reference (quasi-)data

The water files under `examples/data/{psat,density,surface_tension}/water.csv` are quasi-data, following the paper's own practice of discretising correlations so the fit is not biased toward temperature ranges where raw measurements happen to be dense. Regenerate with:

```bash
uv run python examples/data/generate_water_reference.py
```

- **psat, rho** — IAPWS-95, evaluated through feos' own `EquationOfState.multiparameter` using the CoolProp coefficient database vendored at `examples/data/parameters/coolprop_multiparameter.json` (124 fluids). No extra dependency, and the reference and the PC-SAFT model are computed by the same library.
- **γ** — IAPWS R1-76(2014); IAPWS-95 does not provide surface tension. Constants verified verbatim against the [published release](https://iapws.org/public/documents/CH-L9/Surf-H2O-2014.pdf); the equation reproduces its Table 1 column 4 at 0.01/25/200/250/300/350/370 °C.

**Do not hand-write reference correlations from memory.** feos' multiparameter EOS plus that JSON covers any of the 124 fluids; reach for it instead.

### JSON parameter files

`FitResult.to_json(path)` appends/updates a list of feos `PureRecord`-compatible dicts. These files are loaded by `BinaryKijFitter` via `params_path`. `ViscosityFitResult.to_json(path)` patches the `viscosity` field into an existing entry in the same file format.
