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

`pyproject.toml`'s `[tool.uv.sources]` pins `feos` to a locally-compiled wheel at
`../feos/dist/feos-0.10.1-cp310-abi3-macosx_11_0_arm64.whl` (full LTO,
`-Ctarget-cpu=native`, Cargo features trimmed to `pcsaft dft multiparameter ad
rayon`) — same version as PyPI's, different bytes, marker-gated to
darwin+arm64. **If that wheel is absent, `uv sync` fails.** Escape hatch:
delete the `[tool.uv.sources]` `feos` block and run `uv lock` — that is the
whole fix, falling back to the stock PyPI wheel. If you rebuild the wheel at
the same version, plain `uv lock` reuses the cached hash and will not notice
the new bytes; run `uv lock --upgrade-package feos` afterward to keep the
committed lock honest. Skipping that is harmless at runtime (`uv sync`/`uv
run` don't re-verify the hash) — it just leaves `uv.lock` describing stale
bytes.

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
- **`pareto.py`**: `fit_pure_pareto(...)` → `ParetoResult`, solved with pymoo MOEA/D, in its ParallelMOEAD variant (derivative-free — large regions of parameter space have no VLE at all; plain MOEAD is loopwise and would feed the worker pool one point at a time). An `objectives` pair picks the mode — see the new bullet below.
  - **Two `objectives` pairs, both `n_obj=2`.** `("vle", "sft")` (default) is Rehner & Gross 2020: `AAD_vle = mean(AARD(psat), AARD(rho))` in %, and `AAD_sft = mean|Δγ|` in mN/m — γ is an **absolute** deviation because it goes to zero at the critical point, where a relative one diverges. `("psat", "rho")` is Forte et al. 2018: the same two bulk AARDs, each on its own axis instead of pooled behind eq 30, with surface tension out of the objectives and (`sft_path` permitting) no DFT solve anywhere in the search itself — dropping `"sft"` from `objectives` also drops it from the graded-violation term (see the next bullet), so a parameter set with no stable interface is not penalized under this pair. `sft_path` stays optional and, if passed, is still loaded and still reported — see the `hvap_path` bullet below for the one place it does DFT-solve, once, in `.select()`. Unsupported pairs, and `("vle", "sft")` without `sft_path`, raise `ValueError` before fitting starts. Measured on this machine's locally-built feos wheel, whole-search average over a 600-evaluation budget (`pop_size=30, n_gen=20, n_restarts=1, refine=0, seed=1, n_jobs=1`, same surface-tension file loaded by both modes so only the solving differs): `("psat", "rho")` averaged 3.69 ms/evaluation (2.2 s total) against 103.33 ms/evaluation (62.0 s total) for `("vle", "sft")` — **28× cheaper per evaluation**. That run also returned 15 front points against 2; **do not** read that as a front-quality comparison — 600 evaluations at `refine=0`/`n_restarts=1` is far too small a budget for the sft mode, and the gap is a budget artefact, not a resolution one.
  - **Eq 30 is a mean, not a sum, and this has been checked the hard way.** Written out it looks like a sum of two separately-normalized terms, and that reading was implemented and reverted. Forward-evaluating the paper's own water 2B parameters on IAPWS-95 saturation data gives `AARD_psat = 4.56%`, `AARD_rho = 1.76%` → sum 6.32, mean 3.16, against their reported `AAD_vle = 2.14%`. **AARD_psat alone exceeds their whole reported value**, so no summed form reaches 2.14 for parameters that also reproduce their surface tension (which these do, to 0.03 mN/m on all three schemes). The psat error is structural, not a range artefact — `+11.5%` at 280 K, `−5.4%` at 373 K, `+7.7%` at 620 K, still 3.70% if truncated to 500 K. The mean is also what a pooled `1/N_total` average reduces to at equal set sizes (23 and 23 here), which is the likely reconciliation given the paper's much larger density set. `tests/test_pareto.py` guards this with a two-sided ratio band; the upper edge is what catches a revert to the sum.
  - `ParetoResult.select(refs)` returns the eq-32 tangent point as an ordinary `FitResult`, so `to_json`/`plot`/`metrics_table` all work. `refs` defaults per mode: water `(2%, 0.7)`, small alcohols `(2%, 1.5)`, C5+ alcohols `(2%, 3.0)` for `("vle", "sft")`; `(2%, 2%)` for `("psat", "rho")` — and under that pair, with equal refs, the tangent has slope −1, so eq 32 reduces to `(AARD_psat + AARD_rho) / 2`, exactly AAD_vle, and the selected point is just the front's best *pooled* bulk fit rather than a genuine trade-off pick; set `refs` from the two properties' expected error scales if the selection is meant to mean more than that. **The selected *point* is stable; the selected *parameters* are not.** On the union of every water front measured here, 47 points lie within 0.05 of the eq-32 optimum, spanning AAD_vle 1.48–2.68, m 1.17–1.84 and κ_ab 0.065–0.189 — a 57% swing in m for a cost difference inside run-to-run noise. Do not read one run's `.select()` output as definitive parameters.
  - Infeasible parameter sets report a **graded** violation (`0.5 - worst_valid_fraction`), not a flat penalty. About 80% of the default bounds box has no stable interface for an associating fluid; collapsing all of it onto one `(1e6, 1e6)` leaves the search unable to tell "failed two of 25 gamma points" from "no VLE at all". It is **not** a pymoo constraint: MOEA/D asserts `not problem.has_constraints()` in its `_setup`, so `_penalize` folds the violation into both objectives instead, and `_front_from` does the feasibility filtering the constraint used to do. The same function divides the objectives by `refs` — Tchebicheff is applied to raw values with no nadir normalization, so unscaled most weight vectors collapse onto one corner.
  - **`_capped_replacement` is load-bearing, and pymoo does not have it.** MOEA/D-DE caps how many neighbouring subproblems one offspring may take over (Li & Zhang's `nr`; pagmo's `limit`, default 2 — the configuration Rehner & Gross ran). pymoo's `MOEAD._replace` has **no cap**: it assigns the offspring to every neighbour it beats, up to all 20. Diversity collapses and the front contracts. Measured on water at 9600 evals, AAD_sft extent was 0.30 and 0.13 mN/m without the cap, 0.73 with it, reaching 0.64 — evidence internal to this data set, which is the point. (An earlier version added "past NSGA-II's 0.78"; that 0.78 came from the other-hardware table, and a same-machine NSGA-II run reaches 0.45, so the comparison ran the wrong way round.) Selection among improved slots is random, as the paper specifies; first-n or greedy puts the bias straight back.
  - **Do not validate against Table 1's parameters, even though they look close.** `.select()` returns `m=1.1346 σ=2.8109 ε/k=273.25 κ_ab=0.047221 ε_ab/k=3054.6` against the paper's `1.0000 / 2.9375 / 272.03 / 0.044480 / 3125.3` — 0.5% on ε/k, 2–6% on σ/κ_ab/ε_ab/k, 13% on m. It is a coincidence of where one run's tangent landed, and the restarted front settles it: that front *does* reach the paper's operating point (AAD_vle 1.42–8.57, so 2.14% is inside it), and the point sitting there is `m=2.0331 σ=2.2570 ε/k=230.94 κ_ab=0.26527 ε_ab/k=2449.4` — **κ_ab off by a factor of six, m by a factor of two**, at AAD_sft 1.75 vs the paper's 1.59. Same bulk error, a different fluid. Matching AAD_vle does not resolve the degeneracy this whole two-objective exercise exists to expose. The data differs too (bundled quasi-data is saturation-only over 280–620 K; the paper fitted densities to 1073 K), so agreement was never expected. The cap's evidence is the AAD_sft extent, which is internal to this data set.
  - **Scaling ranking, all with the cap, same budget** (`points | AAD_sft range | max gap | eq 32`): observed spans `131 | 0.64–1.36 | 0.225 | 2.961`; eq-32 refs `105 | 1.28–2.22 | 0.278 | 3.082`; raw `71 | 1.68–2.71 | 0.532 | 4.988`. Raw is worst by a wide margin, so **do not "simplify" the scaling away on the grounds that pagmo runs raw objectives.** Single stochastic samples — 0.1 in eq 32 is not signal.
  - **`n_restarts` is the biggest knob, because the water front spans two disconnected parameter basins** — m ≈ 1.83 with κ_ab ≈ 0.19 below AAD_vle 2.2, and the paper's m ≈ 1.1 with κ_ab ≈ 0.04 above 2.5. A single search commits to one basin in its first generations and then slides along it; **no single run has ever covered both**, which is also what the "genuine hole" at the knee is. `n_restarts` reruns with a stepped seed and unions the fronts (`_merge_fronts`), and `_densify` runs once on the merge, not per restart. At *equal* 9600-eval budget, 4×(80×30) dominates 58% of 1×(80×120) while conceding 19%, eq 32 2.781 vs 2.961; 4×(80×120) reaches eq 32 2.685 and dominates 85% of NSGA-II. Cost is linear in `n_restarts` — prefer it over `n_gen` when the front is patchy, `n_gen` when it is smooth but short.
  - **Compare fronts with `coverage`, never by extent.** `coverage(A, B)` is Zitzler's C-metric, the fraction of B dominated by A; asymmetric, so report it both ways. This bullet previously called MOEA/D and NSGA-II "comparable" because their AAD_sft spans matched to 0.008 — **that reading was wrong.** Fronts of equal span can sit entirely behind one another: NSGA-II dominates **63%** of the shipped single-run MOEA/D front and concedes 25%, and three of four older MOEA/D configurations had comparable extents while being 100% dominated. The case for MOEA/D remains paper fidelity (Rehner & Gross's reasons — derivative-free, initial-value independent, infeasible sets suppressed by a large residual — hold for NSGA-II too); on a single run NSGA-II is measurably ahead, and what closes the gap is `n_restarts`, not the solver. **The 301 s / AAD_vle 1.44 NSGA-II figures in `pareto.py`'s first table were recorded on other hardware — do not compare new runs against them.**
  - **No configuration covers the front alone.** Union of the four measured runs: 232 points where the best single run has 210, with the single-run MOEA/D and NSGA-II fronts still contributing 43 and 37 points nothing else found. Run more than one search before trusting a front's ends.
  - **`hvap_path` is reported but never optimized.** It loads into `data` and appears in the metrics of whatever `.select()` returns — a useful cross-check on a fitted point — but eq 30 is psat and rho only, and `_evaluate_point` never reads `data.T_hvap`. Passing it will not pull the front towards better hvap. Pinned by `test_hvap_is_reported_but_never_optimized`. Relatedly, the `psat_weight`/`density_weight`/`hvap_weight` arguments at the `_setup_pure_fit` call are **inert** — they only shape `cost_fn`, which the pareto driver discards — and are passed neutral (1.0) rather than the 3.0/2.0/1.0 copied from `fit_pure`, which made it look as though psat was weighted 3× in the front. Surface tension sits in a related but not identical position under `objectives=("psat", "rho")`: passed and loaded the same way, and never entering the search — `_evaluate_point` skips `_build_functional` on every one of the search's evaluations, which is most of why the pair is cheap. Unlike `hvap_path`, though, it is not DFT-free everywhere: whenever `sft_path` was given, `.select()` still builds a functional and DFT-solves the one point it returns, so the resulting `FitResult` reports `aad_sft` regardless of `objectives` — a single solve, a rounding error against the evaluations the search itself skipped.
  - `n_jobs=-1` (default) evaluates the population across worker processes — every core bar two. **Processes, not threads: feos does not release the GIL** (measured 0.89x on a 16-thread pool, i.e. slower than serial). `feos.Identifier` is also unpicklable, so pymoo's `StarmapParallelization` cannot be used — the problem is a vectorized `Problem` that maps the population itself, and workers rebuild their own feos objects in an initializer. The pool uses **spawn**, since forking a process with feos' live rayon threads risks a child deadlock, and calls `feos.set_num_threads(1)` per worker to avoid oversubscription. Measured 4.3x on 14 workers with identical results.
  - The front is re-evaluated once after the search (at most `pop_size` points, under 2% of a real budget): `res.F` comes back in scaled, penalized space and a penalized row's true objectives cannot be recovered from it.
  - `quiet_solver=True` (default) silences fd-level stderr during the search; feos panics from Rust worker threads on infeasible parameter sets and Python cannot intercept those writes.
  - `refine=4` (default) runs a post-search densification pass (`_densify`): interpolate parameter vectors between adjacent front points, re-evaluate, keep whatever is non-dominated. Under NSGA-II the raw output was a population, not a curve — crowding distance preserves diversity but never forces even spacing, so it came back as clusters separated by voids (measured: 8 of 80 water points within 0.2% of each other on the AAD_vle axis, and a 2.8% stretch holding none). MOEA/D removes that: measured on the same 9600-eval water run, 190 points at median normalized spacing 0.0084 and largest gap 0.093. Every point interpolated across a void is itself non-dominated, so the voids are a sampling artefact. Interpolants are allocated in proportion to segment length in normalized objective space, not evenly — spreading them evenly wastes them inside the clusters. Measured on the 9600-eval water run: 80 → 209 points, median spacing 3.4× finer, 316 evaluations, 7 s.
  - **The raw front is not always a front.** On water the refine pass dominated a whole stretch of it, moving the eq-32 tangent point from `(1.44%, 1.62)` to `(1.90%, 1.29)` — eq 32 from 3.04 to 2.79. Interpolation only reveals this; the dominated points were in the raw search output all along. Trust a `refine=0` front's `.select()` less.
  - Refinement does **not** close a genuine hole, where the straight line between two front points in parameter space does not track the front in objective space. Water keeps one at the knee near AAD_vle = 2%: a second pass moves eq 32 by 0.01 for 3.6× the evaluations. **That hole is the basin boundary** (see the `n_restarts` bullet) — interpolating across it walks through neither basin, which is exactly why arithmetic cannot close it. It needs a run that lands on the other side, i.e. `n_restarts`, not more `refine`.
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
