# CLAUDE.md

## Setup

Use `uv` for all package management — not `pip`.

`pyproject.toml`'s `[tool.uv.sources]` pins `feos` to a locally-compiled wheel at
`../feos/dist/feos-0.10.3-cp310-abi3-macosx_11_0_arm64.whl` (full LTO,
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

`examples/pure/11_water_pareto.py` takes ~4 min; `examples/pure/13_water_cp_pareto.py` (three objectives) runs in seconds.

## Pure component fitting (`src/fit_pcsaft/_pure/`)

- **`fit.py`**: With `sft_path`, `fit_pure` maps its starts over the pareto spawn pool (`n_jobs=-1`, identical result; water sft 28 → 15 s, 2026-10-12) — so a top-level `fit_pure(sft_path=...)` needs an `if __name__ == "__main__"` guard. Not with `q != 0` alone: single-threaded workers lose feos's own rayon threads (toluene 25 → 70 s). With `q == 0` and no sft, `fit_pure_de` evaluates the whole population in one feos `*_derivatives` call per property (`_make_de_population_objective`, row-for-row equal to `cost_fn`; `vectorized=True` forces scipy's `updating="deferred"`, a different DE variant, so same-seed results differ from before — pass `de_kwargs={"vectorized": False, "updating": "immediate"}` to reproduce them). `scipy_result.nfev` counts candidates, not vectorized calls. Results carry `sft_options`, so metrics, `residuals()` and `to_csv` use the fit's DFT grid.
- **`jacobian.py`**: Analytical Jacobian via feos AD (`_make_f_and_df`). Used unless `sft_path` is given or `q != 0`; otherwise falls back to numerical 2-point diff (`_make_f_and_df_numerical` in `_fit_utils.py`). Surface tension has no feos AD path at all, so `_make_f_and_df` raises if `sft` data is present rather than silently returning a short residual vector.
- `param_names`/`params_dict`/`ad_model`/`ad_rows` in `_fit_utils.py` are the single source of the optimiser-vector and AD-row layout — read by `_build_record`, `jacobian.py`, `eval_pure`, `ParetoResult` and `_evaluate_population`.
- **Viscosity CSV format**: Columns `T`, `P`, `eta` (pressure optional; if absent, P_sat is used for liquid). An optional `phase` string column (`'liquid'`/`'vapor'`) guides the EOS density root selection.

### Parameter transformation

The optimizer works on **sqrt-transformed parameters** (`x_internal = sqrt(params)`), so all parameters remain positive without explicit bounds. The cost function squares them internally (`params = x**2`). This applies to `fit_pure` / `fit_pure_de` only; binary fitting uses bounds directly.

- mu is **never initialized at 0.0** (dipole Jacobian is identically zero there)

### Surface tension and pareto fitting

Engine rationale and measurements: `src/fit_pcsaft/_pure/CLAUDE.md`. Operator workflow (`objectives`, `refs`, `n_restarts`, `coverage`, `refine`/`polish`, sft/cp CSV formats): `.claude/skills/fit-pareto/SKILL.md`. The cp triple: `.claude/skills/fit-compound/SKILL.md`.

Rules that hold regardless of where the argument lives:

- `("vle", "sft")` was last measured in 2026-08 at 103.33 ms/evaluation and returned 15 front points against the bulk pair's 2 over the same 600-evaluation budget; **do not** read that as a front-quality comparison — 600 evaluations at `refine=0`/`n_restarts=1` is far too small a budget for the sft mode, and the gap is a budget artefact, not a resolution one.
- Camphor's DE/SBX tie is unexplained; two explanations were falsified (see `_pure/CLAUDE.md`). Do not invent a third story for it.
- `variant="de"` is the default and **moves committed downstream parameters**: any caller that does not pass `variant=` now runs DE with no diff at the call site — TESIS's `s04_fit_pareto.py` included. Pass `variant="sbx"` to reproduce a pre-2026-08 fit. Every benchmark under `benchmarks/` pins its operator explicitly (`test_benchmark_sbx_arms_pin_the_operator_explicitly`).
- Eq 30 is a mean, not a sum — argument in `pareto.py`'s module docstring; `tests/test_pareto.py` guards it with a two-sided ratio band whose upper edge catches a revert to the sum.
- Infeasible parameter sets report a **graded** violation folded into both objectives by `_penalize`, not a pymoo constraint — see its docstring.
- `_capped_replacement` is load-bearing and pymoo does not have it — see its docstring.
- **Do not validate against Table 1's parameters, even though they look close** — see the "Do not lean on it" paragraph of `fit_pure_pareto`'s docstring.
- Do not "simplify" the scaling away on the grounds that pagmo runs raw objectives.
- **The selected *point* is stable; the selected *parameters* are not.** Do not read one run's `.select()` output as definitive parameters.
- **Compare fronts with `coverage`, never by extent.** **The 301 s / AAD_vle 1.44 NSGA-II figures in `pareto.py`'s first table were recorded on other hardware — do not compare new runs against them.**
- Run more than one search before trusting a front's ends.
- **Fitting near the critical point**: classical PC-SAFT has no critical scaling. Bulk data above about Tr = 0.96 carries the model's structural error rather than parameter error and will dominate a relative objective — the bundled water quasi-data stops at 620 K for psat/rho (Tr = 0.958) while γ runs to 640 K.

### Binary k_ij fitting (`src/fit_pcsaft/_binary/`)

See `src/fit_pcsaft/_binary/CLAUDE.md`.

### Data format

CSVs **require a header**. Column names are normalised through `_COL_ALIASES` in `_csv.py` and validated against a `CsvSchema`; unrecognised columns are dropped silently. Aliases carrying unit suffixes (`psat_kPa`, `rho_kg_m3`) are name hints only — **no numeric conversion happens in the loaders**. Units are carried entirely by the `*_unit` keyword arguments into the `Units` dataclass and applied at prediction time.

### Reference (quasi-)data

The water CSVs under `examples/data/{psat,density,heat_capacity,surface_tension}/` are quasi-data; provenance and the regeneration command are in `examples/data/generate_water_reference.py`'s docstring.

**Do not hand-write reference correlations from memory.** feos' multiparameter EOS plus that JSON covers any of the 124 fluids; reach for it instead.
