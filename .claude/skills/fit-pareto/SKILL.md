---
name: fit-pareto
description: Use when running fit_pure_pareto for a pure compound — choosing objectives and refs, budgeting n_restarts vs n_gen, comparing fronts with coverage, reading refine/polish output, and the surface-tension / heat-capacity CSV formats.
---

# Pareto fitting with `fit_pure_pareto`

Operator-facing knobs, moved from the root `CLAUDE.md`. The "do not" rules (Table 1, extent vs coverage, one run's `.select()`, scaling, other-hardware figures) stay in the root file; the engine rationale is in `src/fit_pcsaft/_pure/CLAUDE.md`.

## Selecting a point: `refs`

  - `ParetoResult.select(refs)` returns the eq-32 tangent point as an ordinary `FitResult`, so `to_json`/`plot`/`metrics_table` all work. `refs` defaults to `_DEFAULT_REFS[objectives]`: `(2%, 0.7)` for `("vle", "sft")`, `(2%, 2%)` for `("psat", "rho")`, `(2%, 2%, 2%)` for the cp triple, `(2%, 2%, 0.7)` for `("psat", "rho", "sft")`, `(2%, 2%, 2%, 0.7)` for the quadruple. 0.7 mN/m is Rehner & Gross's water value; their small-alcohol `(2%, 1.5)` and C5+-alcohol `(2%, 3.0)` are not defaults and must be passed. Under `("psat", "rho")`, with equal refs, the tangent has slope −1, so eq 32 reduces to `(AARD_psat + AARD_rho) / 2`, exactly AAD_vle, and the selected point is just the front's best *pooled* bulk fit rather than a genuine trade-off pick; set `refs` from the two properties' expected error scales if the selection is meant to mean more than that. **The selected *point* is stable; the selected *parameters* are not.** On the union of every water front measured here, 47 points lie within 0.05 of the eq-32 optimum, spanning AAD_vle 1.48–2.68, m 1.17–1.84 and κ_ab 0.065–0.189 — a 57% swing in m for a cost difference inside run-to-run noise.

## `hvap_path`, `cp_path`, `sft_path` under the different tuples

  - **`hvap_path` is reported but never optimized.** Pinned by `test_hvap_is_reported_but_never_optimized`. `cp_path` behaves the same way under every tuple without `"cp"`; under the cp triple and the quadruple it is an objective. Relatedly, the `psat_weight`/`density_weight`/`hvap_weight` arguments at the `_setup_pure_fit` call are **inert** — they only shape `cost_fn`, which the pareto driver discards — and are passed neutral (1.0) rather than the 3.0/2.0/1.0 copied from `fit_pure`, which made it look as though psat was weighted 3× in the front.

## Budget: `n_restarts` vs `n_gen`

  - **`n_restarts` is the biggest knob, because the water front spans two disconnected parameter basins** (basin argument: `_merge_fronts` docstring; restarts-beat-depth numbers: `fit_pure_pareto` docstring). Cost is linear in `n_restarts` — prefer it over `n_gen` when the front is patchy, `n_gen` when it is smooth but short.

## Comparing fronts

  - **Compare fronts with `coverage`, never by extent.** `coverage(A, B)` is Zitzler's C-metric, the fraction of B dominated by A; asymmetric, so report it both ways.
  - **No configuration covers the front alone.**

## `refine` and `polish`

  - `refine=4` (default) runs a post-search densification pass (`_densify`), and a second one after `polish` — polished points dominate most raw neighbours, so polish alone can leave a few anchors (2-phenylethanol: 51 → 6). The front is re-sorted by `F[:, 0]` afterwards because `polish_front` returns rows in input order.
  - `polish=True` solves each front point in unit-box coordinates `u = (θ − lo)/(hi − lo)`, and under `("psat", "rho")` on exact feos AD gradients (`_evaluate_with_grad`: the same two `*_derivatives` calls as the batched search, with `parameter_names=param_names(spec)`; 0.28 ms against 0.25 ms for the plain row). Measured on thymol's 170-point production front at `max_iter=50` (2026-09-05): finite differences in raw units 20.6 s, ~460 evaluations/point, 65 % of solves at `maxiter`, summed polished objective 356, eq-32 tangent 0.3992; unit box alone 22 s / Σ 139 / 0.3956; AD + unit box **11–12 s / Σ 103 / 0.3955**. Normalisation is the quality half, AD the speed half. What remains is SLSQP's line search (nfev ≈ 5/iteration even with `jac` — AARD has a kink at every data point), so `max_iter` is a fidelity knob: 150 iterations cost what 50 finite-difference ones did and reached Σ 59. The DFT pairs and the cp triple have no AD for their third objective and stay finite-difference (still in the unit box); SLSQP's perturbed rows go to `evaluate` as one batch (`_fd_map` via `options["workers"]`), so the worker pool solves them side by side — measured (2026-10-12) 87 s → 40 s polish on a small water `("vle","sft")` front, identical F. Parallelising across points is not on the table: feos holds the GIL, `feos.Identifier` does not pickle, and each point warm-starts from its predecessor. The verbose `polish:` line prints its seconds.
  - **The raw front is not always a front.** On water the refine pass dominated a whole stretch of it, moving the eq-32 tangent point from `(1.44%, 1.62)` to `(1.90%, 1.29)` — eq 32 from 3.04 to 2.79. Interpolation only reveals this; the dominated points were in the raw search output all along. Trust a `refine=0` front's `.select()` less.
  - Refinement does **not** close a genuine hole, where the straight line between two front points in parameter space does not track the front in objective space. Water keeps one at the knee near AAD_vle = 2%: a second pass moves eq 32 by 0.01 for 3.6× the evaluations. **That hole is the basin boundary** (see the `n_restarts` bullet) — interpolating across it walks through neither basin, which is exactly why arithmetic cannot close it. It needs a run that lands on the other side, i.e. `n_restarts`, not more `refine`.

## CSV formats

- **Surface tension CSV format**: columns `T`, `sft` (aliases `gamma`, `surface_tension`, `sigma_st`, `st`). Default unit mN/m. `FitResult.aad_sft` is the paper's AAD_sft; `ard_sft` is the relative AARD.
- **Heat capacity CSV format** (`fit_pure_pareto` only): columns `T`, `cp` (aliases `heat_capacity`, `c_p`, `Cp`, `cp_liq`, `cp_J_mol_K`), optional `P` (default kPa via `pressure_unit`); a missing `P` column or a blank cell means saturation pressure. `cp` is the **total** liquid cp, default J/(mol·K) (`heat_capacity_unit`). Always paired with `cp_ig=[A, B, C, D, E]`, DIPPR-107 in J/(kmol·K). `FitResult.ard_cp` is the AARD of the total cp.
