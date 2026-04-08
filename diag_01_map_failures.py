"""Diagnostic 1: Map tp_flash successes/failures across all temperatures.

Uses water_gross2002 + 1-hexanol with kij=0.0.
Tries all 51 sigmoid-spaced feeds at each temperature.
"""

import json
import sys
import numpy as np
import pandas as pd
import feos
import si_units as si

# ── Parameters ────────────────────────────────────────────────────────────────
WATER_PARAMS_PATH = "examples/data/parameters/water_models.json"
HEXANOL_PARAMS_PATH = "examples/data/parameters/pcsaft_gross_sadowski2001.json"
CSV_PATH = "examples/data/lle/1-hexanol_water.csv"
KIJ = 0.0
PRESSURE = 1.0 * si.BAR

# ── Check if we have separate hexanol params or need to build inline ──────────
import os
if not os.path.exists(HEXANOL_PARAMS_PATH):
    # Build hexanol record inline
    hexanol_rec = {
        "identifier": {"name": "1-hexanol"},
        "molarweight": 102.177,
        "m": 3.3312,
        "sigma": 3.7483,
        "epsilon_k": 270.86,
        "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.002566, "epsilon_k_ab": 2778.9}],
    }
    import tempfile
    hexanol_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump([hexanol_rec], hexanol_tmp)
    hexanol_tmp.close()
    HEXANOL_PARAMS_PATH = hexanol_tmp.name
    print(f"  Created temp hexanol params at {HEXANOL_PARAMS_PATH}")

# ── Build records ─────────────────────────────────────────────────────────────
def build_eos(kij):
    water_models = json.load(open(WATER_PARAMS_PATH))
    water_rec = next(r for r in water_models if r["identifier"]["name"] == "water_gross2002")

    hexanol_rec = {
        "identifier": {"name": "1-hexanol"},
        "molarweight": 102.177,
        "m": 3.3312,
        "sigma": 3.7483,
        "epsilon_k": 270.86,
        "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.002566, "epsilon_k_ab": 2778.9}],
    }

    import tempfile
    combined = [water_rec, hexanol_rec]
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(combined, tmp)
    tmp.close()

    params = feos.Parameters.from_json(
        ["water_gross2002", "1-hexanol"],
        tmp.name,
        identifier_option=feos.IdentifierOption.Name,
    )
    os.unlink(tmp.name)

    params_binary = feos.Parameters.new_binary(params.pure_records, k_ij=kij)
    return feos.EquationOfState.pcsaft(params_binary, max_iter_cross_assoc=100)

# ── Probe feos API ─────────────────────────────────────────────────────────────
print("=== PhaseEquilibrium methods ===")
print([m for m in dir(feos.PhaseEquilibrium) if not m.startswith("_")])
print("\n=== PhaseDiagram methods ===")
print([m for m in dir(feos.PhaseDiagram) if not m.startswith("_")])
print()

# Try to build EOS
print("=== Building EOS ===")
try:
    eos = build_eos(KIJ)
    print(f"  EOS built OK")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Sigmoid-spaced feeds (same as lle.py) ────────────────────────────────────
_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
LLE_FEEDS = (1.0 / (1.0 + np.exp(_s))).tolist()
print(f"Feed range: {LLE_FEEDS[0]:.6f} to {LLE_FEEDS[-1]:.6f}, n={len(LLE_FEEDS)}")

# ── Load CSV data ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"\nData: {len(df)} rows, T from {df['temperature_K'].min():.1f} to {df['temperature_K'].max():.1f} K\n")

# ── Aggregate by unique T ─────────────────────────────────────────────────────
unique_T = sorted(df["temperature_K"].dropna().unique())

print(f"{'T[K]':>8}  {'x1_I_exp':>10}  {'x1_II_exp':>10}  {'pred_I':>8}  {'pred_II':>8}  {'nfeed_ok':>8}  {'status':>6}")
print("-" * 80)

successes = []
failures = []
all_errors = {}

for T_K in unique_T:
    mask = np.abs(df["temperature_K"] - T_K) < 0.01
    sub = df[mask]

    x1_I_vals = sub["x1_I"].dropna().values if "x1_I" in sub.columns else np.array([])
    x1_II_vals = sub["x1_II"].dropna().values if "x1_II" in sub.columns else np.array([])
    exp_I = float(np.mean(x1_I_vals)) if len(x1_I_vals) > 0 else None
    exp_II = float(np.mean(x1_II_vals)) if len(x1_II_vals) > 0 else None

    best_result = None
    n_ok = 0
    errors = []

    for z1 in LLE_FEEDS:
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            pe = feos.PhaseEquilibrium.tp_flash(eos, T_K * si.KELVIN, PRESSURE, feed)
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            if abs(x_a - x_b) < 1e-4:
                continue
            n_ok += 1
            pred_I, pred_II = min(x_a, x_b), max(x_a, x_b)
            if best_result is None:
                best_result = (pred_I, pred_II)
        except Exception as e:
            errors.append(str(e))

    exp_I_str = f"{exp_I:.6f}" if exp_I is not None else "         --"
    exp_II_str = f"{exp_II:.6f}" if exp_II is not None else "         --"

    if best_result is not None:
        pred_I, pred_II = best_result
        print(f"{T_K:8.2f}  {exp_I_str:>10}  {exp_II_str:>10}  {pred_I:8.5f}  {pred_II:8.5f}  {n_ok:8d}  OK")
        successes.append(T_K)
    else:
        unique_errs = list(set(errors[:3]))
        err_str = unique_errs[0][:40] if unique_errs else "unknown"
        print(f"{T_K:8.2f}  {exp_I_str:>10}  {exp_II_str:>10}  {'--':>8}  {'--':>8}  {0:8d}  FAIL  [{err_str}]")
        failures.append(T_K)
        all_errors[T_K] = unique_errs

print(f"\n=== SUMMARY ===")
print(f"Total unique temperatures: {len(unique_T)}")
print(f"Successes: {len(successes)} ({100*len(successes)/len(unique_T):.1f}%)")
print(f"Failures:  {len(failures)} ({100*len(failures)/len(unique_T):.1f}%)")
if failures:
    print(f"Failure T range: {min(failures):.1f} – {max(failures):.1f} K")
    print("\nUnique error messages from failures:")
    all_err_msgs = set()
    for errs in all_errors.values():
        all_err_msgs.update(errs)
    for e in sorted(all_err_msgs):
        print(f"  {e}")
