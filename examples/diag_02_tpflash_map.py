"""Map tp_flash successes and failures across all temperatures in the 1-hexanol+water CSV.

Uses water_gross2002 + 1-hexanol (Gross & Sadowski 2001, 2B) with k_ij=0.0.
Tests all 51 sigmoid-spaced feeds plus additional targeted feeds.
"""
import numpy as np
import feos
import si_units as si
import json
from pathlib import Path

# ---- parameters ----
PARAMS_DIR = Path("examples/data/parameters")
WATER_MODELS_JSON = PARAMS_DIR / "water_models.json"
HEXANOL_JSON = PARAMS_DIR / "pcsaft.json"  # standard params, check if it exists

# We'll build records manually
water_record_dict = {
    "identifier": {"cas": "7732-18-5", "name": "water"},
    "molarweight": 18.015,
    "m": 1.0656,
    "sigma": 3.0007,
    "epsilon_k": 366.51,
    "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.034868, "epsilon_k_ab": 2500.7}]
}

hexanol_record_dict = {
    "identifier": {"cas": "111-27-3", "name": "1-hexanol"},
    "molarweight": 102.175,
    "m": 3.3312,
    "sigma": 3.7483,
    "epsilon_k": 270.86,
    "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.002566, "epsilon_k_ab": 2778.9}]
}

import tempfile, os

def build_eos(kij=0.0):
    combined = [water_record_dict, hexanol_record_dict]
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(combined, tmp)
    tmp.close()
    params = feos.Parameters.from_json(
        ["water", "1-hexanol"],
        pure_path=tmp.name
    )
    os.unlink(tmp.name)
    binary_params = feos.Parameters.new_binary(params.pure_records, k_ij=kij)
    return feos.EquationOfState.pcsaft(binary_params, max_iter_cross_assoc=100)

# Build sigmoid-spaced feeds (same as lle.py)
_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
LLE_FEEDS = (1.0 / (1.0 + np.exp(_s))).tolist()

# Unique temperatures from CSV (K)
T_DATA = sorted(set([
    273.2, 278.7, 280.0, 281.8, 283.2, 283.4, 284.2, 285.2, 286.1, 287.8,
    288.2, 290.2, 293.2, 293.9, 296.1, 296.2, 298.2, 300.1, 302.1, 302.9,
    303.1, 303.2, 304.1, 306.2, 308.0, 308.2, 313.0, 313.1, 313.2, 318.0,
    323.2, 333.2, 343.2, 343.5, 353.2, 353.5, 363.2, 363.5, 373.2, 383.2,
    393.2, 403.2, 413.2, 423.2, 433.2, 443.2, 453.2, 463.2, 473.2, 483.2,
    493.2
]))

print(f"Testing {len(T_DATA)} unique temperatures with k_ij=0.0")
print(f"Number of feeds tested per temperature: {len(LLE_FEEDS)}")
print()

eos = build_eos(kij=0.0)
pressure = 1.0 * si.BAR

successes = []
failures = []

for T_K in T_DATA:
    found = False
    best = None
    n_tried = 0
    for z1 in LLE_FEEDS:
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            pe = feos.PhaseEquilibrium.tp_flash(eos, T_K * si.KELVIN, pressure, feed)
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            if abs(x_a - x_b) > 1e-4:
                best = (min(x_a, x_b), max(x_a, x_b), z1, n_tried + 1)
                found = True
                break
        except Exception as e:
            pass
        n_tried += 1
    if found:
        successes.append((T_K, best[0], best[1], best[2], best[3]))
    else:
        failures.append(T_K)

print(f"SUCCESSES ({len(successes)}/{len(T_DATA)}):")
print(f"{'T (K)':>8} {'x1_I (pred)':>12} {'x1_II (pred)':>13} {'feed z1':>10} {'tries':>6}")
for T_K, xi, xii, z1, tries in successes:
    print(f"{T_K:8.1f} {xi:12.6f} {xii:13.6f} {z1:10.4f} {tries:6d}")

print(f"\nFAILURES ({len(failures)}/{len(T_DATA)}):")
for T_K in failures:
    print(f"  T = {T_K:.1f} K")

print(f"\nSummary: {len(successes)} succeeded, {len(failures)} failed out of {len(T_DATA)}")
