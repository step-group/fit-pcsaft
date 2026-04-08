"""Test feos.PhaseDiagram.lle continuation method on 1-hexanol + water.

Compares coverage against tp_flash, and investigates the failing high-T region.
"""
import numpy as np
import feos
import si_units as si
import json, tempfile, os
from pathlib import Path

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

def build_eos(kij=0.0):
    combined = [water_record_dict, hexanol_record_dict]
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(combined, tmp)
    tmp.close()
    params = feos.Parameters.from_json(["water", "1-hexanol"], pure_path=tmp.name)
    os.unlink(tmp.name)
    binary_params = feos.Parameters.new_binary(params.pure_records, k_ij=kij)
    return feos.EquationOfState.pcsaft(binary_params, max_iter_cross_assoc=100)

eos = build_eos(kij=0.0)
pressure = 1.0 * si.BAR

# Experimental data from CSV (unique T and compositions for high-T region)
exp_data = {
    433.2: (None, 0.4237),
    443.2: (0.005, 0.3917),
    453.2: (0.0066, 0.3534),
    463.2: (0.009, 0.3099),
    473.2: (0.0128, 0.2602),
    483.2: (0.0189, 0.2008),
    493.2: (0.0342, 0.1245),
}

print("=" * 70)
print("TEST 1: PhaseDiagram.lle (pressure-based, sweeping temperature)")
print("=" * 70)
print()

# The API: lle(eos, temperature_or_pressure, feed, min_tp, max_tp, npoints)
# For temperature sweep at fixed pressure: pass pressure as temperature_or_pressure
# For pressure sweep at fixed temperature: pass temperature as temperature_or_pressure

# Try with a feed composition in the two-phase region
# Experimental: at 443.2K, x1_I~0.005, x1_II~0.39 → feed ~0.2 is in the two-phase gap
feeds_to_try = [0.20, 0.15, 0.10, 0.30, 0.05, 0.40]

for feed_z1 in feeds_to_try:
    try:
        feed = np.array([feed_z1, 1.0 - feed_z1]) * si.MOL
        pd = feos.PhaseDiagram.lle(
            eos, pressure, feed,
            min_tp=270.0 * si.KELVIN,
            max_tp=500.0 * si.KELVIN,
            npoints=200
        )
        T_arr = pd.liquid.temperature / si.KELVIN
        x1_I = pd.liquid.molefracs[:, 0]
        x1_II = pd.vapor.molefracs[:, 0]
        print(f"feed z1={feed_z1:.3f}: PhaseDiagram.lle SUCCESS, {len(T_arr)} points")
        print(f"  T range: {T_arr.min():.1f}K to {T_arr.max():.1f}K")
        print(f"  x1_I range: {x1_I.min():.6f} to {x1_I.max():.6f}")
        print(f"  x1_II range: {x1_II.min():.6f} to {x1_II.max():.6f}")
        # Show points in the failing region (T >= 430)
        mask = T_arr >= 430.0
        if mask.sum() > 0:
            print(f"  Points with T >= 430K ({mask.sum()} pts):")
            for i in np.where(mask)[0]:
                print(f"    T={T_arr[i]:.1f}K  x1_I={x1_I[i]:.5f}  x1_II={x1_II[i]:.5f}")
        break
    except Exception as e:
        print(f"feed z1={feed_z1:.3f}: FAILED: {e}")

print()
print("=" * 70)
print("TEST 2: More targeted feeds near experimental compositions for failing T")
print("=" * 70)

_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
SIGMOID_FEEDS = (1.0 / (1.0 + np.exp(_s))).tolist()

# Experimental compositions suggest very asymmetric phases near UCST
# x1_I is small (0.005-0.034), x1_II is ~0.12-0.42
# Feed needs to be in the two-phase region
for T_K in sorted(exp_data.keys()):
    exp_I, exp_II = exp_data[T_K]
    # Try feeds between the two experimental compositions
    targeted_feeds = []
    if exp_I is not None and exp_II is not None:
        for frac in [0.3, 0.5, 0.7]:
            targeted_feeds.append(exp_I + frac * (exp_II - exp_I))
    if exp_II is not None:
        targeted_feeds.extend([exp_II * 0.5, exp_II * 0.8, exp_II * 0.9])
    # Also try uniform spacing in the gap
    targeted_feeds.extend([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    targeted_feeds.extend(SIGMOID_FEEDS)

    found = False
    for z1 in targeted_feeds:
        z1 = float(np.clip(z1, 1e-4, 1-1e-4))
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            pe = feos.PhaseEquilibrium.tp_flash(eos, T_K * si.KELVIN, pressure, feed)
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            if abs(x_a - x_b) > 1e-4:
                pred_I, pred_II = min(x_a, x_b), max(x_a, x_b)
                print(f"T={T_K:.1f}K SUCCESS: feed={z1:.4f} → x1_I={pred_I:.5f}, x1_II={pred_II:.5f}  (exp: {exp_I}, {exp_II})")
                found = True
                break
        except Exception:
            pass
    if not found:
        print(f"T={T_K:.1f}K STILL FAILED with all feeds")

print()
print("=" * 70)
print("TEST 3: Check stability at failing temperatures")
print("=" * 70)
print("Using feos.State.stability_analysis and is_stable")

for T_K in [433.2, 453.2, 493.2]:
    exp_I, exp_II = exp_data[T_K]
    # Try a feed in the middle of the two-phase region
    z1_mid = 0.20
    feed = np.array([z1_mid, 1.0 - z1_mid]) * si.MOL
    try:
        state = feos.State(eos, T_K * si.KELVIN, pressure=pressure, moles=feed)
        stable = state.is_stable()
        print(f"T={T_K:.1f}K, z1={z1_mid}: is_stable={stable}")
    except Exception as e:
        print(f"T={T_K:.1f}K, z1={z1_mid}: State creation failed: {e}")

    # Also try near exp compositions
    for z1 in [0.01, 0.05, 0.10, 0.20, 0.30]:
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            state = feos.State(eos, T_K * si.KELVIN, pressure=pressure, moles=feed)
            stable = state.is_stable()
            print(f"  z1={z1:.3f}: is_stable={stable}")
        except Exception as e:
            print(f"  z1={z1:.3f}: State failed: {e}")
