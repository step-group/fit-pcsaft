"""Final diagnostic: find the optimal strategy for _residuals_at_T.

Summarizes:
1. PhaseDiagram.lle can cover full T range (273-426K) at k_ij=0.1
   but requires the right feed
2. For interpolation onto arbitrary T values, we need a strategy
3. For optimization, we compare: (a) tp_flash with targeted feeds vs
   (b) PhaseDiagram.lle (pre-computed) for the current k_ij
"""
import numpy as np
import feos
import si_units as si
import json, tempfile, os

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

pressure = 1.0 * si.BAR

print("=" * 70)
print("STRATEGY A: PhaseDiagram.lle with multiple feeds (best coverage)")
print("=" * 70)

# Try all 7 feeds systematically at k_ij=0.1
# The goal: find the SINGLE best feed that covers 270-426K
eos_k = build_eos(0.1)

best_result = None
best_T_min = 999.0

feed_scan = np.linspace(0.05, 0.60, 20).tolist()
for feed_z1 in feed_scan:
    try:
        feed = np.array([feed_z1, 1.0 - feed_z1]) * si.MOL
        pd = feos.PhaseDiagram.lle(
            eos_k, pressure, feed,
            min_tp=270.0 * si.KELVIN,
            max_tp=500.0 * si.KELVIN,
            npoints=300
        )
        T = pd.liquid.temperature / si.KELVIN
        if len(T) > 0 and T.min() < best_T_min:
            best_T_min = T.min()
            best_result = (feed_z1, pd, len(T))
    except:
        pass

if best_result:
    feed_z1, pd, npts = best_result
    T = pd.liquid.temperature / si.KELVIN
    x1_I = pd.liquid.molefracs[:, 0]
    x1_II = pd.vapor.molefracs[:, 0]
    print(f"Best feed: z1={feed_z1:.3f}, {npts} points, T=[{T.min():.1f},{T.max():.1f}]K")
    print(f"x1_I: [{x1_I.min():.6f}, {x1_I.max():.6f}]")
    print(f"x1_II: [{x1_II.min():.6f}, {x1_II.max():.6f}]")

print()
print("=" * 70)
print("STRATEGY B: Multiple PhaseDiagram.lle calls, merge results")
print("=" * 70)

# Strategy: call PD.lle with multiple feeds and merge unique T points
eos_k = build_eos(0.1)
all_T = []
all_xI = []
all_xII = []

for feed_z1 in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
    try:
        feed = np.array([feed_z1, 1.0 - feed_z1]) * si.MOL
        pd = feos.PhaseDiagram.lle(
            eos_k, pressure, feed,
            min_tp=270.0 * si.KELVIN,
            max_tp=500.0 * si.KELVIN,
            npoints=300
        )
        T = pd.liquid.temperature / si.KELVIN
        xI = pd.liquid.molefracs[:, 0]
        xII = pd.vapor.molefracs[:, 0]
        all_T.extend(T.tolist())
        all_xI.extend(xI.tolist())
        all_xII.extend(xII.tolist())
        print(f"  feed={feed_z1:.2f}: {len(T)} pts, T=[{T.min():.1f},{T.max():.1f}]K")
    except Exception as e:
        print(f"  feed={feed_z1:.2f}: FAILED ({e})")

if all_T:
    all_T = np.array(all_T)
    all_xI = np.array(all_xI)
    all_xII = np.array(all_xII)
    # Sort by T
    idx = np.argsort(all_T)
    all_T, all_xI, all_xII = all_T[idx], all_xI[idx], all_xII[idx]
    print(f"\nMerged: {len(all_T)} total pts, T=[{all_T.min():.1f},{all_T.max():.1f}]K")

print()
print("=" * 70)
print("STRATEGY C: tp_flash with augmented feeds (very dense near T_exp)")
print("=" * 70)

# For the optimization, the key question is whether tp_flash can reliably find
# LLE when the optimizer is varying k_ij. Test robustness for k_ij in [-0.3, 0.3]

_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
SIGMOID_FEEDS = (1.0 / (1.0 + np.exp(_s))).tolist()

# Near the experimental compositions
exp_feeds_low_T = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
ALL_FEEDS = exp_feeds_low_T + SIGMOID_FEEDS

test_T = [273.2, 298.2, 323.2, 363.2, 383.2, 403.2, 413.2, 423.2]
for kij in [0.0, 0.1, 0.2]:
    eos_k = build_eos(kij)
    n_success = 0
    for T_K in test_T:
        for z1 in ALL_FEEDS:
            z1 = float(np.clip(z1, 1e-4, 1-1e-4))
            try:
                feed = np.array([z1, 1.0 - z1]) * si.MOL
                pe = feos.PhaseEquilibrium.tp_flash(eos_k, T_K * si.KELVIN, pressure, feed)
                x_a = float(pe.liquid.molefracs[0])
                x_b = float(pe.vapor.molefracs[0])
                if abs(x_a - x_b) > 1e-4:
                    n_success += 1
                    break
            except:
                pass
    print(f"k_ij={kij:+.2f}: {n_success}/{len(test_T)} temperatures with tp_flash")

print()
print("=" * 70)
print("STRATEGY D: PhaseDiagram.lle + interpolation for all exp T")
print("=" * 70)
print("This is what the existing code already uses internally for phase diagrams.")
print("The key insight: PD.lle computes the FULL binodal curve in one call,")
print("then we interpolate to each experimental T.")
print()

eos_k = build_eos(0.1)
feed = np.array([0.40, 0.60]) * si.MOL
pd = feos.PhaseDiagram.lle(
    eos_k, pressure, feed,
    min_tp=270.0 * si.KELVIN,
    max_tp=500.0 * si.KELVIN,
    npoints=500
)
T_pd = pd.liquid.temperature / si.KELVIN
xI_pd = pd.liquid.molefracs[:, 0]
xII_pd = pd.vapor.molefracs[:, 0]
print(f"PD.lle (k_ij=0.1, feed=0.4): {len(T_pd)} pts, T=[{T_pd.min():.1f},{T_pd.max():.1f}]K")

# Interpolate to experimental temperatures
exp_T_test = [273.2, 283.2, 298.2, 323.2, 343.2, 363.2, 383.2, 403.2, 413.2, 423.2]
exp_xI = [0.00139, 0.0012, 0.0011, 0.0009, 0.001, 0.0012, 0.0016, 0.0021, None, None]
exp_xII = [0.7172, 0.7205, 0.6944, 0.6898, 0.6464, 0.6018, 0.5582, 0.5098, 0.4836, 0.454]

print(f"\n{'T(K)':>6} {'interp_xI':>10} {'interp_xII':>11} {'exp_xI':>9} {'exp_xII':>10}")
for i, T_K in enumerate(exp_T_test):
    if T_pd.min() <= T_K <= T_pd.max():
        xI_interp = float(np.interp(T_K, T_pd, xI_pd))
        xII_interp = float(np.interp(T_K, T_pd, xII_pd))
        print(f"{T_K:6.1f} {xI_interp:10.6f} {xII_interp:11.6f} {str(exp_xI[i]):>9} {str(exp_xII[i]):>10}")
    else:
        print(f"{T_K:6.1f} {'out of range':>10}")

print()
print("=" * 70)
print("KEY FINDING: Why does PD.lle with feed=0.40 reach 270K but feed=0.10 only reaches 406K?")
print("=" * 70)
print()
print("The continuation method starts FROM the initial tp_flash at the feed composition.")
print("The feed determines where on the binodal curve the continuation starts.")
print("With feed=0.40 (in the two-phase region near the middle of the binodal),")
print("it can trace the curve in BOTH directions (toward low T and toward UCST).")
print("With feed=0.10 (near the x1_I branch at high T), it only follows")
print("a short segment near the UCST.")
print()
print("The CORRECT approach for _residuals_at_T is Strategy D:")
print("  1. Call PhaseDiagram.lle ONCE per k_ij evaluation with feed ~0.40")
print("  2. Interpolate the result to each experimental temperature")
print("  3. Avoids cold-start problems entirely")
print()
print("But this means restructuring _residuals_at_T to operate on ALL T at once,")
print("rather than one T at a time.")
